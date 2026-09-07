import copy
import threading
import time
import os
import concurrent.futures
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
import threading

def _alloc_numa_pinned(shape, dtype, numa_node):
    pass
    import ctypes, ctypes.util
    nbytes = 1
    for s in shape: nbytes *= s
    nbytes *= torch.tensor([], dtype=dtype).element_size()
    lib_name = ctypes.util.find_library('numa')
    if lib_name:
        try:
            _numa = ctypes.CDLL(lib_name)
            _numa.numa_alloc_onnode.restype  = ctypes.c_void_p
            _numa.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
            ptr = _numa.numa_alloc_onnode(nbytes, numa_node)
            if ptr:
                buf = torch.frombuffer(
                    (ctypes.c_byte * nbytes).from_address(ptr), dtype=dtype
                ).reshape(shape)
                pinned = buf.pin_memory()
                _numa.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                _numa.numa_free(ptr, nbytes)
                return pinned
        except Exception:
            pass
    return torch.empty(shape, dtype=dtype).pin_memory()

def _rms_norm_bf16(hidden_states, weight, eps=1e-5):

    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states

class InplaceKVCache:
    pass
    def __init__(self, num_layers, batch_size, num_heads, max_seq, k_head_dim, v_head_dim,
                 dtype=torch.bfloat16, b_cpu=None):
        self._nl    = num_layers
        self._B     = batch_size
        self._nH    = num_heads
        self._ms    = max_seq
        self._khd   = k_head_dim
        self._vhd   = v_head_dim
        self._dtype = dtype
        self._bnh   = batch_size * num_heads
        self._b_cpu = b_cpu
        self.key_cache   = [None] * num_layers
        self.value_cache = [None] * num_layers
        self._filled     = [0]    * num_layers

    def _ensure(self, li):
        if self.key_cache[li] is None:
            b_cpu = self._b_cpu
            bkv   = self._bnh
            if b_cpu is not None and 0 < b_cpu < bkv:

                k_c = _alloc_numa_pinned((b_cpu,        self._ms, self._khd), self._dtype, 1)
                k_g = _alloc_numa_pinned((bkv - b_cpu,  self._ms, self._khd), self._dtype, 0)
                v_c = _alloc_numa_pinned((b_cpu,        self._ms, self._vhd), self._dtype, 1)
                v_g = _alloc_numa_pinned((bkv - b_cpu,  self._ms, self._vhd), self._dtype, 0)
                self.key_cache[li]   = (k_c, k_g)
                self.value_cache[li] = (v_c, v_g)
            else:
                self.key_cache[li]   = torch.empty(
                    bkv, self._ms, self._khd, dtype=self._dtype).pin_memory()
                self.value_cache[li] = torch.empty(
                    bkv, self._ms, self._vhd, dtype=self._dtype).pin_memory()

    def _is_split(self, li):
        return isinstance(self.key_cache[li], tuple)

    def append_4d(self, li, k4d, v4d, non_blocking=False):
        pass
        new_seq = k4d.shape[2]
        self._ensure(li)
        t = self._filled[li]
        k_flat = k4d.reshape(self._bnh, new_seq, self._khd)
        v_flat = v4d.reshape(self._bnh, new_seq, self._vhd)
        if self._is_split(li):
            b = self._b_cpu
            k_c, k_g = self.key_cache[li];   v_c, v_g = self.value_cache[li]
            k_c[:, t:t+new_seq, :].copy_(k_flat[:b],  non_blocking=non_blocking)
            k_g[:, t:t+new_seq, :].copy_(k_flat[b:],  non_blocking=non_blocking)
            v_c[:, t:t+new_seq, :].copy_(v_flat[:b],  non_blocking=non_blocking)
            v_g[:, t:t+new_seq, :].copy_(v_flat[b:],  non_blocking=non_blocking)
        else:
            self.key_cache[li][:, t:t+new_seq, :].copy_(k_flat, non_blocking=non_blocking)
            self.value_cache[li][:, t:t+new_seq, :].copy_(v_flat, non_blocking=non_blocking)
        self._filled[li] = t + new_seq

    def get(self, li):
        pass
        t = self._filled[li]
        if self._is_split(li):
            k_c, k_g = self.key_cache[li];   v_c, v_g = self.value_cache[li]
            return (k_c[:, :t, :], k_g[:, :t, :]), (v_c[:, :t, :], v_g[:, :t, :])
        return self.key_cache[li][:, :t, :], self.value_cache[li][:, :t, :]

    def get_4d(self, li):
        pass
        k, v = self.get(li)
        t = self._filled[li]
        if self._is_split(li):
            k_c, k_g = k; v_c, v_g = v
            k = torch.cat([k_c, k_g], dim=0)
            v = torch.cat([v_c, v_g], dim=0)
        return (k.view(self._B, self._nH, t, self._khd),
                v.view(self._B, self._nH, t, self._vhd))

    def seq_len(self, li):
        return self._filled[li]

    def reset(self):
        for i in range(self._nl):
            self._filled[i] = 0

KV_LEN_GS_TABLE = {
    2048: 1280, 4096: 2560, 8192: 4096, 16384: 6144,
    32768: 12288, 65536: 32768, 131072: 65536,
}

def lookup_gs(total_kv):
    pass
    keys = sorted(KV_LEN_GS_TABLE)
    if total_kv <= keys[0]: return KV_LEN_GS_TABLE[keys[0]]
    if total_kv >= keys[-1]: return KV_LEN_GS_TABLE[keys[-1]]
    for i in range(len(keys) - 1):
        k0, k1 = keys[i], keys[i + 1]
        if k0 <= total_kv <= k1:
            g0, g1 = KV_LEN_GS_TABLE[k0], KV_LEN_GS_TABLE[k1]
            return int(g0 + (g1 - g0) * (total_kv - k0) / (k1 - k0))
    return total_kv // 2

class InplaceKVCacheCompressed:
    pass
    def __init__(self, num_layers, batch_size, max_seq, kv_lora_rank, qk_rope_head_dim,
                 dtype=torch.bfloat16, gs=None):
        self._nl  = num_layers
        self._B   = batch_size
        self._ms  = max_seq
        self._clr = kv_lora_rank
        self._qrd = qk_rope_head_dim
        self._dtype = dtype
        self._gs  = gs
        self.cache_c  = [None] * num_layers
        self.cache_pe = [None] * num_layers
        self._filled  = [0]    * num_layers

    def _ensure(self, li):
        if self.cache_c[li] is None:
            if self._gs is not None and 0 < self._gs < self._ms:

                self.cache_c[li]  = (torch.empty(self._B, self._gs, self._clr, dtype=self._dtype).pin_memory(),
                                     torch.empty(self._B, self._ms - self._gs, self._clr, dtype=self._dtype).pin_memory())
                self.cache_pe[li] = (torch.empty(self._B, self._gs, self._qrd, dtype=self._dtype).pin_memory(),
                                     torch.empty(self._B, self._ms - self._gs, self._qrd, dtype=self._dtype).pin_memory())
            else:
                self.cache_c[li]  = torch.empty(self._B, self._ms, self._clr, dtype=self._dtype).pin_memory()
                self.cache_pe[li] = torch.empty(self._B, self._ms, self._qrd, dtype=self._dtype).pin_memory()

    def append(self, li, c_kv, k_pe, non_blocking=False):
        pass
        new_seq = c_kv.shape[1]
        self._ensure(li)
        t = self._filled[li]
        if c_kv.is_cuda: c_kv = c_kv.cpu(); k_pe = k_pe.cpu()
        if isinstance(self.cache_c[li], tuple):
            c_cpu, c_gpu = self.cache_c[li]; p_cpu, p_gpu = self.cache_pe[li]
            gs = self._gs

            t0 = min(t, gs); t1 = min(t + new_seq, gs)
            if t1 > t0:
                c_cpu[:, t0:t1, :].copy_(c_kv[:, t0-t:t1-t, :], non_blocking=non_blocking)
                p_cpu[:, t0:t1, :].copy_(k_pe[:, t0-t:t1-t, :], non_blocking=non_blocking)
            g0 = max(t, gs) - gs; g1 = max(t + new_seq, gs) - gs
            if g1 > g0:
                c_gpu[:, g0:g1, :].copy_(c_kv[:, g0+gs-t:g1+gs-t, :], non_blocking=non_blocking)
                p_gpu[:, g0:g1, :].copy_(k_pe[:, g0+gs-t:g1+gs-t, :], non_blocking=non_blocking)
        else:
            self.cache_c[li][:, t:t+new_seq, :].copy_(c_kv, non_blocking=non_blocking)
            self.cache_pe[li][:, t:t+new_seq, :].copy_(k_pe, non_blocking=non_blocking)
        self._filled[li] = t + new_seq

    def get(self, li):
        pass
        t = self._filled[li]
        if isinstance(self.cache_c[li], tuple):
            c_cpu, c_gpu = self.cache_c[li]; p_cpu, p_gpu = self.cache_pe[li]
            gs = self._gs
            n_cpu = min(t, gs); n_gpu = max(t - gs, 0)
            return (c_cpu[:, :n_cpu, :], c_gpu[:, :n_gpu, :]), (p_cpu[:, :n_cpu, :], p_gpu[:, :n_gpu, :])
        return self.cache_c[li][:, :t, :], self.cache_pe[li][:, :t, :]

    def seq_len(self, li):
        return self._filled[li]

    def reset(self):
        for i in range(self._nl):
            self._filled[i] = 0

def _mla_compressed_forward(self_attn, hidden_states, attention_mask=None,
                            position_ids=None, past_key_value=None,
                            output_attentions=False, use_cache=False,
                            cacheScope_ref=None, **kwargs):
    pass
    import sys as _sys
    _attn_mod = _sys.modules.get('transformers_modules.DeepSeek-V2-Lite.modeling_deepseek')
    if _attn_mod is None:
        raise RuntimeError('_attn_mod not loaded (compressed path needs remote-code module)')
    _rope_fn = _attn_mod.apply_rotary_pos_emb
    bsz, q_len, _ = hidden_states.size()
    li = self_attn.layer_idx
    dev = cacheScope_ref.dev
    _dtype = hidden_states.dtype
    _scale = self_attn.softmax_scale
    _nH = self_attn.num_heads
    _clr = self_attn.kv_lora_rank
    _qrd = self_attn.qk_rope_head_dim
    _nop = self_attn.qk_nope_head_dim
    _vhd = self_attn.v_head_dim

    if self_attn.q_lora_rank is None:
        q = self_attn.q_proj(hidden_states)
    else:
        q = self_attn.q_b_proj(self_attn.q_a_layernorm(self_attn.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, _nH, self_attn.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [_nop, _qrd], dim=-1)
    del q

    compressed_kv = self_attn.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(compressed_kv, [_clr, _qrd], dim=-1)
    c_kv = self_attn.kv_a_layernorm(compressed_kv)
    del compressed_kv
    k_pe = k_pe.view(bsz, q_len, 1, _qrd).transpose(1, 2)

    past_len = cacheScope_ref.inplace_kv.seq_len(li)
    kv_seq_len = past_len + q_len
    cos, sin = self_attn.rotary_emb(k_pe, seq_len=kv_seq_len)
    q_pe, k_pe = _rope_fn(q_pe, k_pe, cos, sin, position_ids)

    cacheScope_ref.inplace_kv.append(li, c_kv, k_pe.squeeze(1), non_blocking=True)
    del c_kv, k_pe

    total_kv = cacheScope_ref.inplace_kv.seq_len(li)

    _kvw = self_attn.kv_b_proj.weight

    _w3  = _kvw.view(_nH, _nop + _vhd, _clr)
    _w_uk = _w3[:, :_nop, :].contiguous()
    _w_uv = _w3[:, _nop:, :].contiguous()

    q_proj_k = torch.bmm(q_nope.reshape(bsz * _nH, q_len, _nop),
                         _w_uk.expand(bsz, _nH, _nop, _clr).reshape(bsz * _nH, _nop, _clr))

    del q_nope

    _kv_ret = cacheScope_ref.inplace_kv.get(li)
    if isinstance(_kv_ret[0], tuple):

        (c_cpu_seg, c_gpu_seg), (p_cpu_seg, p_gpu_seg) = _kv_ret
        c_kv_all = torch.cat([c_cpu_seg, c_gpu_seg], dim=1)
        k_pe_all = torch.cat([p_cpu_seg, p_gpu_seg], dim=1)
    else:
        c_kv_all, k_pe_all = _kv_ret
        c_cpu_seg = c_gpu_seg = p_cpu_seg = p_gpu_seg = None
    c_g = c_kv_all.to(dev)
    p_g = k_pe_all.to(dev)

    if q_len > 1:

        attn_out = torch.zeros(bsz, q_len, _nH * _vhd, device=dev, dtype=_dtype)
        chunk = cacheScope_ref._prefill_chunk_size
        for q_s in range(0, q_len, chunk):
            q_e = min(q_s + chunk, q_len); q_cs = q_e - q_s
            qk = q_proj_k.reshape(bsz * _nH, q_len, _clr)[:, q_s:q_e, :]
            qp = q_pe.reshape(bsz * _nH, q_len, _qrd)[:, q_s:q_e, :]
            _cexp = c_g.reshape(bsz, 1, total_kv, _clr).expand(bsz, _nH, total_kv, _clr).reshape(bsz * _nH, total_kv, _clr)
            _pexp = p_g.reshape(bsz, 1, total_kv, _qrd).expand(bsz, _nH, total_kv, _qrd).reshape(bsz * _nH, total_kv, _qrd)
            out_acc = torch.zeros(bsz * _nH, q_cs, _vhd, device=dev, dtype=torch.float32)
            max_acc = torch.full((bsz * _nH, q_cs, 1), float('-inf'), device=dev, dtype=torch.float32)
            sum_acc = torch.zeros(bsz * _nH, q_cs, 1, device=dev, dtype=torch.float32)
            for kv_s in range(0, total_kv, chunk):
                kv_e = min(kv_s + chunk, total_kv); kvsz = kv_e - kv_s
                c_c = c_g[:, kv_s:kv_e, :]
                p_c = p_g[:, kv_s:kv_e, :]
                sc_n = torch.bmm(qk, _cexp[:, kv_s:kv_e, :].transpose(1, 2))
                sc_p = torch.bmm(qp, _pexp[:, kv_s:kv_e, :].transpose(1, 2))
                scores = (sc_n + sc_p) * _scale
                del sc_n, sc_p
                q_idx = torch.arange(past_len + q_s, past_len + q_e, device=dev).unsqueeze(1)
                kv_idx = torch.arange(kv_s, kv_e, device=dev).unsqueeze(0)
                mask = kv_idx > q_idx
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                del q_idx, kv_idx, mask
                m_c = scores.amax(dim=-1, keepdim=True).to(torch.float32)
                e_c = torch.exp(scores - m_c.to(_dtype))
                s_c = e_c.sum(dim=-1, keepdim=True).to(torch.float32)
                o_c = torch.bmm(e_c.reshape(bsz * _nH, q_cs, kvsz), _cexp[:, kv_s:kv_e, :])
                o_v = torch.bmm(o_c, _w_uv.expand(bsz, _nH, _vhd, _clr).reshape(bsz * _nH, _vhd, _clr).transpose(1, 2))
                del scores, e_c, c_c, p_c
                m_new = torch.maximum(max_acc, m_c)
                w_old = torch.exp(max_acc - m_new)
                w_new = torch.exp(m_c - m_new)
                sum_acc = w_old * sum_acc + w_new * s_c
                out_acc = out_acc * w_old + o_v.to(torch.float32) * w_new
                max_acc = m_new
                del o_c, o_v, m_c, s_c, m_new, w_old, w_new
            sum_acc = sum_acc.clamp(min=1e-6)
            out_norm = (out_acc / sum_acc).to(_dtype)
            out_norm = out_norm.view(bsz, _nH, q_cs, _vhd).transpose(1, 2).contiguous().view(bsz, q_cs, -1)
            attn_out[:, q_s:q_e, :] = self_attn.o_proj(out_norm)
            del out_norm, out_acc, max_acc, sum_acc, qk, qp
    else:

        import threading as _thr

        _cache_gs = getattr(cacheScope_ref.inplace_kv, '_gs', None)
        _ov_gs = getattr(cacheScope_ref, '_kvlen_gs', None)
        if _ov_gs is not None:
            gs = _ov_gs
        else:
            gs = lookup_gs(total_kv)
        if _cache_gs is not None:
            gs = min(gs, _cache_gs)
        gs = max(0, min(gs, total_kv))

        q_proj_k3 = q_proj_k.reshape(bsz * _nH, 1, _clr)
        q_pe3 = q_pe.reshape(bsz * _nH, 1, _qrd)
        _w_uv_t = _w_uv.transpose(1, 2).contiguous()

        cpu_res = {}

        def _cpu_fn():

            qp_cpu = q_proj_k3.view(bsz * _nH, _clr).cpu()
            qpe_cpu = q_pe3.view(bsz * _nH, _qrd).cpu()
            if c_cpu_seg is not None:
                if c_cpu_seg.is_contiguous():
                    c_cpu = c_cpu_seg[:, :gs, :]; p_cpu = p_cpu_seg[:, :gs, :]
                else:
                    c_cpu = c_cpu_seg[:, :gs, :].contiguous(); p_cpu = p_cpu_seg[:, :gs, :].contiguous()
            else:
                c_cpu = c_kv_all[:, :gs, :].contiguous()
                p_cpu = k_pe_all[:, :gs, :].contiguous()
            out_c, mx_c, sm_c = cacheScope_ref._cpu_attn_ext.fused_cpu_attn_mla_compressed(
                qp_cpu, qpe_cpu, c_cpu, p_cpu, _scale)
            cpu_res['o'] = out_c.view(bsz * _nH, 1, _clr)
            cpu_res['m'] = mx_c.view(bsz * _nH, 1, 1)
            cpu_res['s'] = sm_c.view(bsz * _nH, 1, 1)

        cpu_thr = _thr.Thread(target=_cpu_fn, daemon=True)
        cpu_thr.start()

        tg = total_kv - gs
        if tg > 0:
            if c_gpu_seg is not None:

                _gfix = c_cpu_seg.size(1)
                if gs < _gfix:
                    c_gt = torch.cat([c_cpu_seg[:, gs:, :], c_gpu_seg], dim=1).contiguous().to(dev)
                    p_gt = torch.cat([p_cpu_seg[:, gs:, :], p_gpu_seg], dim=1).contiguous().to(dev)
                else:
                    c_gt = c_gpu_seg.to(dev); p_gt = p_gpu_seg.to(dev)
            else:
                c_gt = c_kv_all[:, gs:, :].contiguous().to(dev)
                p_gt = k_pe_all[:, gs:, :].contiguous().to(dev)

            _cexp = c_gt.repeat_interleave(_nH, dim=0)
            _pexp = p_gt.repeat_interleave(_nH, dim=0)
            sc_n = torch.bmm(q_proj_k3, _cexp.transpose(1, 2))
            sc_p = torch.bmm(q_pe3, _pexp.transpose(1, 2))
            scores = (sc_n + sc_p) * _scale
            del sc_n, sc_p
            m_g = scores.amax(dim=-1, keepdim=True)
            e_g = torch.exp(scores - m_g)
            s_g = e_g.sum(dim=-1, keepdim=True)
            o_g = torch.bmm(e_g, _cexp)
            del scores, e_g
        else:
            m_g = torch.full((bsz * _nH, 1, 1), float('-inf'), device=dev, dtype=torch.float32)
            s_g = torch.zeros(bsz * _nH, 1, 1, device=dev, dtype=torch.float32)
            o_g = torch.zeros(bsz * _nH, 1, _clr, device=dev, dtype=torch.float32)

        cpu_thr.join()
        o_c = cpu_res['o'].to(dev); m_c = cpu_res['m'].to(dev); s_c = cpu_res['s'].to(dev)

        M = torch.maximum(m_c, m_g)
        w_c = torch.exp(m_c - M); w_g = torch.exp(m_g - M)
        o_m = w_c * o_c + w_g * o_g
        s_m = w_c * s_c + w_g * s_g
        del o_c, m_c, s_c, o_g, m_g, s_g, w_c, w_g, M

        o_n = (o_m / s_m.clamp(min=1e-6)).to(_dtype)
        del o_m, s_m
        o_v = torch.bmm(o_n, _w_uv_t.expand(bsz, _nH, _clr, _vhd).reshape(bsz * _nH, _clr, _vhd))
        del o_n
        attn_out = o_v.view(bsz, _nH, 1, _vhd).transpose(1, 2).contiguous().view(bsz, 1, _nH * _vhd)
        del o_v
        attn_out = self_attn.o_proj(attn_out)

    del q_proj_k, q_pe, c_g, p_g
    return attn_out, None, past_key_value

class CacheScopeDeepSeekV2:
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.hot_experts = {}
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            use_cache=True,
            trust_remote_code=True,
            local_files_only=True
        )
        self.batch_size = args.batch_size
        self._use_compressed_kv = getattr(args, '_use_compressed_kv', False)
        if self.batch_size == 4:
            self.cache = 5
        elif self.batch_size == 8:
            self.cache = 10
        else:
            self.cache = 15
        self.lm_head = self.model.lm_head
        self.prefil_pre = False
        self.is_decode = False
        self.model = self.model.model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)

        self.n_expert = self.model.config.n_routed_experts
        self.n_shared_experts = 2
        self.expert_selection_stats = []
        self.expert_time_stats = []

        self.expert_selection_history = {}
        self.hit_stats = {}
        for i in range(1, 27):
            if i>0:
                self.expert_selection_history[i] = []
                self.hit_stats[i] = {'hits': 0, 'total': 0}

        self.expert_weight_accumulator = {}
        for i in range(1, 27):
            if i>0:
                self.expert_weight_accumulator[i] = torch.zeros(6, device=self.dev)
        self.cpu_expert_time_per_layer = {i: 0.0 for i in range(1, 27)}

        self.latency_cpu = 5
        self.latency_gpu = 45

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.expert_loc_now = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{(self.n_layer-1) * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)

        self.layer_time_stats = []
        self.layer_time_accumulator = {}
        for i in range(1, 27):
            self.layer_time_accumulator[i] = 0.0
        self.layer_time_details = {
            'all_gpu': [],
            'all_cpu': [],
            'mixed': []
        }

        self.layer_time_accumulator_details = {
            'all_gpu': {i: 0.0 for i in range(1, 27)},
            'all_cpu': {i: 0.0 for i in range(1, 27)},
            'mixed': {i: 0.0 for i in range(1, 27)}
        }
        self.last_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(1, 27)
        }
        self.current_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(1, 27)
        }

        self.layer_data = {}

        tick = time.time()
        self.bring_expert_to_gpu()
        print(f"_")
        print("Model is ready.")
        self.gpuprefil_count=0
        self.cpuprefil_count=0
        self.localprefil_count=0

        _micro_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'microdeepseek.txt')
        with open(_micro_path) as _f:
            self.cpu_time_table = [float(line.strip()) for line in _f]

        import concurrent.futures as _cf
        self._cpu_pool = _cf.ThreadPoolExecutor(max_workers=1)

        self._cpu_expert_pool = _cf.ThreadPoolExecutor(max_workers=8)

        _hidden = self.model.config.hidden_size
        _dummy = torch.zeros(4, _hidden, dtype=self.dtype)
        _warmup_experts = [e for e in range(self.n_expert)
                           if self.expert_loc[1, e] == 0][:2]

        for _cpu_exp in _warmup_experts:
            self._cpu_pool.submit(
                self.model.layers[1].mlp.experts[_cpu_exp], _dummy
            ).result()

        _warmup_futs = [self._cpu_expert_pool.submit(
            self.model.layers[1].mlp.experts[_warmup_experts[i % len(_warmup_experts)]], _dummy
        ) for i in range(8)]
        for _f in _warmup_futs:
            _f.result()

        self._gpu_pool = _cf.ThreadPoolExecutor(max_workers=1)

        self._gpu_experts = [None] + [
            [self.model.layers[i].mlp.experts[j] for j in range(self.n_expert)]
            for i in range(1, self.n_layer)
        ]
        print("_")

        _cfg  = self.model.config
        _nH   = _cfg.num_attention_heads
        _qhd  = _cfg.qk_nope_head_dim + _cfg.qk_rope_head_dim
        _vhd  = _cfg.v_head_dim
        _chunk = 512
        self._kv_stream_h2d  = torch.cuda.Stream()
        self._kv_stream_comp = torch.cuda.Stream()

        self._kv_buf_k = [
            torch.empty(self.batch_size, _nH, _chunk, _qhd, dtype=self.dtype, device=self.dev)
            for _ in range(2)
        ]
        self._kv_buf_v = [
            torch.empty(self.batch_size, _nH, _chunk, _vhd, dtype=self.dtype, device=self.dev)
            for _ in range(2)
        ]

        self._prefill_chunk_size = 256

        self._gp_kv_bufs = {}
        self._gp_stream  = torch.cuda.Stream()

        self._gp_attn_graphs = {}

        try:
            import sys as _sys_ext
            _sys_ext.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from cpu_attn_ext.build import build as _build_cpu_attn
            self._cpu_attn_ext = _build_cpu_attn()
            self._cpu_attn_ext.set_num_threads(16)
        except Exception as _e:
            print(f"_")
            self._cpu_attn_ext = None

        import sys as _sys
        _attn_mod = _sys.modules.get('transformers_modules.DeepSeek-V2-Lite.modeling_deepseek')
        if _attn_mod is not None:
            import types as _types
            import torch.nn.functional as _F
            _orig_forward = _attn_mod.DeepseekV2Attention.forward
            _rope_fn = _attn_mod.apply_rotary_pos_emb
            cacheScope_ref = self

            def _sdpa_forward(self_attn, hidden_states, attention_mask=None,
                               position_ids=None, past_key_value=None,
                               output_attentions=False, use_cache=False, **kwargs):

                if getattr(cacheScope_ref, '_use_compressed_kv', False):
                    return _mla_compressed_forward(
                        self_attn, hidden_states, attention_mask, position_ids,
                        past_key_value, output_attentions, use_cache,
                        cacheScope_ref=cacheScope_ref, **kwargs)
                bsz, q_len, _ = hidden_states.size()
                li = self_attn.layer_idx
                _mem_cb = getattr(cacheScope_ref, '_mem_probe_cb', None)

                if not hasattr(cacheScope_ref, 'inplace_kv'):

                    if self_attn.q_lora_rank is None:
                        q = self_attn.q_proj(hidden_states)
                    else:
                        q = self_attn.q_b_proj(self_attn.q_a_layernorm(self_attn.q_a_proj(hidden_states)))
                    q = q.view(bsz, q_len, self_attn.num_heads, self_attn.q_head_dim).transpose(1, 2)
                    q_nope, q_pe = torch.split(q, [self_attn.qk_nope_head_dim, self_attn.qk_rope_head_dim], dim=-1)
                    compressed_kv = self_attn.kv_a_proj_with_mqa(hidden_states)
                    compressed_kv, k_pe = torch.split(compressed_kv, [self_attn.kv_lora_rank, self_attn.qk_rope_head_dim], dim=-1)
                    k_pe = k_pe.view(bsz, q_len, 1, self_attn.qk_rope_head_dim).transpose(1, 2)
                    kv = (self_attn.kv_b_proj(self_attn.kv_a_layernorm(compressed_kv))
                          .view(bsz, q_len, self_attn.num_heads, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                          .transpose(1, 2))
                    k_nope, value_states = torch.split(kv, [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=-1)
                    kv_seq_len = q_len
                    if past_key_value is not None:
                        kv_seq_len += past_key_value.get_usable_length(q_len, li)
                    cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
                    q_pe, k_pe = _rope_fn(q_pe, k_pe, cos, sin, position_ids)
                    query_states = k_pe.new_empty(bsz, self_attn.num_heads, q_len, self_attn.q_head_dim)
                    query_states[:, :, :, :self_attn.qk_nope_head_dim] = q_nope
                    query_states[:, :, :, self_attn.qk_nope_head_dim:] = q_pe
                    key_states = k_pe.new_empty(bsz, self_attn.num_heads, q_len, self_attn.q_head_dim)
                    key_states[:, :, :, :self_attn.qk_nope_head_dim] = k_nope
                    key_states[:, :, :, self_attn.qk_nope_head_dim:] = k_pe
                    if past_key_value is not None:
                        key_states, value_states = past_key_value.update(
                            key_states, value_states, li, {"sin": sin, "cos": cos})
                    attn_output = _F.scaled_dot_product_attention(
                        query_states, key_states, value_states,
                        attn_mask=None, is_causal=(q_len > 1),
                        dropout_p=self_attn.attention_dropout if self_attn.training else 0.0,
                        scale=self_attn.softmax_scale,
                    )
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, self_attn.num_heads * self_attn.v_head_dim)
                    return self_attn.o_proj(attn_output), None, past_key_value

                if q_len > 1:
                    _chunk  = cacheScope_ref._prefill_chunk_size
                    _scale  = self_attn.softmax_scale
                    _dev    = cacheScope_ref.dev
                    _dtype  = hidden_states.dtype
                    past_len_pf = cacheScope_ref.inplace_kv.seq_len(li)
                    kv_seq_len  = past_len_pf + q_len

                    _cos_ref = hidden_states.new_empty(bsz, 1, 1, self_attn.v_head_dim)
                    cos, sin = self_attn.rotary_emb(_cos_ref, seq_len=kv_seq_len)
                    del _cos_ref

                    attn_output_pf = hidden_states.new_empty(
                        bsz, q_len, self_attn.num_heads * self_attn.v_head_dim)

                    for q_s in range(0, q_len, _chunk):
                        q_e  = min(q_s + _chunk, q_len)
                        q_cs = q_e - q_s
                        hs_c = hidden_states[:, q_s:q_e, :]

                        comp_kv_c = self_attn.kv_a_proj_with_mqa(hs_c)
                        comp_kv_c, k_pe_c = torch.split(
                            comp_kv_c, [self_attn.kv_lora_rank, self_attn.qk_rope_head_dim], dim=-1)
                        kv_c = (self_attn.kv_b_proj(self_attn.kv_a_layernorm(comp_kv_c))
                                .view(bsz, q_cs, self_attn.num_heads,
                                      self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                                .transpose(1, 2))
                        del comp_kv_c
                        k_nope_c, v_c_gpu = torch.split(
                            kv_c, [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=-1)
                        del kv_c

                        k_pe_c = k_pe_c.view(bsz, q_cs, 1, self_attn.qk_rope_head_dim).transpose(1, 2)
                        k_pe_c = k_pe_c.expand(-1, self_attn.num_heads, -1, -1).contiguous()
                        pos_c  = position_ids[:, q_s:q_e] if position_ids is not None else None
                        _, k_pe_c = _rope_fn(k_pe_c, k_pe_c, cos, sin, pos_c)

                        k_nope_cpu = k_nope_c.to('cpu', non_blocking=True); del k_nope_c
                        k_pe_cpu   = k_pe_c.to('cpu',   non_blocking=True); del k_pe_c
                        v_cpu_c    = v_c_gpu.to('cpu',  non_blocking=True); del v_c_gpu
                        torch.cuda.synchronize()
                        k_cpu_c = torch.cat([k_nope_cpu, k_pe_cpu], dim=-1)
                        del k_nope_cpu, k_pe_cpu
                        cacheScope_ref.inplace_kv.append_4d(li, k_cpu_c, v_cpu_c)
                        del k_cpu_c, v_cpu_c

                        if self_attn.q_lora_rank is None:
                            q_c = self_attn.q_proj(hs_c)
                        else:
                            q_c = self_attn.q_b_proj(self_attn.q_a_layernorm(
                                self_attn.q_a_proj(hs_c)))
                        q_c = q_c.view(bsz, q_cs, self_attn.num_heads,
                                       self_attn.q_head_dim).transpose(1, 2)
                        q_nope_c, q_pe_c = torch.split(
                            q_c, [self_attn.qk_nope_head_dim, self_attn.qk_rope_head_dim], dim=-1)
                        del q_c
                        dummy_k = torch.zeros(bsz, 1, q_cs, self_attn.qk_rope_head_dim,
                                              device=_dev, dtype=_dtype)
                        q_pe_c, _ = _rope_fn(q_pe_c, dummy_k, cos, sin, pos_c)
                        del dummy_k
                        q_chunk = q_pe_c.new_empty(bsz, self_attn.num_heads, q_cs,
                                                   self_attn.q_head_dim)
                        q_chunk[:, :, :, :self_attn.qk_nope_head_dim] = q_nope_c
                        q_chunk[:, :, :, self_attn.qk_nope_head_dim:] = q_pe_c
                        del q_nope_c, q_pe_c

                        k_cache_flat, v_cache_flat = cacheScope_ref.inplace_kv.get(li)
                        if isinstance(k_cache_flat, tuple):

                            k_cache_flat = torch.cat(k_cache_flat, dim=0)
                            v_cache_flat = torch.cat(v_cache_flat, dim=0)
                        total_kv = cacheScope_ref.inplace_kv.seq_len(li)

                        out_acc = torch.zeros(bsz, self_attn.num_heads, q_cs, self_attn.v_head_dim,
                                              device=_dev, dtype=torch.float32)
                        max_acc = torch.full((bsz, self_attn.num_heads, q_cs, 1),
                                             float('-inf'), device=_dev, dtype=torch.float32)
                        sum_acc = torch.zeros(bsz, self_attn.num_heads, q_cs, 1,
                                              device=_dev, dtype=torch.float32)

                        for kv_s in range(0, total_kv, _chunk):
                            kv_e  = min(kv_s + _chunk, total_kv)
                            kvsz  = kv_e - kv_s
                            k_c = (k_cache_flat[:, kv_s:kv_e, :]
                                   .to(_dev)
                                   .view(bsz, self_attn.num_heads, kvsz, self_attn.q_head_dim))
                            v_c = (v_cache_flat[:, kv_s:kv_e, :]
                                   .to(_dev)
                                   .view(bsz, self_attn.num_heads, kvsz, self_attn.v_head_dim))

                            scores = torch.matmul(q_chunk, k_c.transpose(-2, -1)) * _scale
                            q_idx  = torch.arange(past_len_pf + q_s,
                                                  past_len_pf + q_e, device=_dev).unsqueeze(1)
                            kv_idx = torch.arange(kv_s, kv_e, device=_dev).unsqueeze(0)
                            mask   = kv_idx > q_idx
                            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                            del q_idx, kv_idx, mask

                            m_c  = scores.amax(dim=-1, keepdim=True).to(torch.float32)
                            e_c  = torch.exp(scores - m_c.to(_dtype))
                            s_c  = e_c.sum(dim=-1, keepdim=True).to(torch.float32)
                            o_c  = torch.matmul(e_c, v_c)

                            m_new = torch.maximum(max_acc, m_c)
                            w_old = torch.exp(max_acc - m_new)
                            w_new = torch.exp(m_c - m_new)
                            sum_acc = w_old * sum_acc + w_new * s_c
                            out_acc = out_acc * w_old + o_c.to(torch.float32) * w_new
                            max_acc = m_new
                            del k_c, v_c, scores, m_c, e_c, s_c, o_c, m_new, w_old, w_new

                        sum_acc = sum_acc.clamp(min=1e-6)
                        out_norm = (out_acc / sum_acc).to(_dtype)
                        del out_acc, max_acc, sum_acc, q_chunk

                        out_norm = out_norm.transpose(1, 2).contiguous().view(bsz, q_cs, -1)
                        attn_output_pf[:, q_s:q_e, :] = self_attn.o_proj(out_norm)
                        del out_norm, k_cache_flat, v_cache_flat

                    return attn_output_pf, None, past_key_value

                compressed_kv = self_attn.kv_a_proj_with_mqa(hidden_states)
                compressed_kv, k_pe = torch.split(
                    compressed_kv, [self_attn.kv_lora_rank, self_attn.qk_rope_head_dim], dim=-1)
                k_pe = k_pe.view(bsz, 1, 1, self_attn.qk_rope_head_dim).transpose(1, 2)
                kv = (self_attn.kv_b_proj(self_attn.kv_a_layernorm(compressed_kv))
                      .view(bsz, 1, self_attn.num_heads, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                      .transpose(1, 2))
                del compressed_kv
                k_nope, value_states = torch.split(kv, [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=-1)
                del kv

                if self_attn.q_lora_rank is None:
                    q = self_attn.q_proj(hidden_states)
                else:
                    q = self_attn.q_b_proj(self_attn.q_a_layernorm(self_attn.q_a_proj(hidden_states)))
                q = q.view(bsz, 1, self_attn.num_heads, self_attn.q_head_dim).transpose(1, 2)
                q_nope, q_pe = torch.split(q, [self_attn.qk_nope_head_dim, self_attn.qk_rope_head_dim], dim=-1)
                del q

                past_len = cacheScope_ref.inplace_kv.seq_len(li)
                kv_seq_len = past_len + 1
                cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
                q_pe, k_pe = _rope_fn(q_pe, k_pe, cos, sin, position_ids)
                query_states = k_pe.new_empty(bsz, self_attn.num_heads, 1, self_attn.q_head_dim)
                query_states[:, :, :, :self_attn.qk_nope_head_dim] = q_nope
                query_states[:, :, :, self_attn.qk_nope_head_dim:] = q_pe
                key_states   = k_pe.new_empty(bsz, self_attn.num_heads, 1, self_attn.q_head_dim)
                key_states[:, :, :, :self_attn.qk_nope_head_dim] = k_nope
                key_states[:, :, :, self_attn.qk_nope_head_dim:] = k_pe
                del q_nope, q_pe, k_nope, k_pe
                if _mem_cb: _mem_cb(li, 'qk_built')

                cacheScope_ref.inplace_kv.append_4d(li, key_states, value_states, non_blocking=True)
                del key_states, value_states
                if _mem_cb: _mem_cb(li, 'kv_del')

                total_kv = cacheScope_ref.inplace_kv.seq_len(li)

                if total_kv <= 512:

                    k4d, v4d = cacheScope_ref.inplace_kv.get_4d(li)
                    full_k = k4d.to(cacheScope_ref.dev, non_blocking=False)
                    if _mem_cb: _mem_cb(li, 'k_h2d')
                    full_v = v4d.to(cacheScope_ref.dev, non_blocking=False)
                    if _mem_cb: _mem_cb(li, 'kv_h2d')
                    attn_output = _F.scaled_dot_product_attention(
                        query_states, full_k, full_v,
                        attn_mask=None, is_causal=False,
                        dropout_p=self_attn.attention_dropout if self_attn.training else 0.0,
                        scale=self_attn.softmax_scale,
                    )
                    del full_k, full_v
                    if _mem_cb: _mem_cb(li, 'sdpa_done')
                else:

                    import threading as _thr
                    _chunk  = 512
                    _scale  = self_attn.softmax_scale
                    _nH     = self_attn.num_heads
                    _qhd    = self_attn.q_head_dim
                    _vhd    = self_attn.v_head_dim
                    _dev    = cacheScope_ref.dev
                    _dtype  = hidden_states.dtype
                    bkv     = bsz * _nH

                    kv_ret = cacheScope_ref.inplace_kv.get(li)
                    if isinstance(kv_ret[0], tuple):

                        (kv_flat_k_cpu, kv_flat_k_gpu), (kv_flat_v_cpu, kv_flat_v_gpu) = kv_ret
                        _kv_split = True
                    else:
                        kv_flat_k, kv_flat_v = kv_ret
                        _kv_split = False

                    _ratio_ov = getattr(cacheScope_ref, '_gp_ratio_override', None)
                    if _ratio_ov is not None:
                        _ratio = _ratio_ov
                        b_cpu = max(0, min(bkv, round(bkv * _ratio)))
                        if _kv_split:
                            b_cpu = min(b_cpu, cacheScope_ref.inplace_kv._b_cpu)
                    else:

                        if bsz >= 32:
                            if total_kv <= 2048:    _ratio, _kv_th = 0.25,   2048
                            elif total_kv <= 4096:  _ratio, _kv_th = 0.375,  4096
                            else:                   _ratio, _kv_th = 0.625,  4096
                        elif bsz >= 16:
                            if total_kv <= 4096:    _ratio, _kv_th = 0.156,  4096
                            elif total_kv <= 8192:  _ratio, _kv_th = 0.3125, 8192
                            elif total_kv <= 16384: _ratio, _kv_th = 0.4375, 16384
                            else:                   _ratio, _kv_th = 0.625,  16384
                        elif bsz >= 8:
                            if total_kv <= 8192:    _ratio, _kv_th = 0.156,  8192
                            elif total_kv <= 16384: _ratio, _kv_th = 0.3125, 16384
                            else:                   _ratio, _kv_th = 0.625,  16384
                        elif bsz >= 4:
                            _ratio  = 0.0625 if total_kv <= 16384 else 0.625
                            _kv_th  = 16384
                        else:
                            _ratio, _kv_th = 0.625, 32768
                        b_cpu = max(0, min(bkv, round(bkv * _ratio)) if total_kv >= _kv_th else 0)

                    if _kv_split:
                        b_cpu = min(b_cpu, cacheScope_ref.inplace_kv._b_cpu)
                    b_gpu = bkv - b_cpu

                    if b_cpu == 0:
                        if _kv_split:
                            k_all = torch.cat([kv_flat_k_cpu, kv_flat_k_gpu], dim=0).to(_dev)
                            v_all = torch.cat([kv_flat_v_cpu, kv_flat_v_gpu], dim=0).to(_dev)
                        else:
                            k_all = kv_flat_k.to(_dev); v_all = kv_flat_v.to(_dev)
                            del kv_flat_k, kv_flat_v
                        k_4d = k_all.view(bsz, _nH, total_kv, _qhd)
                        v_4d = v_all.view(bsz, _nH, total_kv, _vhd)
                        del k_all, v_all
                        attn_output = _F.scaled_dot_product_attention(
                            query_states, k_4d, v_4d,
                            attn_mask=None, is_causal=False,
                            scale=self_attn.softmax_scale,
                        )
                        del k_4d, v_4d
                        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, 1, _nH * _vhd)
                        attn_output = self_attn.o_proj(attn_output)
                        return attn_output, None, past_key_value

                    if _kv_split:
                        _b_cpu_buf = cacheScope_ref.inplace_kv._b_cpu
                        _k_cpu_part = kv_flat_k_cpu[:b_cpu]
                        _v_cpu_part = kv_flat_v_cpu[:b_cpu]
                        if b_cpu == _b_cpu_buf:

                            _k_gpu_part = kv_flat_k_gpu
                            _v_gpu_part = kv_flat_v_gpu
                        else:

                            _k_gpu_part = torch.cat([kv_flat_k_cpu[b_cpu:], kv_flat_k_gpu], dim=0)
                            _v_gpu_part = torch.cat([kv_flat_v_cpu[b_cpu:], kv_flat_v_gpu], dim=0)
                    else:
                        _k_cpu_part = kv_flat_k[:b_cpu]
                        _v_cpu_part = kv_flat_v[:b_cpu]
                        _k_gpu_part = kv_flat_k[b_cpu:]
                        _v_gpu_part = kv_flat_v[b_cpu:]

                    q_g = query_states.view(bkv, 1, _qhd)

                    _cpu_res = {}
                    q_cpu_t    = q_g[:b_cpu].cpu()
                    k_cpu_part = _k_cpu_part
                    v_cpu_part = _v_cpu_part

                    def _cpu_fn():

                        try:
                            os.sched_setaffinity(0, list(range(24, 48)) + list(range(72, 96)))
                        except Exception:
                            pass

                        _ext = cacheScope_ref._cpu_attn_ext
                        if _ext is not None:
                            q_c = q_cpu_t.contiguous()
                            out_unnorm, _, lse_sum = _ext.fused_cpu_attn_mla(
                                q_c, k_cpu_part, v_cpu_part, _scale)
                            _cpu_res['out'] = (out_unnorm / lse_sum.clamp(min=1e-6)).bfloat16()
                        else:

                            torch.set_num_threads(16)
                            out_c = torch.zeros(b_cpu, 1, _vhd, dtype=torch.bfloat16)
                            max_c = torch.full((b_cpu, 1, 1), float('-inf'), dtype=torch.bfloat16)
                            sum_c = torch.zeros(b_cpu, 1, 1, dtype=torch.bfloat16)
                            for cs in range(0, total_kv, _chunk):
                                ce  = min(cs + _chunk, total_kv)
                                k_f = k_cpu_part[:, cs:ce, :].contiguous()
                                v_f = v_cpu_part[:, cs:ce, :]
                                sc  = torch.bmm(q_cpu_t, k_f.transpose(1, 2)) * _scale
                                m_n = sc.amax(dim=-1, keepdim=True)
                                e_n = torch.exp(sc - m_n)
                                s_n = e_n.sum(dim=-1, keepdim=True)
                                o_n = torch.bmm(e_n, v_f)
                                mx  = torch.max(max_c, m_n)
                                wc  = torch.exp(max_c - mx)
                                wn  = torch.exp(m_n - mx)
                                sum_c = wc * sum_c + wn * s_n
                                out_c = out_c * wc + o_n * wn
                                max_c = mx
                            _cpu_res['out'] = out_c / sum_c.clamp(min=1e-6)

                    cpu_thr = _thr.Thread(target=_cpu_fn, daemon=True)
                    cpu_thr.start()

                    _use_sdpa = (total_kv <= 16384)
                    if _use_sdpa:
                        k_gpu_f = _k_gpu_part.to(_dev)
                        v_gpu_f = _v_gpu_part.to(_dev)

                        q_sdpa = q_g[b_cpu:].unsqueeze(1)
                        k_sdpa = k_gpu_f.unsqueeze(1)
                        v_sdpa = v_gpu_f.view(b_gpu, 1, total_kv, _vhd)
                        out_sdpa = _F.scaled_dot_product_attention(
                            q_sdpa, k_sdpa, v_sdpa, scale=self_attn.softmax_scale)
                        out_gpu_norm = out_sdpa.squeeze(1).squeeze(1).to(_dtype)
                        del k_gpu_f, v_gpu_f, q_sdpa, k_sdpa, v_sdpa, out_sdpa
                    else:

                        _gp_k = _k_gpu_part
                        _gp_v = _v_gpu_part
                        q_gpu = q_g[b_cpu:]

                        buf_key = (b_gpu,)
                        if buf_key not in cacheScope_ref._gp_kv_bufs:
                            cacheScope_ref._gp_kv_bufs[buf_key] = {
                                'k': [torch.empty(b_gpu, _chunk, _qhd, dtype=_dtype, device=_dev),
                                      torch.empty(b_gpu, _chunk, _qhd, dtype=_dtype, device=_dev)],
                                'v': [torch.empty(b_gpu, _chunk, _vhd, dtype=_dtype, device=_dev),
                                      torch.empty(b_gpu, _chunk, _vhd, dtype=_dtype, device=_dev)],
                            }
                        buf_k = cacheScope_ref._gp_kv_bufs[buf_key]['k']
                        buf_v = cacheScope_ref._gp_kv_bufs[buf_key]['v']
                        _h2d  = cacheScope_ref._gp_stream

                        _g_key = (b_gpu, _chunk)
                        if _g_key not in cacheScope_ref._gp_attn_graphs:
                            cacheScope_ref._gp_attn_graphs[_g_key] =\
                                cacheScope_ref._build_gp_attn_graph(
                                    b_gpu, _chunk, _qhd, _vhd, _scale, _dev)
                        _st = cacheScope_ref._gp_attn_graphs[_g_key]
                        _use_graph = True
                        _st['q'].copy_(q_gpu)
                        _st['out'].zero_()
                        _st['max'].fill_(float('-inf'))
                        _st['sum'].zero_()

                        n_chunks = (total_kv + _chunk - 1) // _chunk
                        s0 = min(_chunk, total_kv)
                        with torch.cuda.stream(_h2d):
                            buf_k[0][:, :s0, :].copy_(_gp_k[:, :s0, :], non_blocking=True)
                            buf_v[0][:, :s0, :].copy_(_gp_v[:, :s0, :], non_blocking=True)

                        for ci in range(n_chunks):
                            cs = ci * _chunk; ce = min(cs + _chunk, total_kv); csz = ce - cs
                            cur = ci % 2; nxt = 1 - cur
                            torch.cuda.current_stream().wait_stream(_h2d)
                            if ci + 1 < n_chunks:
                                ns = (ci + 1) * _chunk; ne = min(ns + _chunk, total_kv); nsz = ne - ns
                                with torch.cuda.stream(_h2d):
                                    buf_k[nxt][:, :nsz, :].copy_(_gp_k[:, ns:ne, :], non_blocking=True)
                                    buf_v[nxt][:, :nsz, :].copy_(_gp_v[:, ns:ne, :], non_blocking=True)
                            k_c = buf_k[cur][:, :csz, :]
                            v_c = buf_v[cur][:, :csz, :]
                            if _use_graph and csz == _chunk:
                                _st['k'].copy_(k_c)
                                _st['v'].copy_(v_c)
                                _st['graph'].replay()
                            else:
                                sc  = torch.bmm(q_gpu, k_c.transpose(1, 2)) * _scale
                                m_c = sc.amax(dim=-1, keepdim=True).to(torch.float32)
                                e_c = torch.exp(sc.to(torch.float32) - m_c)
                                s_c = e_c.sum(dim=-1, keepdim=True)
                                o_c = torch.bmm(e_c.to(_dtype), v_c)
                                mx  = torch.maximum(_st['max'], m_c)
                                wg  = torch.exp(_st['max'] - mx)
                                wn  = torch.exp(m_c - mx)
                                _st['sum'].mul_(wg).add_(wn * s_c)
                                _st['out'].mul_(wg).add_(o_c.float() * wn)
                                _st['max'].copy_(mx)
                                del sc, m_c, e_c, s_c, o_c, mx, wg, wn
                        out_gpu_norm = (_st['out'] / _st['sum'].clamp(min=1e-6)).to(_dtype)
                        del _gp_k, _gp_v

                    cpu_thr.join()
                    if not _kv_split:
                        del kv_flat_k, kv_flat_v
                    out_cpu_d = _cpu_res['out'].to(_dev)

                    if _use_sdpa:
                        out_gpu_norm = out_gpu_norm.unsqueeze(1)
                    out_all = torch.cat([out_cpu_d, out_gpu_norm], dim=0)
                    del out_cpu_d, out_gpu_norm

                    attn_output = (out_all.view(bsz, _nH, 1, _vhd)
                                   .transpose(1, 2).contiguous()
                                   .view(bsz, 1, _nH * _vhd))
                    del out_all
                    attn_output = self_attn.o_proj(attn_output)
                    return attn_output, None, past_key_value
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, 1, self_attn.num_heads * self_attn.v_head_dim)
                attn_output = self_attn.o_proj(attn_output)
                return attn_output, None, past_key_value

            _attn_mod.DeepseekV2Attention.forward = _sdpa_forward
            print("_")

    def _build_gp_attn_graph(self, b_gpu, chunk, qhd, vhd, scale, dev):
        pass
        q_s   = torch.zeros(b_gpu, 1,     qhd, dtype=torch.bfloat16, device=dev)
        k_s   = torch.zeros(b_gpu, chunk, qhd, dtype=torch.bfloat16, device=dev)
        v_s   = torch.zeros(b_gpu, chunk, vhd, dtype=torch.bfloat16, device=dev)
        out_s = torch.zeros(b_gpu, 1,     vhd, dtype=torch.float32,  device=dev)
        max_s = torch.full((b_gpu, 1, 1), float('-inf'), dtype=torch.float32, device=dev)
        sum_s = torch.zeros(b_gpu, 1, 1, dtype=torch.float32, device=dev)

        for _ in range(3):
            sc = torch.bmm(q_s, k_s.transpose(1, 2)) * scale
            m_c = sc.amax(dim=-1, keepdim=True).float()
            e_c = torch.exp(sc.float() - m_c)
            s_c = e_c.sum(dim=-1, keepdim=True)
            o_c = torch.bmm(e_c.bfloat16(), v_s)
            mx = torch.maximum(max_s, m_c)
            wa = torch.exp(max_s - mx); wn = torch.exp(m_c - mx)
            sum_s.mul_(wa).add_(wn * s_c)
            out_s.mul_(wa).add_(o_c.float() * wn)
            max_s.copy_(mx)
        torch.cuda.synchronize(dev)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sc = torch.bmm(q_s, k_s.transpose(1, 2)) * scale
            m_c = sc.amax(dim=-1, keepdim=True).float()
            e_c = torch.exp(sc.float() - m_c)
            s_c = e_c.sum(dim=-1, keepdim=True)
            o_c = torch.bmm(e_c.bfloat16(), v_s)
            mx = torch.maximum(max_s, m_c)
            wa = torch.exp(max_s - mx); wn = torch.exp(m_c - mx)
            sum_s.mul_(wa).add_(wn * s_c)
            out_s.mul_(wa).add_(o_c.float() * wn)
            max_s.copy_(mx)

        return {'q': q_s, 'k': k_s, 'v': v_s,
                'out': out_s, 'max': max_s, 'sum': sum_s, 'graph': g}

    def bring_non_expert_to_gpu(self):
        pass
        first_layer_mlp = self.model.layers[1].mlp
        self.expert_placeholder = copy.deepcopy(
            first_layer_mlp.experts[0]
        ).to(self.dev)
        self.expert_placeholder2 = copy.deepcopy(
            first_layer_mlp.experts[1]
        ).to(self.dev)
        self.expert_placeholder3 = copy.deepcopy(
            first_layer_mlp.experts[2]
        ).to(self.dev)
        self.expert_placeholder4 = copy.deepcopy(
            first_layer_mlp.experts[3]
        ).to(self.dev)
        self.expert_placeholder5 = copy.deepcopy(
            first_layer_mlp.experts[4]
        ).to(self.dev)
        self.expert_placeholder6 = copy.deepcopy(
            first_layer_mlp.experts[5]
        ).to(self.dev)
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        self.model.layers[0].to(self.dev)
        for i in range(len(self.model.layers)):
            if i!=0:
                self.model.layers[i].self_attn.to(self.dev)
                self.model.layers[i].input_layernorm.to(self.dev)
                self.model.layers[i].mlp.gate.to(self.dev)
                self.model.layers[i].post_attention_layernorm.to(self.dev)
        for i in range(1, 27):
            if i>0:
                self.model.layers[i].mlp.shared_experts.to(self.dev)
    def get_hot_expert(self):
        pass
        if not hasattr(self, 'is_decode') or not self.is_decode:
            return {}

        hot_experts = {}

        for layer_id in range(self.n_layer):

            if layer_id>0:
                expert_ids = self.current_iter_expert_stats[layer_id]['expert_ids']
                token_counts = self.current_iter_expert_stats[layer_id]['token_counts']

                expert_data = list(zip(expert_ids, token_counts))

                sorted_experts = sorted(expert_data, key=lambda x: x[1], reverse=True)

                hot_experts[layer_id] = [expert[0] for expert in sorted_experts]

                self.last_iter_expert_stats[layer_id] = {
                    'expert_ids': expert_ids.copy(),
                    'token_counts': token_counts.copy()
                }

                self.current_iter_expert_stats[layer_id]['expert_ids'].clear()
                self.current_iter_expert_stats[layer_id]['token_counts'].clear()

        self.hot_experts = hot_experts
        return hot_experts
    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        pass

        if popular_experts is None:
            hot_experts_file = './hot/deep.txt'
            if os.path.exists(hot_experts_file):
                try:
                    with open(hot_experts_file, 'r') as f:
                        popular_experts = [tuple(map(int, line.strip().split(',')))
                                         for line in f if line.strip()]
                    print(f"Loaded hot experts from {hot_experts_file}")
                except Exception as e:
                    print(f"Error loading hot experts: {e}")
            else:
                popular_experts = []
                for layer in range(1, 27):
                    for expert in range(40):
                        popular_experts.append((layer, expert))
        n_expert_on_gpu = min(n_expert_on_gpu, len(popular_experts))
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]

            self.expert_loc[i_layer, i_expert] = 1

    def _async_ondemand(self, layer_idx, expert_id, target_placeholder):
        expert = self.model.layers[layer_idx].mlp.experts[expert_id]

        if next(expert.parameters()).is_cuda:
            return

        for name in ['gate_proj', 'up_proj', 'down_proj']:
            w = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name)
            src_weight_data_tensor = w.weight.data
            pinned = src_weight_data_tensor.pin_memory()
            w.weight.data = pinned

        tick = time.time()
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            dst = getattr(target_placeholder, name).weight.data
            src = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name).weight.data
            dst.copy_(src)

        copytime = time.time() - tick
        print(f"_")

    def is_expert_loading(self, placeholder_name):
        pass
        return self.expert_loading_status.get(placeholder_name, False)
    def is_expert_loaded(self, layer_id, expert_id):
        return False
    def bring_expert_to_gpu(self):
        pass
        expert_count = 0
        try:
            for i in range(self.n_layer):
                for j in range(self.n_expert):
                    if self.is_expert_in_gpu(i, j):
                        self.model.layers[i].mlp.experts[j].to(self.dev)
                        expert_count += 1

            with open('test.txt', 'a') as f:
                f.write(f"_")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():

                with open('test.txt', 'a') as f:
                    f.write(f"_")
                raise
            else:
                raise

    def is_expert_in_gpu(self, i_layer, i_expert):
        pass
        return self.expert_loc[i_layer, i_expert] == 1
    def is_expert_in_gpu_now(self, i_layer, i_expert):
        pass

        expert = self.model.layers[i_layer].mlp.experts[i_expert]
        return next(expert.parameters()).is_cuda

    def calc_n_expert_on_gpu(self):
        pass
        fine_expert = self.model.layers[1].mlp.experts[0]
        n_param = sum(p.numel() for p in fine_expert.parameters())
        print(f"Number of parameters in a single expert: {n_param}")

        expert_mem_gb = (n_param * 693 * 2) / (1024 ** 3)
        print(f"693 experts require approximately {expert_mem_gb:.2f} GB of GPU memory")
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev)

        return 593

    def initial_beam_tensor(self, input_tensor):

        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        torch.set_num_threads(16)
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0

        _cfg    = self.model.config
        _max_s  = (input_token or 2048) + output_token
        _nH     = _cfg.num_attention_heads
        _qhd    = _cfg.qk_nope_head_dim + _cfg.qk_rope_head_dim
        _vhd    = _cfg.v_head_dim
        _B      = self.batch_size
        _nl     = self.n_layer
        if getattr(self, '_use_compressed_kv', False):

            _mcfg = self.model.config
            self.inplace_kv = InplaceKVCacheCompressed(
                num_layers=_nl, batch_size=_B, max_seq=_max_s,
                kv_lora_rank=_mcfg.kv_lora_rank,
                qk_rope_head_dim=_mcfg.qk_rope_head_dim,
                dtype=self.dtype,
                gs=(getattr(self, '_kvlen_gs', None) if getattr(self, '_kvlen_gs', None) is not None
                    else _max_s // 2))
        else:
            self.inplace_kv = InplaceKVCache(
                num_layers=_nl, batch_size=_B, num_heads=_nH,
                max_seq=_max_s, k_head_dim=_qhd, v_head_dim=_vhd,
                dtype=self.dtype,

                b_cpu=(max(1, min(_B * _nH - 1, int(_B * _nH * 0.625))) if _B >= 16 else None),
            )

        self.inplace_kv_len = self.inplace_kv._filled

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 1

        self.expert_selection_stats = []
        self.expert_time_stats = []
        if text is None:
            text = ["default input"] * self.batch_size
        elif isinstance(text, str):
            text = [text] * self.batch_size

        input_ids, position_ids, attention_mask = self.tokenize(text,input_token)

        if input_token is not None:

            input_ids = torch.stack([
                ids[:input_token] if len(ids) > input_token else ids
                for ids in input_ids
            ])
            position_ids = torch.stack([
                pos[:input_token] if len(pos) > input_token else pos
                for pos in position_ids
            ])

        tick = time.time()
        self.is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)
        self.token_decode_times = []
        self.perf_stats = {
            'token_embedding': [],
            'self_attention': [],
            'moe_gating': [],
            'expert_compute': [],
            'expert_compute-cpu': []
        }

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=3,
                active=1,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )

        for i_token in range(output_token):

            token_start_time = time.time()

            if self.is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])

            if self.is_decode:

                total_len = self.past_key_values_length + 1
                attention_mask = torch.zeros(
                    (input_ids.shape[0], 1, 1, total_len),
                    dtype=torch.bfloat16,
                    device=self.dev
                )
            else:

                pass
            new_position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
            logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask )

            logits = logits.to("cpu")

            logits = F.softmax(logits, dim=-1)

            self.past_key_values_length += logits.shape[1]
            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True

            probs = probs * new_probs

            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .view(-1, 1)
            )
            token_time = time.time() - token_start_time
            self.token_decode_times.append(token_time)

            if not self.is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            self.is_decode = True
        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all,
            {
                'perf_stats': self.perf_stats,
                'expert_selection': self.expert_selection_stats,
                'expert_time': self.expert_time_stats,
                'layer_time': self.layer_time_stats,
                'outputs': decode_strings,

                'expert_hot_stats': self.get_expert_stats(),

            }
        )
    def tokenize(self, text, input_token):
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text should be str or list of str")
        print(f"_")

        if len(text) < self.batch_size:
            text = text + [text[-1]] * (self.batch_size - len(text))
        elif len(text) > self.batch_size:
            text = text[:self.batch_size]

        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=input_token,
            return_tensors="pt"
        )
        input_ids = encodings.input_ids.to(self.dev)
        attention_mask = encodings.attention_mask.bool().to(self.dev)

        seq_length = input_ids.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.dev
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        if attention_mask.dim() == 2:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = torch.tril(
                torch.ones(seq_length, seq_length, dtype=torch.bool, device=self.dev)
            ).unsqueeze(0).unsqueeze(0)
            bool_mask = causal_mask & pad_mask
            attention_mask = torch.zeros_like(bool_mask, dtype=torch.bfloat16)
            attention_mask = attention_mask.masked_fill(~bool_mask, -1e4)
        return input_ids, position_ids, attention_mask

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, attention_mask ):
        hidden_dim = self.model.config.hidden_size
        tick = time.time()

        inps = self.model.embed_tokens(input_ids)
        self.perf_stats['token_embedding'].append(time.time() - tick)

        if self.is_decode:
            total_decode_start = time.time()
            layer_times = {i: 0.0 for i in range(1, 27)}
            layer_times_fwd = {i: 0.0 for i in range(1, 27)}
            layer_times_mid = {i: 0.0 for i in range(1, 27)}
            layer_times_final = {i: 0.0 for i in range(1, 27)}

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        layer_start_time = time.time()
        layer_total_time = 0.0
        isprefetch=False
        localcount=0

        for i_layer, layer in enumerate(self.model.layers):
            layer_tick = time.time()
            if i_layer == 0:

                inps_residual0 = inps
                inps = _rms_norm_bf16(inps, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
                inps = inps.view(batch_size, seq_len, hidden_dim)
                if not self.is_decode:
                    if (0 < len(self.past_key_value.key_cache) and
                            self.past_key_value.key_cache[0] is not None):
                        self.past_key_value.key_cache[0] = self.past_key_value.key_cache[0].to(self.dev)
                        self.past_key_value.value_cache[0] = self.past_key_value.value_cache[0].to(self.dev)
                attn_out0, _, _ = layer.self_attn(
                    hidden_states=inps,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=self.past_key_value,
                    use_cache=True,
                )
                if not self.is_decode and (0 < len(self.past_key_value.key_cache) and
                        self.past_key_value.key_cache[0] is not None):
                    self.past_key_value.key_cache[0] = self.past_key_value.key_cache[0].cpu()
                    self.past_key_value.value_cache[0] = self.past_key_value.value_cache[0].cpu()
                inps = inps_residual0 + attn_out0
                inps = inps + layer.mlp(_rms_norm_bf16(inps, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon))
            else:

                original_inps_shape = inps.shape
                self.cpu_expert_time_per_layer[i_layer] =0
                inps_residual = inps
                inps = _rms_norm_bf16(inps, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)

                inps = inps.view(batch_size, seq_len, hidden_dim)
                tick = time.time()

                if self.past_key_value is None:
                    from transformers.cache_utils import DynamicCache
                    self.past_key_value = DynamicCache()

                if not self.is_decode:
                    if (i_layer < len(self.past_key_value.key_cache) and
                            self.past_key_value.key_cache[i_layer] is not None):
                        self.past_key_value.key_cache[i_layer] =\
                            self.past_key_value.key_cache[i_layer].to(self.dev)
                        self.past_key_value.value_cache[i_layer] =\
                            self.past_key_value.value_cache[i_layer].to(self.dev)

                attn_out, _, _ = layer.self_attn(
                    hidden_states=inps,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=self.past_key_value,
                    use_cache=True,
                )
                self.perf_stats['self_attention'].append(time.time() - tick)

                if not self.is_decode:
                    if (i_layer < len(self.past_key_value.key_cache) and
                            self.past_key_value.key_cache[i_layer] is not None):
                        _kc = self.past_key_value.key_cache[i_layer]
                        _vc = self.past_key_value.value_cache[i_layer]
                        _kc_pin = torch.empty(_kc.shape, dtype=_kc.dtype, device='cpu', pin_memory=True)
                        _vc_pin = torch.empty(_vc.shape, dtype=_vc.dtype, device='cpu', pin_memory=True)
                        _kc_pin.copy_(_kc, non_blocking=True)
                        _vc_pin.copy_(_vc, non_blocking=True)
                        self.past_key_value.key_cache[i_layer] = _kc_pin
                        self.past_key_value.value_cache[i_layer] = _vc_pin

                inps_residual.add_(attn_out)
                inps = inps_residual
                inps = _rms_norm_bf16(inps, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon)

                layer_idx=i_layer
                if layer_idx not in self.layer_data:
                    self.layer_data[layer_idx] = {
                        "hidden_states": [],
                        "expert_indices": []
                    }
                inps = inps.view(batch_size, seq_len, hidden_dim)

                pre_expert_hidden_states = inps.view(batch_size, seq_len, -1)

                tick = time.time()
                selected_experts,routing_weights,_ = layer.mlp.gate(inps)
                self.perf_stats['moe_gating'].append(time.time() - tick)

                layer_expert_stats = {
                    'layer_id': i_layer,
                    'expert_ids': selected_experts.tolist()
                }
                self.expert_selection_stats.append(layer_expert_stats)

                inps_after_experts = torch.zeros_like(inps, device=self.dev)
                experts = layer.mlp.experts
                _shared_output_holder = [None]
                ondemand_experts = []
                experts_in_placeholder = []
                run_cost_model = (self.batch_size > 1 and not self.is_decode) or\
                                 (self.batch_size > 16 and self.is_decode)
                if run_cost_model:
                    expert_token_counts = {}
                    for expert_id in selected_experts.unique():
                        mask = (selected_experts == expert_id).any(dim=1)
                        expert_token_counts[expert_id.item()] = mask.sum().item()
                    sorted_experts = sorted(expert_token_counts.items(), key=lambda x: x[1], reverse=True)
                    self.current_iter_expert_stats[i_layer] = {
                        'expert_ids': [e[0] for e in sorted_experts],
                        'token_counts': [e[1] for e in sorted_experts]
                    }
                    layer_i_stats = self.current_iter_expert_stats[i_layer]
                    filtered_expert_ids = []
                    filtered_token_counts = []
                    for expert_id, token_count in zip(layer_i_stats['expert_ids'], layer_i_stats['token_counts']):
                        if not self.is_expert_in_gpu_now(i_layer, expert_id):
                            filtered_expert_ids.append(expert_id)
                            filtered_token_counts.append(token_count)
                    sorted_experts = list(zip(filtered_expert_ids, filtered_token_counts))
                    n = len(sorted_experts)
                    cpu_time_table = self.cpu_time_table

                    if self.is_decode:

                        e  = 0.75
                        tg = 4 if self.batch_size < 64 else 2
                        b  = 0.5 if self.batch_size <= 16 else (1.25 if self.batch_size < 64 else 1.35)
                    else:

                        e  = 0.75
                        tg = 4 if self.batch_size < 64 else 2
                        b  = 0.5 if self.batch_size < 4 else (0.8 if self.batch_size < 16 else (1.5 if self.batch_size < 64 else 1.8))

                    TA = sum(cpu_time_table[min(tokens, 1498)] for _, tokens in sorted_experts)
                    TC = TA * b
                    for i in range(n - 1):
                        expert_id, token_count = sorted_experts[i]
                        TG = (1 + i) * e + tg
                        TC = TC - cpu_time_table[min(token_count, 1498)] * b
                        if self.is_decode:
                            if TG < TC:
                                if token_count > 1:
                                    ondemand_experts.append(expert_id)
                                else:
                                    experts_in_placeholder.append(expert_id)
                                if i == n - 2:
                                    if TC - TG > e:
                                        ondemand_experts.append(sorted_experts[i + 1][0])
                                    elif TC - TG > e / 2:
                                        self.prefil_pre = True
                            else:
                                break
                        else:
                            if TG < TC + cpu_time_table[min(token_count, 1498)] * b:
                                ondemand_experts.append(expert_id)
                            else:
                                break

                if self.cpu_offload == 0:
                    print("oo")

                else:

                    inps_after_experts = torch.zeros_like(inps, device=self.dev)
                    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.n_expert + self.n_shared_experts).permute(2, 1, 0)

                    cpu_experts = []
                    gpu_experts = []
                    experts_in_gpu = []
                    experts_remaining = []

                    if self.is_decode:
                        laymid = time.time() - layer_tick
                        laypid = time.time()
                        inps_flat_cpu = inps.view(-1, hidden_dim).to("cpu")
                        selected_experts_cpu = selected_experts.to("cpu")
                        routing_weights_cpu = routing_weights.to("cpu")
                        selected_expert_ids = selected_experts_cpu.unique().tolist()
                    else:
                        inps_flat_cpu = inps.view(-1, hidden_dim).to("cpu")
                        selected_experts_cpu = selected_experts.to("cpu")
                        routing_weights_cpu = routing_weights.to("cpu")
                        selected_expert_ids = selected_experts_cpu.unique().tolist()

                    gpu_results = []
                    cpu_results = []
                    gpu_time = 0.0
                    cpu_time = 0.0
                    for i_expert in selected_expert_ids:
                        if self.is_expert_in_gpu_now(i_layer, i_expert):
                            gpu_experts.append(i_expert)
                            experts_in_gpu.append(i_expert)
                            continue
                        if i_expert in ondemand_experts:
                            experts_remaining.append(i_expert)
                        else:
                            cpu_experts.append(i_expert)
                    def process_gpu_experts():
                        nonlocal gpu_time, inps_after_experts
                        start_time = time.time()
                        def process_experts_in_gpu():
                            results = []
                            for i_expert in experts_in_gpu:
                                mask = (selected_experts == i_expert).any(dim=1)
                                batch_mask = mask.view(batch_size, seq_len)
                                expert_input = inps[batch_mask].view(-1, hidden_dim)
                                tick = time.time()
                                expert_output = self.run_expert_at_gpu(i_layer, i_expert, expert_input)
                                self.perf_stats['expert_compute'].append(time.time() - tick)
                                flat_mask = mask.view(-1)
                                weights = routing_weights[flat_mask].gather(
                                    1, (selected_experts[flat_mask] == i_expert).long().argmax(dim=1, keepdim=True)
                                )
                                expert_output = expert_output * weights
                                mask_index = mask.nonzero().squeeze(1)
                                results.append((mask_index, expert_output))
                            return results

                        def process_experts_remaining():
                            results = []
                            for i_expert in experts_remaining:
                                mask = (selected_experts == i_expert).any(dim=1)
                                if not mask.any():
                                    continue
                                batch_mask = mask.view(batch_size, seq_len)
                                expert_input = inps[batch_mask].view(-1, hidden_dim)
                                self._async_ondemand(i_layer, i_expert, self.expert_placeholder5)
                                expert_output = self.expert_placeholder5(expert_input)
                                flat_mask = mask.view(-1)
                                weights = routing_weights[flat_mask].gather(
                                    1, (selected_experts[flat_mask] == i_expert).long().argmax(dim=1, keepdim=True)
                                )
                                expert_output = expert_output * weights
                                mask_index = mask.nonzero().squeeze(1)
                                results.append((mask_index, expert_output))
                            return results

                        if not experts_remaining:
                            results = process_experts_in_gpu()
                        else:
                            threads = []
                            results = []

                            def run_and_collect(func):
                                res = func()
                                results.extend(res)

                            threads.append(threading.Thread(target=run_and_collect, args=(process_experts_in_gpu,)))
                            threads.append(threading.Thread(target=run_and_collect, args=(process_experts_remaining,)))

                            for t in threads:
                                t.start()
                            for t in threads:
                                t.join()

                        for mask_index, expert_output in results:
                            expert_output = expert_output.view(-1, hidden_dim)
                            inps_after_experts = inps_after_experts.view(-1, hidden_dim)
                            inps_after_experts.index_add_(
                                0,
                                mask_index,
                                expert_output.to(inps_after_experts.dtype)
                            )
                        inps_after_experts = inps_after_experts.view(batch_size, seq_len, hidden_dim)

                        _shared_output_holder[0] = self.model.layers[i_layer].mlp.shared_experts(inps)
                        gpu_time = time.time() - start_time

                    def process_cpu_experts():
                        nonlocal cpu_time
                        start_time = time.time()
                        if self.is_decode:

                            flat_se = selected_experts_cpu.view(-1, selected_experts_cpu.shape[-1])
                            flat_rw = routing_weights_cpu.view(-1, routing_weights_cpu.shape[-1])

                            def run_one_expert(i_expert):
                                mask = (flat_se == i_expert).any(dim=1)
                                if not mask.any():
                                    return None
                                expert_input = inps_flat_cpu[mask]
                                tick = time.time()
                                expert_output = self.run_expert_at_cpu(i_layer, i_expert, expert_input)
                                self.perf_stats['expert_compute-cpu'].append(time.time() - tick)
                                weights = flat_rw[mask].gather(
                                    1, (flat_se[mask] == i_expert).long().argmax(dim=1, keepdim=True)
                                )
                                expert_output = expert_output * weights
                                mask_index = mask.nonzero().squeeze(1)
                                return (mask_index, expert_output)

                            futs = {self._cpu_expert_pool.submit(run_one_expert, ie): ie
                                    for ie in cpu_experts}
                            for fut in concurrent.futures.as_completed(futs):
                                result = fut.result()
                                if result is not None:
                                    cpu_results.append(result)
                        else:
                            flat_se = selected_experts_cpu.view(-1, selected_experts_cpu.shape[-1])
                            flat_rw = routing_weights_cpu.view(-1, routing_weights_cpu.shape[-1])

                            def run_one_expert_prefill(i_expert):
                                mask = (flat_se == i_expert).any(dim=1)
                                if not mask.any():
                                    return None
                                expert_input = inps_flat_cpu[mask]
                                tick = time.time()
                                expert_output = self.run_expert_at_cpu(i_layer, i_expert, expert_input)
                                self.perf_stats['expert_compute-cpu'].append(time.time() - tick)
                                weights = flat_rw[mask].gather(
                                    1, (flat_se[mask] == i_expert).long().argmax(dim=1, keepdim=True)
                                )
                                expert_output = expert_output * weights
                                mask_index = mask.nonzero().squeeze(1)
                                return (mask_index, expert_output)

                            futs = {self._cpu_expert_pool.submit(run_one_expert_prefill, ie): ie
                                    for ie in cpu_experts}
                            for fut in concurrent.futures.as_completed(futs):
                                result = fut.result()
                                if result is not None:
                                    cpu_results.append(result)
                        cpu_time = time.time() - start_time

                    parallel_start = time.time()
                    gpu_future = self._gpu_pool.submit(process_gpu_experts)
                    cpu_future = self._cpu_pool.submit(process_cpu_experts)
                    gpu_future.result()
                    cpu_future.result()

                    if self.is_decode:
                        torch.cuda.synchronize()
                    parallel_time = time.time() - parallel_start

                    max_thread_time = max(gpu_time, cpu_time)
                    parallel_degree = (gpu_time + cpu_time) / parallel_time if parallel_time > 0 else 1.0
                    print(f"L{i_layer:2d} gpu={gpu_time*1000:6.1f}ms cpu={cpu_time*1000:6.1f}ms wall={parallel_time*1000:6.1f}ms para={parallel_degree:.2f}x")

                    if cpu_results:
                        all_indices = torch.cat([idx for idx, _ in cpu_results], dim=0)
                        all_outputs = torch.cat(
                            [out.view(-1, hidden_dim) for _, out in cpu_results], dim=0
                        )
                        inps_after_experts.view(-1, hidden_dim).index_add_(
                            0,
                            all_indices.to(self.dev),
                            all_outputs.to(self.dev).to(inps_after_experts.dtype),
                        )

                inps_after_experts.add_(_shared_output_holder[0].reshape(batch_size, seq_len, hidden_dim))
                del _shared_output_holder

                inps_residual.add_(inps_after_experts.reshape(batch_size, seq_len, hidden_dim))
                inps = inps_residual
                del inps_after_experts, inps_residual
                if self.is_decode:
                    layer_time = time.time() - layer_tick
                    layer_time_final = time.time() - laypid
                    self.layer_time_accumulator[i_layer] += layer_time

                    self.layer_time_stats.append({
                        'layer_id': i_layer,
                        'time': layer_time,
                        'token_step': self.past_key_values_length
                    })

                print("prefil gpuexpet cout",self.gpuprefil_count,"bendigpuexpet cout",self.localprefil_count,"cpuexpet cout",self.cpuprefil_count)
                if self.is_decode:
                    layer_time = time.time() - layer_tick
                    layer_times[i_layer] += layer_time
                    layer_times_mid[i_layer] += laymid
                    layer_times_final[i_layer] += layer_time_final
                    self.layer_time_accumulator[i_layer] += layer_time

        hot_experts = self.get_hot_expert()

        if self.is_decode:

            print("_")
            for i in range(1, 27):

                layer_time = layer_times[i] * 1000
                layer_times_mids = layer_times_mid[i] * 1000
                layer_times_finals = layer_times_final[i] * 1000

            avg_layer_time = sum(layer_times.values()) / self.n_layer
            total_cpu_time = sum(self.cpu_expert_time_per_layer.values())
            avg_cpu_ratio = (total_cpu_time / avg_layer_time * 100) if avg_layer_time > 0 else 0

            os.makedirs('./log', exist_ok=True)
            with open('./log/expert_stats.txt', 'a') as f:
                f.write("_")
                for i in range(1, 27):
                    cpu_time = self.cpu_expert_time_per_layer[i] * 1000
                    layer_time = layer_times[i] * 1000
                    layer_time_mids = layer_times_mid[i] * 1000
                    layer_time_finals = layer_times_final[i] * 1000

                    f.write(f"_")

        other_ops_start = time.time()
        print("ok")
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        if self.is_decode:
            other_ops_time = time.time() - other_ops_start
            total_decode_time = time.time() - total_decode_start
            print(f"_")
            print(f"_")
            print(f"_")
            print(f"_")
        return lm_logis

        """Run the expert at CPU"""

    def run_expert_at_gpu(self, i_layer, i_expert, inps ):
        pass
        start_time = time.time()
        result = self._gpu_experts[i_layer][i_expert](inps)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        token_count = inps.shape[0]
        return result

    def run_expert_at_cpu(self, i_layer, i_expert, inps ):
        pass
        start_time = time.time()
        result = self.model.layers[i_layer].mlp.experts[i_expert](inps)
        elapsed = time.time() - start_time
        if self.is_decode:
            self.cpu_expert_time_per_layer[i_layer] += elapsed
        token_count = inps.shape[0]
        return result
    def get_expert_stats(self):
        pass
        stats = {
            'hot_experts': {i: {'count': 0, 'hot': 0, 'veryhot': 0} for i in range(1, 27)},
            'hot_counts': {2: 0, 3: 0, 4: 0, 5: 0},
            'token_distribution': {}
        }

        for record in self.expert_time_stats:
            layer = record['layer_id']
            token_count = record['token_count']
            expert_status = record['status']

            stats['hot_experts'][layer]['count'] += 1
            if expert_status == 'hot':
                stats['hot_experts'][layer]['hot'] += 1
            elif expert_status == 'veryhot':
                stats['hot_experts'][layer]['veryhot'] += 1

        layer_hot_counts = {i: 0 for i in range(1, 27)}
        for record in self.expert_time_stats:
            if record['status'] in ['hot', 'veryhot']:
                layer_hot_counts[record['layer_id']] += 1

        for count in layer_hot_counts.values():
            if count >= 2 and count <= 5:
                stats['hot_counts'][count] += 1

        stats['token_distribution'] = {}
        token_counts = [r['token_count'] for r in self.expert_time_stats]
        unique_counts = set(token_counts)

        for count in unique_counts:
            stats['token_distribution'][count] = token_counts.count(count)

        return stats

