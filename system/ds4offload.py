pass
import json
import os
import sys
import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors import safe_open

INFERENCE_DIR = "/home/share/data/model/DeepSeek-V4-Flash-INT8/inference"
MODEL_DIR = "/home/share/data/model/DeepSeek-V4-Flash-INT8"
sys.path.insert(0, INFERENCE_DIR)

import types as _types
_fht_mod = _types.ModuleType("fast_hadamard_transform")

def _hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :].clone(), x[..., 1, :].clone()
        x = torch.stack([a + b, a - b], dim=-2).view(*x.shape[:-3], n)
        h *= 2
    return x * scale

_fht_mod.hadamard_transform = _hadamard_transform
sys.modules["fast_hadamard_transform"] = _fht_mod

import model as _model_module
from model import Transformer, ModelArgs

_original_linear = _model_module.linear

def _patched_linear(x, weight, bias=None):
    if weight.dtype == torch.int8:
        scale = getattr(weight, 'scale', None)
        orig_device = x.device

        if weight.device != orig_device:
            x = x.to(weight.device)
            if bias is not None:
                bias = bias.to(weight.device)
        w = weight.float() * (scale.view(-1, 1).float() if scale is not None else 1.0)
        out = F.linear(x.float(), w, bias.float() if bias is not None else None)
        return out.to(orig_device)

    x_cast = x.to(weight.dtype)
    bias_cast = bias.to(weight.dtype) if bias is not None else None
    return F.linear(x_cast, weight, bias_cast)

_model_module.linear = _patched_linear

import torch.nn as nn
_OrigLinear = _model_module.Linear
class _PatchedLinear(_OrigLinear):
    def forward(self, x):
        return _patched_linear(x, self.weight, getattr(self, 'bias', None))
_model_module.Linear = _PatchedLinear

for cls_name in ['ColumnParallelLinear', 'RowParallelLinear']:
    orig = getattr(_model_module, cls_name)
    class _PatchedParallel(orig):
        def forward(self, x):
            return _patched_linear(x, self.weight, getattr(self, 'bias', None))
    _PatchedParallel.__name__ = cls_name
    setattr(_model_module, cls_name, _PatchedParallel)

_original_sparse_attn = _model_module.sparse_attn

def _patched_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    if topk_idxs.device != q.device:
        topk_idxs = topk_idxs.to(q.device)
    return _original_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

_model_module.sparse_attn = _patched_sparse_attn

import kernel as _kernel_module
_noop_fp4 = lambda x, block_size=32, inplace=False: x
_kernel_module.fp4_act_quant = _noop_fp4
_model_module.fp4_act_quant = _noop_fp4

def _patched_rotate_activation(x: torch.Tensor) -> torch.Tensor:
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x.to(torch.bfloat16), scale=x.size(-1) ** -0.5)

_model_module.rotate_activation = _patched_rotate_activation

_OrigIndexer = _model_module.Indexer
_orig_indexer_forward = _OrigIndexer.forward

def _patched_indexer_forward(self, x, qr, start_pos, offset):

    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
    ratio = self.compress_ratio
    rd = self.rope_head_dim
    end_pos = start_pos + seqlen
    if self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis
    q = self.wq_b(qr)
    q = q.unflatten(-1, (self.n_local_heads, self.head_dim)).to(torch.bfloat16)
    _model_module.apply_rotary_emb(q[..., -rd:], freqs_cis)
    q = _model_module.rotate_activation(q)

    self.compressor(x, start_pos)
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
    kv = self.kv_cache[:bsz, :end_pos // ratio].to(q.dtype)
    index_score = torch.einsum("bshd,btd->bsht", q, kv)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    dev = index_score.device
    if start_pos == 0:
        mask = torch.arange(seqlen // ratio, device=dev).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1, device=dev).unsqueeze(1) // ratio
        index_score += torch.where(mask, float("-inf"), torch.zeros(1, device=dev))
    topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
    if start_pos == 0:
        mask = topk_idxs >= torch.arange(1, seqlen + 1, device=dev).unsqueeze(1) // ratio
        topk_idxs = torch.where(mask, -1, topk_idxs + offset)
    else:
        topk_idxs += offset
    return topk_idxs

_OrigIndexer.forward = _patched_indexer_forward

_OrigCompressor = _model_module.Compressor
_orig_compressor_forward = _OrigCompressor.forward

def _patched_compressor_forward(self, x: torch.Tensor, start_pos: int):

    if start_pos > 0 and x.size(1) == 1:
        return _orig_compressor_forward(self, x, start_pos)

    assert self.kv_cache is not None
    bsz, seqlen, _ = x.size()
    ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
    dtype = x.dtype
    x = x.float()
    kv = self.wkv(x)
    score = self.wgate(x)

    should_compress = seqlen >= ratio
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    offset_state = ratio if overlap else 0

    if remainder > 0:

        kv, kv_rem = kv.split([cutoff, remainder], dim=1)
        score_rem = score[:, cutoff:] + self.ape[:remainder]
        score = score[:, :cutoff]
        self.kv_state[:bsz, offset_state : offset_state+remainder] = kv_rem
        self.score_state[:bsz, offset_state : offset_state+remainder] = score_rem

    if overlap and cutoff >= ratio:
        self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
        self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape

    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + self.ape
    if overlap:
        kv = self.overlap_transform(kv, 0)
        score = self.overlap_transform(score, float("-inf"))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)

    if not should_compress:
        return

    kv = self.norm(kv.to(dtype))

    freqs_cis = self.freqs_cis[start_pos : start_pos + cutoff : ratio]
    _model_module.apply_rotary_emb(kv[..., -rd:], freqs_cis)
    if self.rotate:
        kv = _model_module.rotate_activation(kv)
        _model_module.fp4_act_quant(kv, _model_module.fp4_block_size, True)
    else:
        _model_module.act_quant(kv[..., :-rd], 64, _model_module.scale_fmt, _model_module.scale_dtype, True)

    base = start_pos // ratio
    self.kv_cache[:bsz, base : base + cutoff // ratio] = kv
    return kv

_OrigCompressor.forward = _patched_compressor_forward

_kernel_module.act_quant = lambda x, block_size=64, scale_fmt=None, scale_dtype=None, inplace=False: x
_model_module.act_quant = _kernel_module.act_quant

_OrigAttention = _model_module.Attention
_orig_attn_forward = _OrigAttention.forward

def _patched_attn_forward(self, x, start_pos):
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
    win = self.window_size
    ratio = self.compress_ratio
    rd = self.rope_head_dim
    if self.compress_ratio and self.compressor.kv_cache is None:

        self.compressor.kv_cache = self.kv_cache[:, win:]
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis
    qr = q = self.q_norm(self.wq_a(x))
    q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
    q = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16)
    _model_module.apply_rotary_emb(q[..., -rd:], freqs_cis)

    kv = self.wkv(x)
    kv = self.kv_norm(kv)
    _model_module.apply_rotary_emb(kv[..., -rd:], freqs_cis)

    if start_pos == 0:
        topk_idxs = _model_module.get_window_topk_idxs(win, bsz, seqlen, start_pos).to(kv.device)
        if self.compress_ratio:
            offset = kv.size(1)
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = _model_module.get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset).to(kv.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()
        if seqlen <= win:
            self.kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
        if self.compress_ratio:
            if (kv_compress := self.compressor(x, start_pos)) is not None:

                kv = torch.cat([kv, kv_compress.to(kv.device).to(kv.dtype)], dim=1)
        o = _model_module.sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    elif seqlen > 1:

        dev = self.kv_cache.device
        cpu_buf = getattr(self, 'cpu_compress_kv', None)

        n_compress_hist = start_pos // ratio if ratio else 0

        kv_win = self.kv_cache[:bsz]
        if cpu_buf is not None and n_compress_hist > 0:
            kv_hist = torch.cat([kv_win, cpu_buf[:bsz, :n_compress_hist].to(dev)], dim=1)
        else:
            kv_hist = kv_win
        hist_len = kv_hist.shape[1]

        kv_full = torch.cat([kv_hist, kv], dim=1)
        total_kv = kv_full.shape[1]

        attn_bias = torch.zeros(seqlen, total_kv, device=dev, dtype=torch.float32)
        causal_mask = torch.ones(seqlen, seqlen, device=dev, dtype=torch.bool).triu(diagonal=1)
        attn_bias[:, hist_len:].masked_fill_(causal_mask, float('-inf'))

        q_sdpa = q.permute(0, 2, 1, 3).float()
        kv_sdpa = kv_full.unsqueeze(1).float()

        o = F.scaled_dot_product_attention(
            q_sdpa, kv_sdpa, kv_sdpa,
            attn_mask=attn_bias[None, None],
            scale=self.softmax_scale,
        )
        o = o.to(q.dtype).permute(0, 2, 1, 3).contiguous()

        for t in range(seqlen):
            self.kv_cache[:bsz, (start_pos + t) % win] = kv[:, t, :]
        if ratio:
            self.compressor(x, start_pos)
    else:

        topk_idxs = _model_module.get_window_topk_idxs(win, bsz, seqlen, start_pos).to(kv.device)
        if self.compress_ratio:
            compress_topk_idxs = self.indexer(x, qr, start_pos, win) if self.indexer is not None else\
                _model_module.get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, win).to(kv.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()
        self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        if self.compress_ratio:
            self.compressor(x, start_pos)
            cpu_buf = getattr(self, 'cpu_compress_kv', None)
            n_compress = start_pos // self.compress_ratio

            _parallel_thresh = getattr(self, '_attn_parallel_thresh', 1024)
            use_parallel = (cpu_buf is not None and cpu_buf.device.type == 'cpu'
                            and n_compress > 0 and n_compress >= _parallel_thresh)

            if use_parallel:

                scale = self.softmax_scale

                win_topk = topk_idxs[:, :, :win].int().contiguous()
                o_win = _model_module.sparse_attn(
                    q, self.kv_cache[:bsz], self.attn_sink, win_topk, scale
                )

                q_cpu = q.squeeze(1).to('cpu', non_blocking=True)
                kv_cpu = cpu_buf[:bsz, :n_compress]

                q_f = q_cpu.float()
                kv_f = kv_cpu.float()
                scores_c = torch.einsum('bnh,bsh->bns', q_f, kv_f) * scale
                m_c = scores_c.amax(dim=-1, keepdim=True)
                e_c = torch.exp(scores_c - m_c)
                s_c = e_c.sum(dim=-1, keepdim=True)
                o_c = torch.einsum('bns,bsh->bnh', e_c, kv_f)

                torch.cuda.synchronize()
                kv_win_f = self.kv_cache[:bsz].float()
                q_f_gpu = q.squeeze(1).float()
                scores_w = torch.einsum('bnh,bsh->bns', q_f_gpu, kv_win_f) * scale
                m_w = scores_w.amax(dim=-1, keepdim=True)
                e_w = torch.exp(scores_w - m_w)
                s_w = e_w.sum(dim=-1, keepdim=True)
                o_w = torch.einsum('bns,bsh->bnh', e_w, kv_win_f)

                o_c_d = o_c.to(self.kv_cache.device, non_blocking=True)
                m_c_d = m_c.to(self.kv_cache.device, non_blocking=True)
                s_c_d = s_c.to(self.kv_cache.device, non_blocking=True)
                torch.cuda.synchronize()

                m_g = torch.max(m_w, m_c_d)
                w_w = torch.exp(m_w - m_g)
                w_c = torch.exp(m_c_d - m_g)
                denom = (w_w * s_w + w_c * s_c_d).clamp(min=1e-9)
                o_merged = (o_w * w_w + o_c_d * w_c) / denom
                o = o_merged.to(q.dtype).unsqueeze(1)
            elif cpu_buf is not None and topk_idxs.shape[-1] > win:
                n_ctok = topk_idxs.shape[-1] - win
                dev = self.kv_cache.device
                if self.indexer is None:

                    compress_gpu = cpu_buf[:bsz, :n_ctok].to(dev, non_blocking=False)
                else:

                    c_idxs_flat_cpu = (topk_idxs[:, :, win:].long() - win).clamp(min=0).view(bsz, n_ctok).cpu()
                    compress_gpu = cpu_buf[:bsz].gather(
                        1, c_idxs_flat_cpu.unsqueeze(-1).expand(-1, -1, cpu_buf.shape[-1])
                    ).to(dev, non_blocking=False)
                kv_full = torch.cat([self.kv_cache[:bsz], compress_gpu], dim=1)
                new_topk_idxs = torch.arange(win + n_ctok, device=dev, dtype=torch.int32).view(
                    1, 1, -1).expand(bsz, seqlen, -1).contiguous()
                o = _model_module.sparse_attn(q, kv_full, self.attn_sink, new_topk_idxs, self.softmax_scale)
            else:
                o = _model_module.sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            o = _model_module.sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)

    _model_module.apply_rotary_emb(o[..., -rd:], freqs_cis, True)

    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    x = self.wo_b(o.flatten(2))
    return x

_OrigAttention.forward = _patched_attn_forward

def _load_model_weights_int8(model: Transformer, ckpt_path: str):
    pass
    index_file = os.path.join(ckpt_path, "model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    base_to_scale = {}
    for k in weight_map:
        if k.endswith(".weight_scale"):
            base = k[:-len("_scale")]
            base_to_scale[base] = k

    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    model_sd = {**model_params, **model_buffers}

    scale_cache: dict = {}

    loaded = skipped = int8_loaded = 0
    files_processed = set()

    for key, fname in weight_map.items():
        if fname in files_processed:
            continue

    file_to_keys: dict = {}
    for key, fname in weight_map.items():
        file_to_keys.setdefault(fname, []).append(key)

    for fname in sorted(file_to_keys):
        fpath = os.path.join(ckpt_path, fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.endswith(".weight_scale"):
                    scale_cache[key] = tensor
                    continue
                if key not in model_sd:
                    skipped += 1
                    continue
                param = model_sd[key]
                if tensor.dtype == torch.int8:

                    parts = key.rsplit(".", 1)
                    if len(parts) == 2:
                        parent_key, attr = parts
                        parent = model
                        for p in parent_key.split("."):
                            parent = getattr(parent, p)

                        object.__setattr__(parent, attr,
                                           tensor.detach().requires_grad_(False))

                        model_sd[key] = getattr(parent, attr)
                    int8_loaded += 1
                else:
                    if param.shape == tensor.shape:
                        param.data.copy_(tensor.to(param.dtype))
                    else:
                        param.data = tensor.to(param.dtype)
                loaded += 1

    scale_attached = 0
    for base_key, scale_key in base_to_scale.items():
        if scale_key in scale_cache and base_key in model_sd:
            param = model_sd[base_key]
            param.scale = scale_cache[scale_key]
            scale_attached += 1

    print(f"Loaded {loaded} BF16/FP32 + {int8_loaded} INT8 tensors, "
          f"{scale_attached} scales attached, {skipped} skipped.")

class CacheScopeDeepSeekV4:
    pass

    def __init__(self, args):
        self.dev = torch.device("cuda:0")
        self.batch_size = args.batch_size
        self.cpu_offload = args.cpu_offload
        self.beam_width = getattr(args, "beam_width", 1)

        config_path = os.path.join(INFERENCE_DIR, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        cfg["max_batch_size"] = self.batch_size
        cfg["max_seq_len"] = getattr(args, 'max_seq_len', 4096)

        cfg["expert_dtype"] = None
        cfg["dtype"] = "bf16"
        cfg["scale_fmt"] = None
        cfg["scale_dtype"] = "fp32"
        self.model_args = ModelArgs(**cfg)

        print(f"n_layers={self.model_args.n_layers}, "
              f"n_experts={self.model_args.n_routed_experts}, "
              f"dim={self.model_args.dim}")

        print("Building model structure on CPU...")
        with torch.device("cpu"):
            self.model = Transformer(self.model_args)

        print("Loading INT8 weights...")
        tick = time.time()
        _load_model_weights_int8(self.model, MODEL_DIR)
        print(f"Weight load: {time.time()-tick:.1f}s")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.n_layer = self.model_args.n_layers
        self.n_expert = self.model_args.n_routed_experts

        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=np.int8)

        self.expert_freq = np.zeros((self.n_layer, self.n_expert), dtype=np.int64)

        n_gpu = self._calc_n_gpu_experts()
        print(f"GPU expert budget: {n_gpu}/{self.n_layer * self.n_expert} experts")

        self._move_non_experts_to_gpu()
        self._move_experts_to_gpu(n_gpu)
        self._setup_od_placeholders()
        self._patch_moe_parallel()

        self.prefill_chunk_size = getattr(args, 'prefill_chunk_size', 2048)
        self._reset_perf_stats()
        print("Model ready.")

    def _reset_perf_stats(self):
        self.perf_stats = {
            "token_embedding": [],
            "self_attention": [],
            "moe_gating": [],
            "expert_compute": [],
            "expert_compute-cpu": [],
        }

    def _setup_od_placeholders(self):
        pass

        src_expert = None
        for lid in range(self.n_layer):
            for eid in range(self.n_expert):
                if self.expert_loc[lid, eid] == 0:
                    src_expert = self.model.layers[lid].ffn.experts[eid]
                    break
            if src_expert is not None:
                break

        if src_expert is None:

            self.od_placeholder = []
            self._od_stream_h2d  = None
            self._od_stream_gemm = None
            return

        import copy
        def _make_ph():
            ph = copy.deepcopy(src_expert)
            self._move_module_with_scale(ph, self.dev)
            return ph

        self.od_placeholder = [_make_ph(), _make_ph()]
        self._od_stream_h2d  = torch.cuda.Stream()
        self._od_stream_gemm = torch.cuda.Stream()

        table_path = os.path.join(os.path.dirname(__file__), 'microdeepseek.txt')
        try:
            with open(table_path) as f:
                vals = [float(l.strip()) for l in f if l.strip()]
            self._cpu_time_table = vals
            self._cpu_time_table_max = len(vals) - 1
        except Exception:
            self._cpu_time_table = None
            self._cpu_time_table_max = 0

        self._cm_e_decode  = 0.0015
        self._cm_e_prefill = 0.0020
        self._cm_alpha = 0.1

    def _async_ondemand_ds4(self, layer_id, expert_id, ph_idx):
        pass
        expert = self.model.layers[layer_id].ffn.experts[expert_id]
        ph = self.od_placeholder[ph_idx]
        with torch.cuda.stream(self._od_stream_h2d):
            for attr in ('w1', 'w2', 'w3'):
                src_w = getattr(expert, attr).weight.data
                dst_w = getattr(ph, attr).weight.data
                dst_w.copy_(src_w, non_blocking=True)
                if hasattr(src_w, 'scale') and src_w.scale is not None:
                    if not hasattr(dst_w, 'scale') or dst_w.scale is None:
                        dst_w.scale = src_w.scale.to(self.dev, non_blocking=True)
                    else:
                        dst_w.scale.copy_(src_w.scale, non_blocking=True)

    def _patch_moe_parallel(self):
        pass
        expert_freq = self.expert_freq
        expert_loc  = self.expert_loc
        n_experts   = self.n_expert
        dev         = self.dev
        cacheScope = self

        for layer_id in range(self.n_layer):
            moe  = self.model.layers[layer_id].ffn
            orig_gate_fwd = moe.gate.forward

            def make_moe_wrapper(lid, moe_ref):
                def patched_forward(x, input_ids=None):
                    shape = x.size()
                    x_flat = x.view(-1, moe_ref.dim)

                    weights, indices = orig_gate_fwd(x_flat, input_ids)
                    counts = torch.bincount(indices.flatten().long(),
                                            minlength=n_experts).cpu().numpy()
                    expert_freq[lid] += counts

                    with torch.inference_mode(False):
                        y = torch.zeros(x_flat.shape[0], moe_ref.dim,
                                        dtype=torch.float32, device=dev)
                    _scatter_lock = threading.Lock()

                    active = [i for i in range(moe_ref.experts_start_idx,
                                               moe_ref.experts_end_idx)
                              if counts[i] > 0]
                    gpu_res  = [i for i in active if expert_loc[lid, i] == 1]
                    cpu_rem  = [(i, int(counts[i])) for i in active if expert_loc[lid, i] == 0]

                    cpu_rem.sort(key=lambda t: t[1], reverse=True)

                    od_list  = []
                    cpu_list = []
                    if cacheScope.od_placeholder and cpu_rem:
                        tbl     = cacheScope._cpu_time_table
                        tbl_max = cacheScope._cpu_time_table_max
                        _e = (cacheScope._cm_e_decode if cacheScope.is_decode
                              else cacheScope._cm_e_prefill)
                        if tbl is not None:
                            _cpu_times = [tbl[min(tc, tbl_max)] for _, tc in cpu_rem]
                            T_cpu = sum(_cpu_times)
                            T_gpu = 0.0
                            split = 0
                            for k, ((eid, tc), ct) in enumerate(zip(cpu_rem, _cpu_times)):
                                T_gpu += _e
                                T_cpu -= ct
                                if T_gpu >= T_cpu:
                                    split = k + 1
                                    break
                            else:
                                split = len(cpu_rem)

                            split = min(split, len(cacheScope.od_placeholder))
                            if not cacheScope.is_decode and split == len(cpu_rem) and len(cpu_rem) > 1:
                                split -= 1
                            od_list  = [eid for eid, _ in cpu_rem[:split]]
                            cpu_list = [eid for eid, _ in cpu_rem[split:]]
                        else:
                            cpu_list = [eid for eid, _ in cpu_rem]
                    else:
                        cpu_list = [eid for eid, _ in cpu_rem]

                    def run_gpu_resident():
                        with torch.inference_mode(False):
                            for ei in gpu_res:
                                idx, top = torch.where(indices == ei)
                                if idx.numel() == 0:
                                    continue
                                out = moe_ref.experts[ei](x_flat[idx], weights[idx, top, None])
                                with _scatter_lock:
                                    y.index_add_(0, idx, out.float())

                    def run_gpu_od():
                        if not od_list:
                            return
                        h2d_s  = cacheScope._od_stream_h2d
                        gemm_s = cacheScope._od_stream_gemm
                        phs    = cacheScope.od_placeholder
                        pending = []
                        ph_idx = 0

                        with torch.inference_mode(False):
                            for ei in od_list:
                                cacheScope._async_ondemand_ds4(lid, ei, ph_idx % len(phs))
                                ev = torch.cuda.Event()
                                ev.record(h2d_s)
                                idx, top = torch.where(indices == ei)
                                pending.append((ei, ev, ph_idx % len(phs), idx, top))
                                ph_idx += 1

                                if len(pending) >= 2:
                                    p_ei, p_ev, p_ph, p_idx, p_top = pending.pop(0)
                                    gemm_s.wait_event(p_ev)
                                    with torch.cuda.stream(gemm_s):
                                        out = phs[p_ph](x_flat[p_idx], weights[p_idx, p_top, None])
                                        gemm_s.synchronize()
                                    with _scatter_lock:
                                        y.index_add_(0, p_idx, out.float())

                            for p_ei, p_ev, p_ph, p_idx, p_top in pending:
                                gemm_s.wait_event(p_ev)
                                with torch.cuda.stream(gemm_s):
                                    out = phs[p_ph](x_flat[p_idx], weights[p_idx, p_top, None])
                                    gemm_s.synchronize()
                                with _scatter_lock:
                                    y.index_add_(0, p_idx, out.float())

                    def run_cpu():
                        if not cpu_list:
                            return
                        x_cpu  = x_flat.to('cpu')
                        idx_cpu = indices.to('cpu')
                        w_cpu  = weights.to('cpu')
                        results = []
                        for ei in cpu_list:
                            idx_c, top_c = torch.where(idx_cpu == ei)
                            if idx_c.numel() == 0:
                                continue
                            out_c = moe_ref.experts[ei](x_cpu[idx_c], w_cpu[idx_c, top_c, None])
                            results.append((idx_c, out_c))
                        if results:
                            all_idx = torch.cat([r[0] for r in results]).to(dev)
                            all_out = torch.cat([r[1].view(-1, moe_ref.dim)
                                                 for r in results]).to(dev)
                            with _scatter_lock:
                                y.index_add_(0, all_idx, all_out.float())

                    if cpu_rem:
                        t1 = threading.Thread(target=run_gpu_resident)
                        t2 = threading.Thread(target=run_gpu_od)
                        t3 = threading.Thread(target=run_cpu)
                        t1.start(); t2.start(); t3.start()
                        t1.join();  t2.join();  t3.join()
                    else:
                        run_gpu_resident()

                    y_out = y + moe_ref.shared_experts(x_flat).float()
                    return y_out.to(x.dtype).view(shape)

                return patched_forward

            moe.forward = make_moe_wrapper(layer_id, moe)

    def _patch_moe_freq_tracking(self):
        pass
        expert_freq = self.expert_freq
        n_experts = self.n_expert

        for layer_id in range(self.n_layer):
            gate = self.model.layers[layer_id].ffn.gate
            orig_gate_fwd = gate.forward

            def make_gate_wrapper(orig, lid):
                def gate_forward(x, input_ids=None):
                    weights, indices = orig(x, input_ids)

                    counts = torch.bincount(indices.flatten().long(),
                                            minlength=n_experts).cpu().numpy()
                    expert_freq[lid] += counts
                    return weights, indices
                return gate_forward

            gate.forward = make_gate_wrapper(orig_gate_fwd, layer_id)

    def _calc_n_gpu_experts(self) -> int:

        expert_bytes = 3 * self.model_args.moe_inter_dim * self.model_args.dim

        available = int(20 * 1024**3)
        n = available // expert_bytes
        return min(n, self.n_layer * self.n_expert)

    def _move_scale_to_device(self, module, device):
        pass
        for name, param in list(module.named_parameters(recurse=False)):
            if hasattr(param, 'scale') and param.scale is not None:
                param.scale = param.scale.to(device)

        for name, val in list(module.__dict__.items()):
            if isinstance(val, torch.Tensor) and not isinstance(val, torch.nn.Parameter):
                moved = val.to(device)
                object.__setattr__(module, name, moved)
                if hasattr(val, 'scale') and val.scale is not None:
                    moved.scale = val.scale.to(device)
        for child in module.children():
            self._move_scale_to_device(child, device)

    def _move_module_with_scale(self, module, device):
        module.to(device)
        self._move_scale_to_device(module, device)

    def _move_non_experts_to_gpu(self):
        print("Moving non-expert params to GPU...")
        self._move_module_with_scale(self.model.embed, self.dev)
        self._move_module_with_scale(self.model.norm, self.dev)
        self._move_module_with_scale(self.model.head, self.dev)
        for attr in ['hc_head_fn', 'hc_head_base', 'hc_head_scale']:
            getattr(self.model, attr).data = getattr(self.model, attr).data.to(self.dev)

        self._pre_shrink_kv_caches_on_cpu()
        for layer in self.model.layers:
            self._move_module_with_scale(layer.attn, self.dev)
            self._move_module_with_scale(layer.attn_norm, self.dev)
            self._move_module_with_scale(layer.ffn_norm, self.dev)
            self._move_module_with_scale(layer.ffn.gate, self.dev)
            self._move_module_with_scale(layer.ffn.shared_experts, self.dev)
            for attr in ['hc_attn_fn', 'hc_attn_base', 'hc_attn_scale',
                         'hc_ffn_fn', 'hc_ffn_base', 'hc_ffn_scale']:
                getattr(layer, attr).data = getattr(layer, attr).data.to(self.dev)

        self._cast_kv_caches_to_bf16()

        self._setup_compress_kv_cpu_offload()

    def _pre_shrink_kv_caches_on_cpu(self):
        pass
        bs = self.model_args.max_batch_size
        win = self.model_args.window_size
        for layer in self.model.layers:
            attn = layer.attn

            win_kv = attn.kv_cache[:bs, :win].contiguous()
            attn.register_buffer('kv_cache', win_kv, persistent=False)

    def _setup_compress_kv_cpu_offload(self):
        pass
        win = self.model_args.window_size
        max_seq = self.model_args.max_seq_len
        bs = self.model_args.max_batch_size
        compress_ratios = self.model_args.compress_ratios
        head_dim = self.model_args.head_dim
        n_cpu_bufs = 0

        for lid, layer in enumerate(self.model.layers):
            attn = layer.attn
            ratio = compress_ratios[lid] if lid < len(compress_ratios) else 0

            if ratio == 0:

                win_kv = attn.kv_cache[:bs, :win].contiguous()
                attn.register_buffer('kv_cache', win_kv, persistent=False)
                attn.cpu_compress_kv = None
                continue

            win_kv = attn.kv_cache[:bs, :win].contiguous()
            attn.register_buffer('kv_cache', win_kv, persistent=False)

            compress_len = max_seq // ratio
            if ratio == 128:

                compress_buf = torch.zeros(bs, compress_len, head_dim,
                                           dtype=torch.bfloat16, device=self.dev)
            else:

                compress_buf = torch.zeros(bs, compress_len, head_dim,
                                           dtype=torch.bfloat16).pin_memory()
            attn.cpu_compress_kv = compress_buf

            attn.compressor.kv_cache = compress_buf

            attn.compressor.freqs_cis = attn.freqs_cis
            if attn.indexer is not None:
                attn.indexer.freqs_cis = attn.freqs_cis
            n_cpu_bufs += 1

        n_gpu_bufs = sum(1 for lid, layer in enumerate(self.model.layers)
                         if (compress_ratios[lid] if lid < len(compress_ratios) else 0) == 128)
        print(f"Compress KV: {n_gpu_bufs} HCA layers on GPU, {n_cpu_bufs - n_gpu_bufs} CSA layers on CPU pinned, "
              f"win_kv kept on GPU ({win} entries/layer)")

    def _cast_kv_caches_to_bf16(self):
        pass
        for layer in self.model.layers:
            attn = layer.attn
            if hasattr(attn, 'kv_cache') and attn.kv_cache is not None:
                attn.register_buffer('kv_cache', attn.kv_cache.to(torch.bfloat16), persistent=False)

            if hasattr(attn, 'indexer') and attn.indexer is not None:
                idx = attn.indexer
                if hasattr(idx, 'kv_cache') and idx.kv_cache is not None:
                    idx.register_buffer('kv_cache', idx.kv_cache.to(torch.bfloat16), persistent=False)
                if hasattr(idx, 'compressor') and idx.compressor is not None:
                    ic = idx.compressor
                    if hasattr(ic, 'kv_cache') and ic.kv_cache is not None:
                        ic.register_buffer('kv_cache', ic.kv_cache.to(torch.bfloat16), persistent=False)

    def _move_experts_to_gpu(self, n_gpu: int):
        pass
        hot_file = "./hot/ds4.txt"
        popular = []
        if os.path.exists(hot_file):
            try:
                with open(hot_file) as f:
                    popular = [tuple(map(int, l.strip().split(',')))
                                for l in f if l.strip()]
                print(f"Loaded {len(popular)} hot experts from {hot_file}")
            except Exception as e:
                print(f"Hot expert file load failed: {e}, using sequential order")
                popular = []

        if not popular:

            popular = [(lid, eid)
                       for lid in range(self.n_layer)
                       for eid in range(self.n_expert)]

        print(f"Moving {n_gpu} experts to GPU...")
        count = 0
        try:
            for (layer_id, expert_id) in popular:
                if count >= n_gpu:
                    break
                expert = self.model.layers[layer_id].ffn.experts[expert_id]
                if expert is not None:
                    self._move_module_with_scale(expert, self.dev)
                    self.expert_loc[layer_id, expert_id] = 1
                    count += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at expert {count}, stopping.")
                torch.cuda.empty_cache()
            else:
                raise
        print(f"Moved {count} experts to GPU.")

    def save_hot_experts(self):
        pass
        os.makedirs("./hot", exist_ok=True)
        pairs = []
        for lid in range(self.n_layer):
            for eid in range(self.n_expert):
                freq = int(self.expert_freq[lid, eid])
                if freq > 0:
                    pairs.append((lid, eid, freq))

        pairs.sort(key=lambda x: (-x[2], x[0], x[1]))

        zero = [(lid, eid, 0)
                for lid in range(self.n_layer)
                for eid in range(self.n_expert)
                if self.expert_freq[lid, eid] == 0]
        all_pairs = pairs + zero
        with open("./hot/ds4.txt", "w") as f:
            for lid, eid, _ in all_pairs:
                f.write(f"{lid},{eid}\n")
        print(f"Saved hot expert list: {len(pairs)} active experts → ./hot/ds4.txt")

    def tokenize(self, texts, input_token=None):
        if isinstance(texts, str):
            texts = [texts]
        while len(texts) < self.batch_size:
            texts = texts + [texts[-1]]
        texts = texts[:self.batch_size]
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=input_token or 2048, return_tensors="pt",
        )
        input_ids = enc.input_ids.to(self.dev)
        if input_token is not None:
            input_ids = input_ids[:, :input_token]
        return input_ids

    @torch.inference_mode()
    def _forward_step(self, input_ids, start_pos):
        return self.model.forward(input_ids, start_pos)

    def generate(self, text=None, output_token=20, input_token=None):
        self._reset_perf_stats()
        torch.set_num_threads(16)

        if text is None:
            text = ["Hello world"] * self.batch_size
        elif isinstance(text, str):
            text = [text] * self.batch_size

        input_ids = self.tokenize(text, input_token)
        bsz, prompt_len = input_ids.shape
        print(f"Input: bs={bsz}, seq_len={prompt_len}")

        chunk = self.prefill_chunk_size

        tick = time.time()
        self.is_decode = False
        if prompt_len <= chunk:
            logits = self._forward_step(input_ids, start_pos=0)
        else:

            logits = self._forward_step(input_ids[:, :chunk], start_pos=0)
            for c_start in range(chunk, prompt_len, chunk):
                c_end = min(c_start + chunk, prompt_len)
                logits = self._forward_step(input_ids[:, c_start:c_end], start_pos=c_start)
        prefill_time = time.time() - tick
        print(f"Prefill: {prefill_time*1000:.1f}ms")

        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = [next_token]
        start_pos = prompt_len

        tick = time.time()
        self.is_decode = True
        self.token_decode_times = []
        for _ in range(output_token - 1):
            t0 = time.time()
            logits = self._forward_step(next_token, start_pos=start_pos)
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            start_pos += 1
            self.token_decode_times.append(time.time() - t0)

        decode_time = time.time() - tick

        all_tokens = torch.cat(generated, dim=1)
        decode_strings = [
            self.tokenizer.decode(all_tokens[i].tolist(), skip_special_tokens=True)
            for i in range(bsz)
        ]

        total_activations = int(self.expert_freq.sum())
        gpu_activations = int(sum(
            self.expert_freq[lid, eid]
            for lid in range(self.n_layer)
            for eid in range(self.n_expert)
            if self.expert_loc[lid, eid] == 1
        ))
        hit_rate = gpu_activations / max(total_activations, 1)
        print(f"GPU expert hit rate: {gpu_activations}/{total_activations} = {hit_rate*100:.1f}%")

        stats = {
            "perf_stats": self.perf_stats,
            "expert_selection": [], "expert_time": [], "layer_time": [],
            "outputs": decode_strings,
            "expert_hot_stats": {
                "hit_rate": hit_rate,
                "gpu_activations": gpu_activations,
                "total_activations": total_activations,
            },
        }
        return prefill_time, decode_time, hit_rate, stats

