pass
import argparse
import contextlib
import json
import os
import sys
import threading
import time

import torch
import transformers

from qwenoffload import FiddlerQwenWithOffloading, InplaceKVCache, _rms_norm_bf16

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import bench_resident_kv as rkv
from deepseek import FiddlerDeepSeekV2, InplaceKVCache as DS2InplaceKVCache
from mixoffload import FiddlerMixtralWithOffloading

_TIMING_BREAKDOWN = {
    'project_qk': 0.0,
    'extract_kv': 0.0,
    'prep_chunks': 0.0,
    'run_bkv': 0.0,
    'o_proj': 0.0,
    'count': 0,
}

OURS_BUDGET_GIB = 4
BASELINE_CONFIG = {
    "solid":       dict(ratio=0.8, budget_gib=2, arch="pingpong", model_key=None),
    "scout":       dict(ratio=0.8, budget_gib=2, arch="twolayer", model_key=None),
    "hybridgen":   dict(ratio=1.0, budget_gib=2, arch="twolayer", model_key=None),
    "ours-bkv":    dict(ratio=1.0, budget_gib=OURS_BUDGET_GIB, arch="ours_bkv", model_key=None),
    "ours-kvlen":  dict(ratio=1.0, budget_gib=OURS_BUDGET_GIB, arch="ours_kvlen", model_key=None),

    "fastdecode":  dict(ratio=1.0, budget_gib=0, arch="twolayer", model_key=None),
}

BASELINE_TO_PREP_MODE = {
    "solid": "pingpong",
    "scout": "twolayer",
    "hybridgen": "twolayer",
    "ours-bkv": "bkv",
    "ours-kvlen": "kvlen",
    "fastdecode": "twolayer",
}

class BenchQwen(FiddlerQwenWithOffloading):
    pass

    gpu_mem_budget_gb = 32

    def __init__(self, args):

        self._cost_log_lock = threading.Lock()
        self._cost_log_file = open(os.devnull, "w")
        super().__init__(args)

    def get_expert_module(self, layer_idx, expert_idx):
        pass
        return self.model.layers[layer_idx].mlp.experts[expert_idx]

    def calc_n_expert_on_gpu(self):
        n_param = sum(
            p.numel() for p in self.get_expert_module(0, 0).parameters()
        )
        expert_size_bytes = n_param * 2
        max_gpu_bytes = self.gpu_mem_budget_gb * 1024 ** 3
        current_allocated = torch.cuda.memory_allocated()
        reserved_bytes = 10 * 1024 ** 3
        available_bytes = max_gpu_bytes - current_allocated - reserved_bytes
        n_from_mem = int(available_bytes / expert_size_bytes)
        n_total = self.n_layer * self.n_expert
        return max(16, min(n_from_mem, n_total))

    def compute_attention_with_split_kv(self, layer, hidden_states, attention_mask,
                                        position_embeddings, layer_idx, device):
        pass
        if not hasattr(self, '_attn_timing'):
            self._attn_timing = {}
        self._attn_timing[layer_idx] = {}

        t0 = time.time()
        result = super().compute_attention_with_split_kv(
            layer, hidden_states, attention_mask, position_embeddings, layer_idx, device
        )
        self._attn_timing[layer_idx]['total'] = time.time() - t0
        return result

class BenchDeepSeek(FiddlerDeepSeekV2):
    pass

    gpu_mem_budget_gb = 32

    def __init__(self, args):
        self._cost_log_lock = threading.Lock()
        self._cost_log_file = open(os.devnull, "w")
        super().__init__(args)

        if not hasattr(self, 'perf_stats'):
            self.perf_stats = {}

        for key in ['token_embedding', 'self_attention', 'moe_gating', 'expert_compute',
                    'expert_compute-cpu', 'expert_compute-gpu', 'expert_h2d', 'expert_d2h']:
            if key not in self.perf_stats:
                self.perf_stats[key] = []

    @staticmethod
    def _get_b_cpu(batch_size, cfg):
        pass
        bkv = batch_size * cfg.num_attention_heads
        cpu_group_ratio = 0.625
        if batch_size < 16:
            return None
        return max(1, min(bkv - 1, int(bkv * cpu_group_ratio)))

    def get_expert_module(self, layer_idx, expert_idx):
        pass
        return self.model.layers[layer_idx].mlp.experts[expert_idx]

    def calc_n_expert_on_gpu(self):
        n_param = sum(
            p.numel() for p in self.get_expert_module(1, 0).parameters()
        )
        expert_size_bytes = n_param * 2
        max_gpu_bytes = self.gpu_mem_budget_gb * 1024 ** 3
        current_allocated = torch.cuda.memory_allocated()
        reserved_bytes = 10 * 1024 ** 3
        available_bytes = max_gpu_bytes - current_allocated - reserved_bytes
        n_from_mem = int(available_bytes / expert_size_bytes)
        n_total = self.n_layer * self.n_expert
        return max(16, min(n_from_mem, n_total))

    def compute_attention_with_split_kv(self, layer, hidden_states, attention_mask,
                                        position_embeddings, layer_idx, device):
        pass
        if not hasattr(self, '_attn_timing'):
            self._attn_timing = {}
        self._attn_timing[layer_idx] = {}
        t0 = time.time()
        result = super().compute_attention_with_split_kv(
            layer, hidden_states, attention_mask, position_embeddings, layer_idx, device
        )
        self._attn_timing[layer_idx]['total'] = time.time() - t0
        return result

class BenchMixtral(FiddlerMixtralWithOffloading):
    pass

    gpu_mem_budget_gb = 32

    def __init__(self, args):
        self._cost_log_lock = threading.Lock()
        self._cost_log_file = open(os.devnull, "w")
        super().__init__(args)

    @staticmethod
    def _get_b_cpu(batch_size, cfg):
        pass
        bkv = batch_size * cfg.num_key_value_heads
        cpu_group_ratio = 0.5625
        if batch_size < 16:
            return None
        return max(1, min(bkv - 1, int(bkv * cpu_group_ratio)))

    def get_expert_module(self, layer_idx, expert_idx):
        pass
        return self.model.layers[layer_idx].block_sparse_moe.experts[expert_idx]

    def calc_n_expert_on_gpu(self):
        n_param = sum(
            p.numel() for p in self.get_expert_module(0, 0).parameters()
        )
        expert_size_bytes = n_param * 2
        max_gpu_bytes = self.gpu_mem_budget_gb * 1024 ** 3
        current_allocated = torch.cuda.memory_allocated()
        reserved_bytes = 10 * 1024 ** 3
        available_bytes = max_gpu_bytes - current_allocated - reserved_bytes
        n_from_mem = int(available_bytes / expert_size_bytes)
        n_total = self.n_layer * self.n_expert
        return max(16, min(n_from_mem, n_total))

    def compute_attention_with_split_kv(self, layer, hidden_states, attention_mask,
                                        position_embeddings, layer_idx, device):
        pass
        if not hasattr(self, '_attn_timing'):
            self._attn_timing = {}
        self._attn_timing[layer_idx] = {}
        t0 = time.time()
        result = super().compute_attention_with_split_kv(
            layer, hidden_states, attention_mask, position_embeddings, layer_idx, device
        )
        self._attn_timing[layer_idx]['total'] = time.time() - t0
        return result

def build_model(args):
    ns = argparse.Namespace(
        model=args.model,
        cpu_offload=1,
        batch_size=args.batch_size,
        cache=2,
        beam_width=1,
    )

    model_path_lower = args.model.lower()
    if "deepseek" in model_path_lower or "ds2" in model_path_lower:
        model = BenchDeepSeek(ns)
    elif "mixtral" in model_path_lower:
        model = BenchMixtral(ns)
    else:
        model = BenchQwen(ns)

    model.gpu_mem_budget_gb = args.gpu_mem_gb

    if hasattr(model, 'set_kv_cache_split_config'):
        model.set_kv_cache_split_config(mode="hybrid", ratio=0.3)
    return model

def real_prefill(model, prefill_tokens, max_seq=None):
    pass
    model.is_decode = False
    model.past_key_value = transformers.cache_utils.DynamicCache()
    model.past_key_values_length = 0

    cfg = model.model.config
    max_seq = max_seq if max_seq is not None else (prefill_tokens + 65536 + 16)

    if isinstance(model, BenchDeepSeek):
        model.inplace_kv = DS2InplaceKVCache(
            model.n_layer, model.batch_size,
            cfg.num_attention_heads,
            max_seq,
            k_head_dim=cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            dtype=model.dtype,
            b_cpu=model._get_b_cpu(model.batch_size, cfg),
        )
    else:
        model.inplace_kv = InplaceKVCache(
            model.n_layer, model.batch_size,
            cfg.num_key_value_heads,
            max_seq,
            getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads),
            b_cpu=model._get_b_cpu(model.batch_size, cfg),
        )

    input_ids = torch.randint(0, 1000, (model.batch_size, prefill_tokens),
                              dtype=torch.long, device=model.dev)
    position_ids = torch.arange(prefill_tokens, device=model.dev).unsqueeze(0).expand(
        model.batch_size, -1
    )
    attention_mask = torch.ones(
        model.batch_size, 1, prefill_tokens, prefill_tokens,
        dtype=torch.bool, device=model.dev
    )

    with torch.no_grad():

        if isinstance(model, BenchDeepSeek):
            forward_fn = model.mixtral_forward
        elif isinstance(model, BenchMixtral):
            forward_fn = model.mixtral_forward
        else:
            forward_fn = model.qwen_forward
        logits = forward_fn(input_ids, position_ids, attention_mask)

    model.is_decode = True
    model.past_key_values_length = prefill_tokens
    return logits

def seed_remaining_kv(model, current_depth, target_depth):
    pass
    cfg = model.model.config
    remaining = target_depth - current_depth
    if remaining <= 0:
        model.past_key_values_length = target_depth
        return

    if isinstance(model, BenchDeepSeek):
        k_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        v_head_dim = cfg.v_head_dim
        shape_k = (model.batch_size, cfg.num_attention_heads, remaining, k_head_dim)
        shape_v = (model.batch_size, cfg.num_attention_heads, remaining, v_head_dim)
        base_k = torch.randn(shape_k, dtype=torch.float32).to(torch.bfloat16)
        base_v = torch.randn(shape_v, dtype=torch.float32).to(torch.bfloat16)
    else:
        head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        shape = (model.batch_size, cfg.num_key_value_heads, remaining, head_dim)
        base_k = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        base_v = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)

    for layer_idx in range(model.n_layer):
        model.inplace_kv.append_4d(layer_idx, base_k.clone(), base_v.clone())

    model.past_key_values_length = target_depth

def free_preprocessed_kv(model):
    pass
    if hasattr(model, '_preprocessed_kv'):
        del model._preprocessed_kv
        import gc
        gc.collect()
        torch._C._host_emptyCache()

def preprocess_kv_for_decode(model, ratio, budget_gib, model_key, mode):
    pass
    assert mode in ("bkv", "kvlen", "pingpong", "twolayer"), f"_"
    free_preprocessed_kv(model)
    model._preprocessed_kv = {}

    cfg = model.model.config
    batch_size = model.batch_size
    num_kv_heads = cfg.num_key_value_heads
    bkv = batch_size * num_kv_heads

    print(f"_")

    for i_layer in range(model.n_layer):

        k_real, v_real = _extract_real_kv(model, i_layer)

        model.inplace_kv.key_cache[i_layer] = None
        model.inplace_kv.value_cache[i_layer] = None
        kv = k_real.shape[1]
        ekv = max(1, round(kv * ratio))
        k_eff = k_real[:, kv - ekv:, :]
        v_eff = v_real[:, kv - ekv:, :]
        rkv.BUDGET_BYTES = budget_gib * rkv.GiB

        if mode in ("bkv", "kvlen"):

            b_cpu_anchor, gs_anchor = rkv.TABLES[model_key][(batch_size, ekv)]

        if mode == "bkv":
            b_cpu = max(0, min(b_cpu_anchor, bkv))
            b_gpu = bkv - b_cpu
            n_res = rkv.n_resident_bkv(b_cpu, b_gpu, ekv)
            b_cpu_remain = b_cpu - n_res

            k_cpu_n1 = k_eff[:b_cpu_remain].clone() if b_cpu_remain > 0 else None
            v_cpu_n1 = v_eff[:b_cpu_remain].clone() if b_cpu_remain > 0 else None
            k_gpu_pin, v_gpu_pin = _prep_real_pingpong_chunks(k_eff[b_cpu:], v_eff[b_cpu:])

            model._preprocessed_kv[i_layer] = {
                'ekv': ekv,
                'k_cpu_n1': k_cpu_n1, 'v_cpu_n1': v_cpu_n1,
                'k_gpu_pin': k_gpu_pin, 'v_gpu_pin': v_gpu_pin,
                'b_cpu': b_cpu, 'b_gpu': b_gpu, 'n_res': n_res,
            }
        elif mode == "kvlen":
            gs = max(0, min(gs_anchor, ekv))
            n_res = rkv.n_resident_kvlen(bkv, gs, ekv)
            gs, n_res = rkv.align_gs_to_chunk(gs, n_res)
            cpu_tokens = ekv - gs - n_res

            k_cpu_n1 = k_eff[:, :cpu_tokens, :].clone() if cpu_tokens > 0 else None
            v_cpu_n1 = v_eff[:, :cpu_tokens, :].clone() if cpu_tokens > 0 else None
            k_gpu_pin, v_gpu_pin = _prep_real_pingpong_chunks(
                k_eff[:, cpu_tokens:cpu_tokens + gs, :], v_eff[:, cpu_tokens:cpu_tokens + gs, :]
            )

            model._preprocessed_kv[i_layer] = {
                'ekv': ekv,
                'k_cpu_n1': k_cpu_n1, 'v_cpu_n1': v_cpu_n1,
                'k_gpu_pin': k_gpu_pin, 'v_gpu_pin': v_gpu_pin,
                'gs': gs, 'n_res': n_res,
            }
        elif mode == "pingpong":

            n_res = rkv.n_resident_kvlen(bkv, 0, ekv)
            gs_final = ekv - n_res
            gs_final, n_res = rkv.align_gs_to_chunk(gs_final, n_res)

            k_gpu_pin, v_gpu_pin = _prep_real_pingpong_chunks(
                k_eff[:, :gs_final, :], v_eff[:, :gs_final, :]
            )

            model._preprocessed_kv[i_layer] = {
                'ekv': ekv,
                'k_gpu_pin': k_gpu_pin, 'v_gpu_pin': v_gpu_pin,
                'n_res': n_res,
            }
        else:

            n_res = rkv.n_resident_kvlen(bkv, 0, ekv)
            cpu_tokens = ekv - n_res

            k_cpu_n1 = k_eff[:, :cpu_tokens, :].clone() if cpu_tokens > 0 else None
            v_cpu_n1 = v_eff[:, :cpu_tokens, :].clone() if cpu_tokens > 0 else None

            model._preprocessed_kv[i_layer] = {
                'ekv': ekv,
                'k_cpu_n1': k_cpu_n1, 'v_cpu_n1': v_cpu_n1,
                'n_res': n_res,
            }

    print(f"_")

def _extract_real_kv(model, i_layer):
    pass
    k_part, v_part = model.inplace_kv.get(i_layer)
    if isinstance(k_part, tuple):
        k_cpu, k_gpu = k_part
        v_cpu, v_gpu = v_part
        return torch.cat([k_cpu, k_gpu], dim=0), torch.cat([v_cpu, v_gpu], dim=0)
    return k_part, v_part

def _prep_real_pingpong_chunks(k_slice, v_slice, chunk=None):
    pass
    chunk = chunk or rkv.CHUNK
    length = k_slice.shape[1]
    if length <= 0:
        return [], []
    n_chunks = (length + chunk - 1) // chunk
    k_chunks, v_chunks = [], []
    for i in range(n_chunks):
        s, e = i * chunk, min((i + 1) * chunk, length)
        k_chunks.append(k_slice[:, s:e, :].contiguous().pin_memory())
        v_chunks.append(v_slice[:, s:e, :].contiguous().pin_memory())
    return k_chunks, v_chunks

def _project_qk(model, layer, inps, position_embeddings, i_layer):
    pass
    cfg = model.model.config
    batch_size, seq_len, _ = inps.shape

    if isinstance(model, BenchDeepSeek):

        import sys as _sys
        _attn_mod = _sys.modules.get('transformers_modules.DeepSeek-V2-Lite.modeling_deepseek')
        if _attn_mod is None:
            raise RuntimeError("_")
        _rope_fn = _attn_mod.apply_rotary_pos_emb

        attn = layer.self_attn
        num_heads = cfg.num_attention_heads
        q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        bkv = batch_size * num_heads

        if attn.q_lora_rank is None:
            q = attn.q_proj(inps)
        else:
            q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(inps)))
        q = q.view(batch_size, seq_len, num_heads, q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
        del q

        past_len = model.inplace_kv.seq_len(i_layer)
        kv_seq_len = past_len + seq_len

        cos, sin = attn.rotary_emb(q_pe, seq_len=kv_seq_len)
        position_ids_tensor = torch.full(
            (batch_size, seq_len), past_len, dtype=torch.long, device=model.dev
        )

        dummy_k = torch.zeros_like(q_pe)
        q_pe, _ = _rope_fn(q_pe, dummy_k, cos, sin, position_ids_tensor)
        del dummy_k

        query_states = q_pe.new_empty(batch_size, num_heads, seq_len, q_head_dim)
        query_states[:, :, :, :cfg.qk_nope_head_dim] = q_nope
        query_states[:, :, :, cfg.qk_nope_head_dim:] = q_pe
        del q_nope, q_pe

        q_g = query_states.bfloat16().view(bkv, 1, q_head_dim)
        return q_g

    else:

        if isinstance(model, BenchMixtral):
            from transformers.models.mixtral.modeling_mixtral import apply_rotary_pos_emb
        else:
            from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // num_heads)
        groups = num_heads // num_kv_heads

        attn = layer.self_attn
        q = attn.q_proj(inps).view(batch_size, seq_len, num_heads, head_dim)
        k_new = attn.k_proj(inps).view(batch_size, seq_len, num_kv_heads, head_dim)
        if hasattr(attn, 'q_norm'):
            q = attn.q_norm(q)
        if hasattr(attn, 'k_norm'):
            k_new = attn.k_norm(k_new)
        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)
        cos, sin = position_embeddings
        q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        bkv = batch_size * num_kv_heads
        q_g = q.bfloat16().view(bkv, groups, head_dim)
        return q_g

def _get_pp_stream(bkv):
    pass
    key = (bkv, rkv.CHUNK)
    if key not in rkv._pp_bufs:
        rkv._pp_bufs[key] = {
            'k': [torch.empty(bkv, rkv.CHUNK, rkv.QHD, dtype=rkv.dtype, device=rkv.DEV),
                  torch.empty(bkv, rkv.CHUNK, rkv.QHD, dtype=rkv.dtype, device=rkv.DEV)],
            'v': [torch.empty(bkv, rkv.CHUNK, rkv.VHD, dtype=rkv.dtype, device=rkv.DEV),
                  torch.empty(bkv, rkv.CHUNK, rkv.VHD, dtype=rkv.dtype, device=rkv.DEV)],
            'stream': torch.cuda.Stream(device=rkv.DEV),
        }
    return rkv._pp_bufs[key]['stream']

def _attn_dims(model, layer, cfg, batch_size):
    pass
    if isinstance(model, BenchDeepSeek):
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_attention_heads
        out_head_dim = cfg.v_head_dim
        scale = layer.self_attn.softmax_scale
    else:
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        out_head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // num_heads)
        scale = out_head_dim ** -0.5
    bkv = batch_size * num_kv_heads
    return num_heads, bkv, out_head_dim, scale

def _decode_layer_pingpong(model, layer, i_layer, inps, position_embeddings, ratio, budget_gib):
    pass
    cfg = model.model.config
    batch_size, seq_len, _ = inps.shape
    num_heads, bkv, head_dim, scale = _attn_dims(model, layer, cfg, batch_size)

    q_g = _project_qk(model, layer, inps, position_embeddings, i_layer).to(model.dev)

    if hasattr(model, '_preprocessed_kv') and i_layer in model._preprocessed_kv:
        prep = model._preprocessed_kv[i_layer]
        k_chunks = prep['k_gpu_pin']
        v_chunks = prep['v_gpu_pin']
        n_res = prep['n_res']
        rkv.BUDGET_BYTES = budget_gib * rkv.GiB
    else:

        k_real, v_real = _extract_real_kv(model, i_layer)
        total = k_real.shape[1]
        ekv = max(1, round(total * ratio))
        k_eff = k_real[:, total - ekv:, :]
        v_eff = v_real[:, total - ekv:, :]

        rkv.BUDGET_BYTES = budget_gib * rkv.GiB
        n_res = rkv.n_resident_kvlen(bkv, 0, ekv)
        gs_final = ekv - n_res
        gs_final, n_res = rkv.align_gs_to_chunk(gs_final, n_res)

        k_chunks, v_chunks = _prep_real_pingpong_chunks(k_eff[:, :gs_final, :], v_eff[:, :gs_final, :])

    if n_res > 0:
        oa_r, ma_r, sa_r = rkv._resident_layer(q_g, bkv, n_res, scale)
    else:
        oa_r = torch.zeros(bkv, rkv.GROUPS, head_dim, device=model.dev, dtype=torch.float32)
        ma_r = torch.full((bkv, rkv.GROUPS, 1), float('-inf'), device=model.dev, dtype=torch.float32)
        sa_r = torch.zeros(bkv, rkv.GROUPS, 1, device=model.dev, dtype=torch.float32)

    oa_s, ma_s, sa_s = rkv._pingpong_stream(q_g, k_chunks, v_chunks, bkv, scale)
    oa, ma, sa = rkv._pairwise_merge(oa_r, ma_r, sa_r, oa_s, ma_s, sa_s)
    out = (oa / sa.clamp(min=1e-6)).bfloat16()

    attn_out = out.view(batch_size, num_heads, seq_len, head_dim)
    attn_out = attn_out.to(inps.dtype).transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    return layer.self_attn.o_proj(attn_out)

def _decode_layer_twolayer(model, layer, i_layer, inps, position_embeddings, ratio, budget_gib,
                           pin_cpu=True):
    pass
    cfg = model.model.config
    batch_size, seq_len, _ = inps.shape
    num_heads, bkv, head_dim, scale = _attn_dims(model, layer, cfg, batch_size)

    q_g = _project_qk(model, layer, inps, position_embeddings, i_layer).to(model.dev)

    if hasattr(model, '_preprocessed_kv') and i_layer in model._preprocessed_kv:
        prep = model._preprocessed_kv[i_layer]
        k_cpu = prep['k_cpu_n1']
        v_cpu = prep['v_cpu_n1']
        n_res = prep['n_res']
        cpu_tokens = 0 if k_cpu is None else k_cpu.shape[1]
        rkv.BUDGET_BYTES = budget_gib * rkv.GiB
    else:

        k_real, v_real = _extract_real_kv(model, i_layer)
        total = k_real.shape[1]
        ekv = max(1, round(total * ratio))
        k_eff = k_real[:, total - ekv:, :]
        v_eff = v_real[:, total - ekv:, :]

        rkv.BUDGET_BYTES = budget_gib * rkv.GiB
        n_res = rkv.n_resident_kvlen(bkv, 0, ekv)
        cpu_tokens = ekv - n_res
        k_cpu = k_eff[:, :cpu_tokens, :].contiguous()
        v_cpu = v_eff[:, :cpu_tokens, :].contiguous()

    cpu_res = {}
    def _cpu_fn():
        try:
            os.sched_setaffinity(0, rkv._NUMA1_CPUS if pin_cpu else rkv._CONTEND_CPUS)
        except Exception:
            pass
        q_cpu = q_g.cpu().contiguous()
        out_c, max_c, sum_c = rkv._local_bmm_softmax(q_cpu, k_cpu, v_cpu, scale)
        cpu_res.update(out=out_c, max=max_c, sum=sum_c)

    thr = None
    if cpu_tokens > 0:
        thr = threading.Thread(target=_cpu_fn, daemon=True)
        thr.start()

    if n_res > 0:
        oa_r, ma_r, sa_r = rkv._resident_layer(q_g, bkv, n_res, scale)
    else:
        oa_r = torch.zeros(bkv, rkv.GROUPS, head_dim, device=model.dev, dtype=torch.float32)
        ma_r = torch.full((bkv, rkv.GROUPS, 1), float('-inf'), device=model.dev, dtype=torch.float32)
        sa_r = torch.zeros(bkv, rkv.GROUPS, 1, device=model.dev, dtype=torch.float32)

    torch.cuda.synchronize(model.dev)
    if cpu_tokens > 0:
        thr.join()
        max_c = cpu_res['max'].to(model.dev)
        sum_c = cpu_res['sum'].to(model.dev)
        out_c = cpu_res['out'].to(model.dev)
        o_all, m_all, s_all = rkv._pairwise_merge(oa_r, ma_r, sa_r, out_c, max_c, sum_c)
        out = (o_all / s_all.clamp(min=1e-6)).bfloat16()
    else:
        out = (oa_r / sa_r.clamp(min=1e-6)).bfloat16()

    attn_out = out.view(batch_size, num_heads, seq_len, head_dim)
    attn_out = attn_out.to(inps.dtype).transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    return layer.self_attn.o_proj(attn_out)

def _ours_extract_and_slice(model, layer, i_layer, inps, position_embeddings, ratio, budget_gib):
    pass
    q_g = _project_qk(model, layer, inps, position_embeddings, i_layer).to(model.dev)
    k_real, v_real = _extract_real_kv(model, i_layer)
    kv = k_real.shape[1]
    ekv = max(1, round(kv * ratio))
    k_eff = k_real[:, kv - ekv:, :]
    v_eff = v_real[:, kv - ekv:, :]
    rkv.BUDGET_BYTES = budget_gib * rkv.GiB
    return q_g, k_eff, v_eff, ekv

def _ours_finish(out, batch_size, seq_len, num_heads, head_dim, inps_dtype, layer):
    attn_out = out.view(batch_size, num_heads, seq_len, head_dim)
    attn_out = attn_out.to(inps_dtype).transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    return layer.self_attn.o_proj(attn_out)

def _decode_layer_ours_bkv(model, layer, i_layer, inps, position_embeddings, ratio, budget_gib, model_key):
    pass
    cfg = model.model.config
    batch_size, seq_len, _ = inps.shape
    num_heads, bkv, head_dim, scale = _attn_dims(model, layer, cfg, batch_size)

    t0 = time.time()
    q_g = _project_qk(model, layer, inps, position_embeddings, i_layer).to(model.dev)
    torch.cuda.synchronize()
    _TIMING_BREAKDOWN['project_qk'] += time.time() - t0

    if hasattr(model, '_preprocessed_kv') and i_layer in model._preprocessed_kv:
        prep = model._preprocessed_kv[i_layer]
        k_cpu_n1 = prep['k_cpu_n1']
        v_cpu_n1 = prep['v_cpu_n1']
        k_gpu_pin = prep['k_gpu_pin']
        v_gpu_pin = prep['v_gpu_pin']
        b_cpu = prep['b_cpu']
        b_gpu = prep['b_gpu']
        n_res = prep['n_res']
        ekv = prep['ekv']
        rkv.BUDGET_BYTES = budget_gib * rkv.GiB
    else:

        t0 = time.time()
        k_real, v_real = _extract_real_kv(model, i_layer)
        kv = k_real.shape[1]
        ekv = max(1, round(kv * ratio))
        k_eff = k_real[:, kv - ekv:, :]
        v_eff = v_real[:, kv - ekv:, :]
        rkv.BUDGET_BYTES = budget_gib * rkv.GiB
        _TIMING_BREAKDOWN['extract_kv'] += time.time() - t0

        b_cpu_anchor, _gs_anchor = rkv.TABLES[model_key][(batch_size, ekv)]
        b_cpu = max(0, min(b_cpu_anchor, bkv))
        b_gpu = bkv - b_cpu
        n_res = rkv.n_resident_bkv(b_cpu, b_gpu, ekv)
        b_cpu_remain = b_cpu - n_res

        k_cpu_n1 = k_eff[:b_cpu_remain].contiguous() if b_cpu_remain > 0 else None
        v_cpu_n1 = v_eff[:b_cpu_remain].contiguous() if b_cpu_remain > 0 else None

        t0 = time.time()
        k_gpu_pin, v_gpu_pin = _prep_real_pingpong_chunks(k_eff[b_cpu:], v_eff[b_cpu:])
        _TIMING_BREAKDOWN['prep_chunks'] += time.time() - t0

    t0 = time.time()
    out = rkv.run_bkv_resident(
        batch_size, ekv, q_g, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin,
        b_cpu, b_gpu, n_res,
    )
    torch.cuda.synchronize()
    _TIMING_BREAKDOWN['run_bkv'] += time.time() - t0

    t0 = time.time()
    result = _ours_finish(out, batch_size, seq_len, num_heads, head_dim, inps.dtype, layer)
    torch.cuda.synchronize()
    _TIMING_BREAKDOWN['o_proj'] += time.time() - t0

    _TIMING_BREAKDOWN['count'] += 1
    return result

def _decode_layer_ours_kvlen(model, layer, i_layer, inps, position_embeddings, ratio, budget_gib, model_key):
    pass
    cfg = model.model.config
    batch_size, seq_len, _ = inps.shape
    num_heads, bkv, head_dim, scale = _attn_dims(model, layer, cfg, batch_size)

    if hasattr(model, '_preprocessed_kv') and i_layer in model._preprocessed_kv:
        prep = model._preprocessed_kv[i_layer]
        q_g = _project_qk(model, layer, inps, position_embeddings, i_layer).to(model.dev)
        k_cpu_n1 = prep['k_cpu_n1']
        v_cpu_n1 = prep['v_cpu_n1']
        k_gpu_pin = prep['k_gpu_pin']
        v_gpu_pin = prep['v_gpu_pin']
        gs = prep['gs']
        n_res = prep['n_res']
        ekv = prep['ekv']
        rkv.BUDGET_BYTES = budget_gib * rkv.GiB
    else:

        q_g, k_eff, v_eff, ekv = _ours_extract_and_slice(
            model, layer, i_layer, inps, position_embeddings, ratio, budget_gib
        )

        _b_cpu_anchor, gs_anchor = rkv.TABLES[model_key][(batch_size, ekv)]
        gs = max(0, min(gs_anchor, ekv))
        n_res = rkv.n_resident_kvlen(bkv, gs, ekv)
        gs, n_res = rkv.align_gs_to_chunk(gs, n_res)
        cpu_tokens = ekv - gs - n_res

        k_cpu_n1 = k_eff[:, :cpu_tokens, :].contiguous() if cpu_tokens > 0 else None
        v_cpu_n1 = v_eff[:, :cpu_tokens, :].contiguous() if cpu_tokens > 0 else None
        k_gpu_pin, v_gpu_pin = _prep_real_pingpong_chunks(
            k_eff[:, cpu_tokens:cpu_tokens + gs, :], v_eff[:, cpu_tokens:cpu_tokens + gs, :]
        )

    out = rkv.run_kvlen_resident(
        batch_size, ekv, q_g, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin, gs, n_res,
    )
    return _ours_finish(out, batch_size, seq_len, num_heads, head_dim, inps.dtype, layer)

def load_expert_dist(model_key="qwen3"):
    pass
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"expert_dist_{model_key}.json")
    with open(path) as f:
        data = json.load(f)
    data["steps"] = [s for s in data["steps"] if s.get("step") != "prefill"]
    return data

def _pick_cpu_expert_id(model, i_layer):
    pass
    if isinstance(model, BenchDeepSeek) and i_layer == 0:
        return 0
    if not hasattr(model, "_cpu_expert_id_cache"):
        model._cpu_expert_id_cache = {}
    if i_layer not in model._cpu_expert_id_cache:
        cpu_ids = [e for e in range(model.n_expert) if model.expert_loc[i_layer, e] == 0]
        model._cpu_expert_id_cache[i_layer] = cpu_ids[0] if cpu_ids else 0
    return model._cpu_expert_id_cache[i_layer]

def simulate_expert_compute(model, i_layer, step_idx, expert_dist, timing, mixtral_style=False,
                            shared_stream=None, contend_cpu=False):
    pass
    hidden_dim = model.model.config.hidden_size
    dev = model.dev
    dtype = torch.bfloat16

    steps = expert_dist["steps"]
    step_rec = steps[step_idx % len(steps)]
    layer_rec = step_rec["layers"][i_layer]
    hit_counts = layer_rec["hit"]
    miss_counts = layer_rec["miss"]

    t0 = time.time()
    for n_tok in hit_counts:
        if n_tok <= 0:
            continue
        x = torch.randn(n_tok, hidden_dim, dtype=dtype, device=dev)
        model.expert_placeholder(x)
    torch.cuda.synchronize(dev)
    hit_elapsed = time.time() - t0
    timing.setdefault("expert_hit", 0.0)
    timing["expert_hit"] += hit_elapsed
    timing["expert"] += hit_elapsed

    cpu_expert_id = _pick_cpu_expert_id(model, i_layer)

    if not hasattr(model, '_sim_fixed_experts'):
        _src_layer = 1 if model.n_layer > 1 else 0
        _pin_raw = model.get_expert_module(_src_layer, 3).state_dict()
        _pinned = {k: v.pin_memory() if not v.is_pinned() else v for k, v in _pin_raw.items()}
        import intel_extension_for_pytorch as ipex
        _cpu_exp1 = ipex.optimize(model.get_expert_module(_src_layer, 1).to('cpu'),
                                   dtype=torch.bfloat16, auto_kernel_selection=True)
        _cpu_exp2 = ipex.optimize(model.get_expert_module(_src_layer, 2).to('cpu'),
                                   dtype=torch.bfloat16, auto_kernel_selection=True)
        model._sim_fixed_experts = {'pinned_state': _pinned,
                                     'cpu_experts': [_cpu_exp1, _cpu_exp2], 'pp_idx': 0}
    _cpu_pool = model._sim_fixed_experts['cpu_experts']

    if mixtral_style:
        if contend_cpu:
            try:
                os.sched_setaffinity(0, rkv._CONTEND_CPUS)
            except Exception:
                pass
        t0 = time.time()
        for n_tok in miss_counts:
            if n_tok <= 0:
                continue
            x = torch.randn(n_tok, hidden_dim, dtype=dtype, device="cpu")
            model._sim_fixed_experts['pp_idx'] = (model._sim_fixed_experts['pp_idx'] + 1) % 2
            _cpu_pool[model._sim_fixed_experts['pp_idx']](x)
        miss_elapsed = time.time() - t0
        timing.setdefault("expert_miss_cpu", 0.0)
        timing["expert_miss_cpu"] += miss_elapsed
        timing["expert"] += miss_elapsed
        return

    lock = threading.Lock()
    queue = sorted(miss_counts)
    gpu_elapsed = [0.0]
    cpu_elapsed = [0.0]

    if isinstance(model, BenchDeepSeek) and i_layer == 0:

        return

    if not hasattr(model, '_sim_fixed_experts'):
        _src_layer = 1 if model.n_layer > 1 else 0

        _pin_raw = model.get_expert_module(_src_layer, 3).state_dict()
        _pinned = {k: v.pin_memory() if not v.is_pinned() else v for k, v in _pin_raw.items()}

        import intel_extension_for_pytorch as ipex
        _cpu_exp1 = ipex.optimize(model.get_expert_module(_src_layer, 1).to('cpu'),
                                   dtype=torch.bfloat16, auto_kernel_selection=True)
        _cpu_exp2 = ipex.optimize(model.get_expert_module(_src_layer, 2).to('cpu'),
                                   dtype=torch.bfloat16, auto_kernel_selection=True)
        model._sim_fixed_experts = {'pinned_state': _pinned,
                                     'cpu_experts': [_cpu_exp1, _cpu_exp2], 'pp_idx': 0}
    real_cpu_state = model._sim_fixed_experts['pinned_state']
    _cpu_pool = model._sim_fixed_experts['cpu_experts']

    def _gpu_od_thread():
        t0 = time.time()
        stream_cm = torch.cuda.stream(shared_stream) if shared_stream is not None\
            else contextlib.nullcontext()
        with stream_cm:
            while True:
                with lock:
                    if not queue:
                        break
                    n_tok = queue.pop()
                if n_tok <= 0:
                    continue
                model.expert_placeholder2.load_state_dict(real_cpu_state)
                x = torch.randn(n_tok, hidden_dim, dtype=dtype, device=dev)
                model.expert_placeholder2(x)
        if shared_stream is not None:
            shared_stream.synchronize()
        else:
            torch.cuda.synchronize(dev)
        gpu_elapsed[0] = time.time() - t0

    def _cpu_thread():
        if contend_cpu:
            try:
                os.sched_setaffinity(0, rkv._CONTEND_CPUS)
            except Exception:
                pass
        t0 = time.time()
        while True:
            with lock:
                if not queue:
                    break
                n_tok = queue.pop(0)
            if n_tok <= 0:
                continue
            x = torch.randn(n_tok, hidden_dim, dtype=dtype, device="cpu")

            model._sim_fixed_experts['pp_idx'] = (model._sim_fixed_experts['pp_idx'] + 1) % 2
            _cpu_pool[model._sim_fixed_experts['pp_idx']](x)
        cpu_elapsed[0] = time.time() - t0

    tg = threading.Thread(target=_gpu_od_thread, daemon=True)
    tc = threading.Thread(target=_cpu_thread, daemon=True)
    tg.start()
    tc.start()
    tg.join()
    tc.join()

    timing.setdefault("expert_miss_gpu_od", 0.0)
    timing["expert_miss_gpu_od"] += gpu_elapsed[0]
    timing.setdefault("expert_miss_cpu", 0.0)
    timing["expert_miss_cpu"] += cpu_elapsed[0]

    timing["expert"] += max(gpu_elapsed[0], cpu_elapsed[0])

@torch.no_grad()
def minimal_decode_step(model, hidden_states, attention_mask, timing, baseline="ours",
                         expert_dist=None, step_idx=0, mixtral_style=False, no_overlap=False):
    pass
    hidden_dim = model.model.config.hidden_size
    batch_size, seq_len, _ = hidden_states.shape
    inps = hidden_states

    if hasattr(model, '_attn_timing'):
        model._attn_timing.clear()

    baseline_cfg = BASELINE_CONFIG.get(baseline)

    for i_layer, layer in enumerate(model.model.layers):
        inps_residual = inps
        inps = _rms_norm_bf16(
            inps, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon
        )
        inps = inps.view(batch_size, seq_len, hidden_dim)

        if isinstance(model, BenchDeepSeek):
            position_embeddings = None
        else:
            position_embeddings = model.model.rotary_emb(
                inps,
                torch.full(
                    (batch_size, seq_len), model.past_key_values_length,
                    dtype=torch.long, device=model.dev,
                ),
            )

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        t0_cpu = time.time()

        serial_mode = no_overlap or baseline in ("ours-bkv", "ours-kvlen", "fastdecode")

        contend_cpu = expert_dist is not None and baseline in ("scout", "hybridgen") and not serial_mode
        shared_od_stream = None
        if expert_dist is not None and baseline == "solid" and not serial_mode:
            num_kv_heads = model.model.config.num_attention_heads if isinstance(model, BenchDeepSeek) else model.model.config.num_key_value_heads
            shared_od_stream = _get_pp_stream(batch_size * num_kv_heads)

        skip_expert_simulation = isinstance(model, BenchDeepSeek) and i_layer == 0

        expert_thread = None
        if expert_dist is not None and not serial_mode and not skip_expert_simulation:

            layer_idx_in_dist = i_layer if not isinstance(model, BenchDeepSeek) else (i_layer - 1)
            expert_thread = threading.Thread(
                target=simulate_expert_compute,
                args=(model, layer_idx_in_dist, step_idx, expert_dist, timing),
                kwargs=dict(mixtral_style=mixtral_style, shared_stream=shared_od_stream,
                            contend_cpu=contend_cpu),
                daemon=True,
            )
            expert_thread.start()

        if baseline == "ours":
            attn_output, _ = model.compute_attention_with_split_kv(
                layer=layer,
                hidden_states=inps,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                layer_idx=i_layer,
                device=model.dev,
            )
        elif baseline == "ours-bkv":
            attn_output = _decode_layer_ours_bkv(
                model, layer, i_layer, inps, position_embeddings,
                baseline_cfg["ratio"], baseline_cfg["budget_gib"], baseline_cfg["model_key"],
            )
        elif baseline == "ours-kvlen":
            attn_output = _decode_layer_ours_kvlen(
                model, layer, i_layer, inps, position_embeddings,
                baseline_cfg["ratio"], baseline_cfg["budget_gib"], baseline_cfg["model_key"],
            )
        elif baseline_cfg["arch"] == "pingpong":
            attn_output = _decode_layer_pingpong(
                model, layer, i_layer, inps, position_embeddings,
                baseline_cfg["ratio"], baseline_cfg["budget_gib"],
            )
        else:
            attn_output = _decode_layer_twolayer(
                model, layer, i_layer, inps, position_embeddings,
                baseline_cfg["ratio"], baseline_cfg["budget_gib"],
                pin_cpu=not contend_cpu,
            )

        ev_end.record()
        torch.cuda.synchronize()
        elapsed_cpu = time.time() - t0_cpu
        elapsed_gpu = ev_start.elapsed_time(ev_end) / 1000.0

        if expert_thread is not None:
            expert_thread.join()

        if serial_mode and expert_dist is not None and not skip_expert_simulation:
            layer_idx_in_dist = i_layer if not isinstance(model, BenchDeepSeek) else (i_layer - 1)
            simulate_expert_compute(
                model, layer_idx_in_dist, step_idx, expert_dist, timing,
                mixtral_style=mixtral_style, shared_stream=None,
            )

        timing["attn"] += elapsed_cpu
        timing.setdefault("attn_cpu_wall", 0.0)
        timing["attn_cpu_wall"] += elapsed_cpu
        timing.setdefault("attn_gpu_time", 0.0)
        timing["attn_gpu_time"] += elapsed_gpu

        if hasattr(model, '_attn_timing') and i_layer in model._attn_timing:
            for k, v in model._attn_timing[i_layer].items():
                timing.setdefault(f"attn_{k}", 0.0)
                timing[f"attn_{k}"] += v

        inps = inps_residual + attn_output
        inps_residual = inps
        inps = _rms_norm_bf16(
            inps, layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
        )

        torch.cuda.synchronize()
        t0 = time.time()
        if isinstance(model, BenchDeepSeek) and i_layer == 0:
            expert_out = layer.mlp(inps.view(-1, hidden_dim))
        else:
            expert_out = model.expert_placeholder(inps.view(-1, hidden_dim))
        torch.cuda.synchronize()
        timing["expert"] += time.time() - t0

        inps = inps_residual + expert_out.view(batch_size, seq_len, hidden_dim)

    return inps

def setup_kv(model, prefill_tokens, target_kv_len):
    pass
    real_prefill(model, prefill_tokens, max_seq=target_kv_len + 32)
    seed_remaining_kv(model, prefill_tokens, target_kv_len)

def run_single(model, batch_size, target_kv_len, prefill_tokens, strategy, baseline,
               expert_dist, mixtral_style, warmup, iters, setup=True, no_overlap=False):
    pass
    if setup:
        setup_kv(model, prefill_tokens, target_kv_len)

        mode = BASELINE_TO_PREP_MODE.get(baseline)
        if mode is not None:
            baseline_cfg = BASELINE_CONFIG[baseline]
            preprocess_kv_for_decode(model, baseline_cfg["ratio"], baseline_cfg["budget_gib"],
                                    baseline_cfg["model_key"], mode)

    cfg = model.model.config
    hidden_states = torch.randn(
        batch_size, 1, cfg.hidden_size, dtype=torch.bfloat16, device=model.dev
    )
    attention_mask = torch.ones(
        batch_size, 1, 1, target_kv_len + 1, dtype=torch.bool, device=model.dev
    )

    if baseline == "ours":

        timing_prime = {"attn": 0.0, "expert": 0.0}
        minimal_decode_step(model, hidden_states, attention_mask, timing_prime, baseline=baseline,
                             expert_dist=None, step_idx=0, mixtral_style=mixtral_style,
                             no_overlap=no_overlap)

    for w in range(warmup):
        timing = {"attn": 0.0, "expert": 0.0}
        minimal_decode_step(model, hidden_states, attention_mask, timing, baseline=baseline,
                             expert_dist=expert_dist, step_idx=w, mixtral_style=mixtral_style,
                             no_overlap=no_overlap)

    totals = {"attn": 0.0, "expert": 0.0}
    step_times = []
    for it in range(iters):
        timing = {"attn": 0.0, "expert": 0.0}
        t0 = time.time()
        minimal_decode_step(model, hidden_states, attention_mask, timing, baseline=baseline,
                             expert_dist=expert_dist, step_idx=warmup + it,
                             mixtral_style=mixtral_style, no_overlap=no_overlap)
        step_times.append(time.time() - t0)
        for k, v in timing.items():
            totals[k] = totals.get(k, 0.0) + v

    n = iters
    avg_step = sum(step_times) / n
    result = {
        "batch_size": batch_size, "target_kv_len": target_kv_len,
        "prefill_tokens": prefill_tokens, "strategy": strategy, "baseline": baseline,
        "avg_step_ms": avg_step * 1000,
    }
    for k, v in totals.items():
        result[f"{k}_ms"] = v / n * 1000
    result["other_ms"] = result["avg_step_ms"] - result["attn_ms"] - result["expert_ms"]
    return result

def print_result(result):
    print("_")
    print(f"batch_size={result['batch_size']} target_kv_len={result['target_kv_len']} "
          f"prefill_tokens={result['prefill_tokens']} strategy={result['strategy']} "
          f"baseline={result['baseline']}")
    avg_step = result["avg_step_ms"]
    print(f"_")

    avg_attn = result["attn_ms"]
    avg_expert = result["expert_ms"]
    print(f"  attention:    {avg_attn:.2f} ms ({avg_attn/avg_step*100:.1f}%)")

    if "attn_cpu_wall_ms" in result:
        print(f"    └─ CPU wall time: {result['attn_cpu_wall_ms']:.2f} ms")
    if "attn_gpu_time_ms" in result:
        v = result["attn_gpu_time_ms"]
        print(f"    └─ GPU kernel time: {v:.2f} ms")
        if "attn_cpu_wall_ms" in result:
            cpu_wall = result["attn_cpu_wall_ms"]
            overhead = cpu_wall - v
            print(f"    └─ CPU overhead (wall-GPU): {overhead:.2f} ms ({overhead/cpu_wall*100:.1f}%)")

    for k in sorted(result.keys()):
        if k.startswith("attn_") and k not in ("attn_ms", "attn_cpu_wall_ms", "attn_gpu_time_ms"):
            print(f"    └─ {k[5:-3]}: {result[k]:.2f} ms")

    print(f"  expert(GPU):  {avg_expert:.2f} ms ({avg_expert/avg_step*100:.1f}%)")
    for k in ("expert_hit_ms", "expert_miss_gpu_od_ms", "expert_miss_cpu_ms"):
        if k in result:
            print(f"    └─ {k[7:-3]}: {result[k]:.2f} ms")
    print(f"_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="/home/share/data/model/Qwen3-30B-A3B")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--target_kv_len", type=int, default=65536,
                    help="_")
    ap.add_argument("--prefill_tokens", type=int, default=64,
                    help="_")
    ap.add_argument("--strategy", type=str, choices=["bkv", "kvlen"], default="bkv")
    ap.add_argument("--gpu_mem_gb", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--baseline", type=str,
                    choices=["ours", "ours-bkv", "ours-kvlen", "solid", "scout", "hybridgen", "fastdecode"],
                    default="ours-kvlen",
                    help="_"
                         "_"
                         "_"
                         "_")
    ap.add_argument("--expert_dist_model", type=str, default="qwen3",
                    help="_"
                         "_")
    ap.add_argument("--no_overlap", action="store_true",
                    help="_"
                         "_"
                         "_")
    args = ap.parse_args()

    model_path_lower = args.model.lower()
    if "deepseek" in model_path_lower or "ds2" in model_path_lower:
        model_key = "ds2"
    elif "mixtral" in model_path_lower:
        model_key = "mixtral"
    else:
        model_key = "qwen3"
    rkv.set_model(model_key)

    for bl in ["ours-bkv", "ours-kvlen", "solid", "scout", "hybridgen", "fastdecode"]:
        BASELINE_CONFIG[bl]["model_key"] = model_key

    expert_dist = None
    mixtral_style = False
    if args.expert_dist_model != "none":
        print(f"_")
        expert_dist = load_expert_dist(args.expert_dist_model)
        mixtral_style = (args.expert_dist_model == "mixtral")

    print(f"_")
    model = build_model(args)

    if BASELINE_CONFIG.get(args.baseline, {}).get("arch") == "twolayer":

        torch.set_num_threads(16)

    print(f"_")
    result = run_single(
        model, args.batch_size, args.target_kv_len, args.prefill_tokens, args.strategy,
        args.baseline, expert_dist, mixtral_style, args.warmup, args.iters,
        no_overlap=args.no_overlap,
    )
    print_result(result)

    if hasattr(model.inplace_kv, '_append_timing'):
        at = model.inplace_kv._append_timing
        if at['total']:
            avg_append_total = sum(at['total']) / len(at['total'])
            avg_append_reshape = sum(at['reshape']) / len(at['reshape'])
            avg_append_copy = sum(at['copy']) / len(at['copy'])
            print(f"_")
            print(f"  total:   {avg_append_total*1000:.3f} ms")
            print(f"  reshape: {avg_append_reshape*1000:.3f} ms")
            print(f"  copy:    {avg_append_copy*1000:.3f} ms")
            print(f"_")

    if '_TIMING_BREAKDOWN' in globals() and _TIMING_BREAKDOWN['count'] > 0:
        print("\n" + "="*70)
        print("_")
        print("="*70)

        count = _TIMING_BREAKDOWN['count']
        total_attn = sum(_TIMING_BREAKDOWN[k] for k in ['project_qk', 'extract_kv', 'prep_chunks', 'run_bkv', 'o_proj'])

        print(f"_")
        print("-"*70)

        for key in ['project_qk', 'extract_kv', 'prep_chunks', 'run_bkv', 'o_proj']:
            total_ms = _TIMING_BREAKDOWN[key] * 1000
            per_layer = total_ms / count if count > 0 else 0
            pct = (_TIMING_BREAKDOWN[key] / total_attn * 100) if total_attn > 0 else 0
            print(f"{key:<20} {total_ms:>10.2f}   {per_layer:>12.2f}        {pct:>5.1f}%")

        print("-"*70)
        print(f"_")
        print(f"_")
        print("="*70)

if __name__ == "__main__":
    main()

