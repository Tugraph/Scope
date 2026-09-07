import copy
import threading
import time
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
import threading

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FLASH_AVAILABLE = True
except ImportError:
    _flash_attn_func = None
    _FLASH_AVAILABLE = False

def _rms_norm_bf16(hidden_states, weight, eps=1e-5):
    pass
    input_dtype = hidden_states.dtype

    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states

def _alloc_numa_pinned(shape, dtype, numa_node):
    pass
    import ctypes, ctypes.util
    nbytes = 1
    for s in shape:
        nbytes *= s
    nbytes *= torch.tensor([], dtype=dtype).element_size()

    lib_name = ctypes.util.find_library('numa')
    if lib_name:
        try:
            _numa = ctypes.CDLL(lib_name)
            _numa.numa_alloc_onnode.restype = ctypes.c_void_p
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

class InplaceKVCache:
    pass
    def __init__(self, num_layers, batch_size, num_kv_heads, max_seq, head_dim,
                 dtype=torch.bfloat16, b_cpu=None):
        self.key_cache   = [None] * num_layers
        self.value_cache = [None] * num_layers
        self._filled     = [0] * num_layers
        self._bs         = batch_size
        self._kv_h       = num_kv_heads
        self._max_seq    = max_seq
        self._head_dim   = head_dim
        self._bkv        = batch_size * num_kv_heads
        self._dtype      = dtype

        self._b_cpu      = b_cpu

    def _ensure_buffer(self, layer_idx):
        if self.key_cache[layer_idx] is None:
            b_cpu = self._b_cpu
            bkv   = self._bkv
            ms    = self._max_seq
            hd    = self._head_dim
            dt    = self._dtype

            if b_cpu is not None and 0 < b_cpu < bkv:

                k_cpu = _alloc_numa_pinned((b_cpu,       ms, hd), dt, numa_node=1)
                k_gpu = _alloc_numa_pinned((bkv - b_cpu, ms, hd), dt, numa_node=0)
                v_cpu = _alloc_numa_pinned((b_cpu,       ms, hd), dt, numa_node=1)
                v_gpu = _alloc_numa_pinned((bkv - b_cpu, ms, hd), dt, numa_node=0)

                self.key_cache[layer_idx]   = (k_cpu, k_gpu)
                self.value_cache[layer_idx] = (v_cpu, v_gpu)
            else:
                shape = (bkv, ms, hd)
                self.key_cache[layer_idx]   = torch.empty(shape, dtype=dt).pin_memory()
                self.value_cache[layer_idx] = torch.empty(shape, dtype=dt).pin_memory()

    def _is_split(self, layer_idx):
        return isinstance(self.key_cache[layer_idx], tuple)

    def append_4d(self, layer_idx, k_4d, v_4d):
        pass
        new_seq = k_4d.shape[2]
        self._ensure_buffer(layer_idx)
        t   = self._filled[layer_idx]
        bkv = self._bkv
        hd  = self._head_dim
        k_flat = k_4d.contiguous().reshape(bkv, new_seq, hd)
        v_flat = v_4d.contiguous().reshape(bkv, new_seq, hd)

        if self._is_split(layer_idx):
            b_cpu = self._b_cpu
            k_c, k_g = self.key_cache[layer_idx]
            v_c, v_g = self.value_cache[layer_idx]
            k_c[:, t:t+new_seq, :].copy_(k_flat[:b_cpu])
            k_g[:, t:t+new_seq, :].copy_(k_flat[b_cpu:])
            v_c[:, t:t+new_seq, :].copy_(v_flat[:b_cpu])
            v_g[:, t:t+new_seq, :].copy_(v_flat[b_cpu:])
        else:
            self.key_cache[layer_idx][:, t:t+new_seq, :].copy_(k_flat)
            self.value_cache[layer_idx][:, t:t+new_seq, :].copy_(v_flat)

        self._filled[layer_idx] = t + new_seq

    def get(self, layer_idx):
        pass
        t = self._filled[layer_idx]
        if self._is_split(layer_idx):
            k_c, k_g = self.key_cache[layer_idx]
            v_c, v_g = self.value_cache[layer_idx]
            return (k_c[:, :t, :], k_g[:, :t, :]), (v_c[:, :t, :], v_g[:, :t, :])
        return (self.key_cache[layer_idx][:, :t, :],
                self.value_cache[layer_idx][:, :t, :])

    def seq_len(self, layer_idx):
        return self._filled[layer_idx]

    def reset(self):
        for i in range(len(self.key_cache)):
            self.key_cache[i]   = None
            self.value_cache[i] = None
            self._filled[i]     = 0

class CacheScopeQwen:
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.hot_experts = {}
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            use_cache=True,
            trust_remote_code=True,
        )
        self.cache =args.cache
        self.batch_size = args.batch_size
        if self.batch_size==4:
           self.cache=5
        else:
           self.cache=7
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].mlp.experts[0]
        ).to(self.dev)
        self.expert_placeholder2 = copy.deepcopy(
            self.model.layers[0].mlp.experts[1]
        ).to(self.dev)
        self.expert_placeholder3 = copy.deepcopy(
            self.model.layers[0].mlp.experts[2]
        ).to(self.dev)
        self.expert_placeholder4 = copy.deepcopy(
            self.model.layers[0].mlp.experts[3]
        ).to(self.dev)
        self.prefil_pre=False
        self.expert_placeholder5 = copy.deepcopy(
            self.model.layers[0].mlp.experts[5]
        ).to(self.dev)
        self.expert_placeholder6 = copy.deepcopy(
            self.model.layers[0].mlp.experts[6]
        ).to(self.dev)
        self.expert_placeholder5_inused = False
        self.expert_placeholder6_inused = False

        self.expert_to_placeholder = {}
        self.expert_placeholder_inused=False
        self.expert_placeholder2_inused=False
        self.expert_placeholder3_inused=False
        self.expert_placeholder4_inused=False
        self.placeholder_lock = threading.Lock()
        self.prefetch_layers=0
        self.is_decode = False
        self.placeholder_to_expert = {
            'expert_placeholder': None,
            'expert_placeholder2': None,
            'expert_placeholder3': None,
            'expert_placeholder4': None
        }
        self.expert_loading_status = {
            'expert_placeholder': True,
            'expert_placeholder2': True,
            'expert_placeholder3': True,
            'expert_placeholder4': True
        }
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].mlp.experts)

        self.expert_selection_stats = []
        self.expert_time_stats = []
        self.prefetch_list = {}
        self.prefetching_list = {}
        self.expert_selection_history = {}
        self.hit_stats = {}
        for i in range(self.n_layer):
            self.expert_selection_history[i] = []
            self.hit_stats[i] = {'hits': 0, 'total': 0}

        self.expert_weight_accumulator = {}
        for i in range(self.n_layer):
            self.expert_weight_accumulator[i] = torch.zeros(self.n_expert, device=self.dev)

        self.cpu_expert_time_per_layer = {i: 0.0 for i in range(self.n_layer)}
        self.gpu_expert_time_per_layer = {i: 0.0 for i in range(self.n_layer)}

        self.detail_gpu_gemm      = {i: 0.0 for i in range(self.n_layer)}
        self.detail_od_h2d        = {i: 0.0 for i in range(self.n_layer)}
        self.detail_od_gemm       = {i: 0.0 for i in range(self.n_layer)}
        self.detail_cpu_d2h       = {i: 0.0 for i in range(self.n_layer)}
        self.detail_cpu_gemm      = {i: 0.0 for i in range(self.n_layer)}
        self.detail_cpu_h2d       = {i: 0.0 for i in range(self.n_layer)}
        self.detail_n_od          = {i: 0   for i in range(self.n_layer)}
        self.detail_n_cpu         = {i: 0   for i in range(self.n_layer)}

        self.detail_gpu_wall      = {i: 0.0 for i in range(self.n_layer)}
        self.detail_cpu_wall      = {i: 0.0 for i in range(self.n_layer)}
        self.pf_gpu_wall          = {i: 0.0 for i in range(self.n_layer)}
        self.pf_cpu_wall          = {i: 0.0 for i in range(self.n_layer)}

        self.pf_gpu_gemm          = {i: 0.0 for i in range(self.n_layer)}
        self.pf_cpu_gemm          = {i: 0.0 for i in range(self.n_layer)}
        self.pf_od_h2d            = {i: 0.0 for i in range(self.n_layer)}
        self.pf_od_gemm           = {i: 0.0 for i in range(self.n_layer)}
        self.pf_n_gpu             = {i: 0   for i in range(self.n_layer)}
        self.pf_n_cpu             = {i: 0   for i in range(self.n_layer)}
        self.pf_n_od              = {i: 0   for i in range(self.n_layer)}

        self._cm_e_decode  = 1.5
        self._cm_e_prefill = 2.0
        self._cm_alpha = 0.0

        self._cm_coeff_table = [
            (8,    1.4),
            (64,   2.5),
            (512,  4.0),
            (float('inf'), 4.5),
        ]

        self.detail_cpu_kernel    = {i: 0.0 for i in range(self.n_layer)}
        self.detail_cpu_kernel_ntok = {i: [] for i in range(self.n_layer)}
        self.pf_cpu_kernel        = {i: 0.0 for i in range(self.n_layer)}
        self.pf_cpu_kernel_ntok   = {i: [] for i in range(self.n_layer)}

        self.attention_device_per_layer = ["cpu"] * self.n_layer

        for i in range(min(3, self.n_layer)):
            self.attention_device_per_layer[i] = "cuda:0"

        self.kv_cache_split_ratio = 0.3
        self.kv_cache_split_mode = "hybrid"
        self.prefill_chunk_size = 256
        self._split_ratio_adapt = True

        self.decode_group_size = None

        self.cpu_group_ratio = 0.5625
        self.cpu_pipeline_ratio = 0.0
        self.cpu_pipeline_adapt = False

        self._decode_kv_buffers = None

        self.latency_cpu = 5
        self.latency_gpu = 45

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.expert_loc_now = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)

        self._ipex_enabled = False
        try:
            import intel_extension_for_pytorch as ipex
            n_optimized = 0
            for i_layer in range(self.n_layer):
                for i_expert in range(self.n_expert):
                    if self.expert_loc[i_layer, i_expert] == 0:
                        expert = self.model.layers[i_layer].mlp.experts[i_expert]
                        self.model.layers[i_layer].mlp.experts[i_expert] = ipex.optimize(
                            expert, dtype=torch.bfloat16, auto_kernel_selection=True
                        )
                        n_optimized += 1
            self._ipex_enabled = True
            print(f"_")
        except Exception as e:
            print(f"_")

        self._numa_bound = False
        self._numa1_cpus = list(range(24, 48)) + list(range(72, 96))
        try:
            numa0_cpus = list(range(0, 24)) + list(range(48, 72))
            os.sched_setaffinity(0, numa0_cpus)
            torch.set_num_threads(len(numa0_cpus))
            self._numa_bound = True
            print(f"_")
        except Exception as e:
            print(f"_")

        _n_pinned = 0
        for i_layer in range(self.n_layer):
            for i_expert in range(self.n_expert):
                if self.expert_loc[i_layer, i_expert] == 0:
                    expert = self.model.layers[i_layer].mlp.experts[i_expert]
                    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                        try:
                            proj = getattr(expert, proj_name)
                            w = proj.weight.data
                            if not w.is_pinned():
                                proj.weight.data = w.pin_memory()
                            _n_pinned += 1
                        except Exception:
                            pass
        print(f"_")

        _H = self.model.config.hidden_size
        _warmup_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        _warmup_inps = [torch.zeros(n, _H, dtype=torch.bfloat16) for n in _warmup_sizes]
        _n_warmed = 0
        for i_layer in range(self.n_layer):
            for i_expert in range(self.n_expert):
                if self.expert_loc[i_layer, i_expert] == 0:
                    try:
                        with torch.no_grad():
                            exp = self.model.layers[i_layer].mlp.experts[i_expert]
                            for _wi in _warmup_inps:
                                exp(_wi)
                        _n_warmed += 1
                    except Exception:
                        pass
        print(f"_")

        self.layer_time_stats = []
        self.layer_time_accumulator = {}
        for i in range(self.n_layer):
            self.layer_time_accumulator[i] = 0.0
        self.layer_time_details = {
            'all_gpu': [],
            'all_cpu': [],
            'mixed': []
        }

        self.layer_time_accumulator_details = {
            'all_gpu': {i: 0.0 for i in range(self.n_layer)},
            'all_cpu': {i: 0.0 for i in range(self.n_layer)},
            'mixed': {i: 0.0 for i in range(self.n_layer)}
        }
        self.last_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(self.n_layer)
        }
        self.current_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(self.n_layer)
        }

        self.layer_data = {}

        self.set_kv_cache_split_config(mode="hybrid", ratio=0.3)

        self.kv_cache_stats = {
            'gpu_transfer_time': [],
            'gpu_compute_time': [],
            'cpu_compute_time': [],
            'cpu_to_gpu_transfer_time': [],
            'layer_attention_stats': {i: {'gpu_transfer': 0.0, 'gpu_compute': 0.0,
                                         'cpu_compute': 0.0, 'cpu_to_gpu': 0.0}
                                     for i in range(self.n_layer)}
        }

        self.cpu_attention_backups = {}

        self._linshi_buf = []
        self._expert_stats_buf = []

        self.expert_selection_count = np.zeros((self.n_layer, self.n_expert), dtype=int)
        tick = time.time()
        self.bring_expert_to_gpu()

        self._cpu_attn_ext = None
        try:
            import sys
            _ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpu_attn_ext')
            if _ext_dir not in sys.path:
                sys.path.insert(0, _ext_dir)
            from build import build as _build_cpu_attn
            self._cpu_attn_ext = _build_cpu_attn(verbose=False)

            _n_omp = max(1, os.cpu_count() // 2)
            self._cpu_attn_ext.set_num_threads(_n_omp)
        except Exception as _e:
            pass

        self._cpu_expert_ext = None
        try:
            import sys
            _exp_ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpu_expert_ext')
            if _exp_ext_dir not in sys.path:
                sys.path.insert(0, _exp_ext_dir)
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location('cpu_expert_build', os.path.join(_exp_ext_dir, 'build.py'))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            self._cpu_expert_ext = _mod.build(verbose=False)
            print(f"CPU expert GIL-free ext loaded: {self._cpu_expert_ext}")
        except Exception as _e:
            print(f"CPU expert ext load failed: {_e}, falling back to Python dispatch")

        self._gpu_expert_weights = {}
        for _l in range(self.n_layer):
            for _e in range(self.n_expert):
                if self.expert_loc[_l, _e] == 1:
                    _exp = self.model.layers[_l].mlp.experts[_e]
                    self._gpu_expert_weights[(_l, _e)] = (
                        _exp.gate_proj.weight,
                        _exp.up_proj.weight,
                        _exp.down_proj.weight,
                    )

        self._warmup_kernels()

    def _warmup_kernels(self):
        pass
        import io, contextlib
        self.is_decode = False
        cfg = self.model.config
        _head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        self.inplace_kv = InplaceKVCache(
            self.n_layer, self.batch_size,
            cfg.num_key_value_heads,
            64,
            _head_dim,
            b_cpu=self._get_b_cpu(self.batch_size, cfg),
        )
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 1
        self.expert_selection_stats = []
        self.expert_time_stats = []
        self.perf_stats = {
            'token_embedding': [], 'self_attention': [],
            'attention_kv_cache_transfer': [], 'moe_gating': [],
            'expert_compute': [], 'expert_compute-cpu': []
        }
        dummy_ids = torch.zeros(
            (self.batch_size, 4), dtype=torch.long, device=self.dev
        )
        dummy_pos = torch.arange(4, device=self.dev).unsqueeze(0).expand(self.batch_size, -1)
        dummy_mask = torch.ones(
            (self.batch_size, 1, 4, 4), dtype=torch.bool, device=self.dev
        )
        with torch.no_grad():

            self.qwen_forward(dummy_ids, dummy_pos, dummy_mask)
            self.is_decode = True
            self.past_key_values_length = 4

            dec_ids = torch.zeros((self.batch_size, 1), dtype=torch.long, device=self.dev)
            dec_pos = torch.full((self.batch_size, 1), 4, dtype=torch.long, device=self.dev)
            dec_mask = torch.ones(
                (self.batch_size, 1, 1, 5), dtype=torch.bool, device=self.dev
            )
            self.qwen_forward(dec_ids, dec_pos, dec_mask)
        torch.cuda.synchronize()

        self.is_decode = False
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0
        self.inplace_kv = None
        self._linshi_buf.clear()
        self._expert_stats_buf.clear()

    def bring_non_expert_to_gpu(self):
        pass
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):

            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].mlp.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)

    def get_hot_expert(self):
        pass
        if not hasattr(self, 'is_decode') or not self.is_decode:
            return {}

        hot_experts = {}

        for layer_id in range(self.n_layer):

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

        return hot_experts
    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        pass
        if popular_experts is None:

            hot_experts_file = './hot/qwen.txt'
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

        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def _async_ondemand(self, layer_idx, expert_id, target_placeholder, stream=None):
        expert = self.model.layers[layer_idx].mlp.experts[expert_id]
        if next(expert.parameters()).is_cuda:
            return

        ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(torch.cuda.current_stream())
        with ctx:
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                dst = getattr(target_placeholder, name).weight.data
                src = getattr(expert, name).weight.data
                dst.copy_(src, non_blocking=True)

    def _async_load_expert(self, layer_idx, expert_id):
        expert = self.model.layers[layer_idx].mlp.experts[expert_id]

        if next(expert.parameters()).is_cuda:
            return

        for name in ['gate_proj', 'up_proj', 'down_proj']:
            w = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name)
            src_weight_data_tensor = w.weight.data
            pinned = src_weight_data_tensor.pin_memory()
            w.weight.data = pinned

        target_placeholder = None

        if layer_idx % 2 == 1:
            if not self.expert_placeholder3_inused:
                target_placeholder = self.expert_placeholder3
                self.expert_placeholder3_inused = True
                self.expert_loading_status['expert_placeholder3'] = True
            elif not self.expert_placeholder4_inused:
                target_placeholder = self.expert_placeholder4
                self.expert_placeholder4_inused = True
                self.expert_loading_status['expert_placeholder4'] = True

        else:
            if not self.expert_placeholder_inused:
                target_placeholder = self.expert_placeholder
                self.expert_placeholder_inused = True
                self.expert_loading_status['expert_placeholder'] = True
            elif not self.expert_placeholder2_inused:
                target_placeholder = self.expert_placeholder2
                self.expert_placeholder2_inused = True
                self.expert_loading_status['expert_placeholder2'] = True

        if target_placeholder is None:
            raise RuntimeError("No available expert placeholder")

        if target_placeholder == self.expert_placeholder:
            self.placeholder_to_expert['expert_placeholder'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder2:
            self.placeholder_to_expert['expert_placeholder2'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder3:
            self.placeholder_to_expert['expert_placeholder3'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder4:
            self.placeholder_to_expert['expert_placeholder4'] = (layer_idx, expert_id)

        tick = time.time()
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            dst = getattr(target_placeholder, name).weight.data
            src = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name).weight.data
            dst.copy_(src)

        if target_placeholder == self.expert_placeholder:
            self.expert_loading_status['expert_placeholder'] = False
        elif target_placeholder == self.expert_placeholder2:
            self.expert_loading_status['expert_placeholder2'] = False
        elif target_placeholder == self.expert_placeholder3:
            self.expert_loading_status['expert_placeholder3'] = False
        elif target_placeholder == self.expert_placeholder4:
            self.expert_loading_status['expert_placeholder4'] = False

        copytime = time.time() - tick
        self.expert_to_placeholder[(layer_idx, expert_id)] = target_placeholder
    def is_expert_loading(self, placeholder_name):
        pass
        return self.expert_loading_status.get(placeholder_name, False)
    def is_expert_loaded(self, layer_id, expert_id):
        pass

        return (layer_id not in self.prefetching_list or
                expert_id not in self.prefetching_list[layer_id])
    def release_placeholder(self, layer_idx, expert_id):
        for placeholder_name in ['expert_placeholder', 'expert_placeholder2',
                              'expert_placeholder3', 'expert_placeholder4']:
            stored_expert = self.placeholder_to_expert[placeholder_name]
            if stored_expert and (stored_expert[0] < layer_idx or
                    (stored_expert[0] == self.n_layer - 1 and layer_idx <= 1)):

                setattr(self, f"{placeholder_name}_inused", False)
                self.placeholder_to_expert[placeholder_name] = None

    def get_kv_cache_for_layer(self, layer_idx, device):
        pass
        if self.past_key_value is None:
            return None

        if isinstance(device, str):
            device = torch.device(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1024**2

        from transformers.cache_utils import DynamicCache
        new_cache = DynamicCache()

        n_layers = len(self.past_key_value.key_cache)
        for layer in range(n_layers):
            if layer < len(self.past_key_value.key_cache):
                key = self.past_key_value.key_cache[layer]
                value = self.past_key_value.value_cache[layer]
                if layer == layer_idx:
                    new_cache.update(key.to(device), value.to(device), layer_idx=layer)
                else:
                    new_cache.update(key.cpu(), value.cpu(), layer_idx=layer)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1024**2

        return new_cache

    def save_kv_cache_to_cpu(self, layer_idx, kv_cache):
        pass
        if kv_cache is None:
            return

        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**2

        if hasattr(kv_cache, '_cache'):
            for layer in kv_cache._cache:
                key, value = kv_cache._cache[layer]

                kv_cache._cache[layer] = (key.to('cpu'), value.to('cpu'))

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024**2

        self.past_key_value = kv_cache

    def set_attention_device(self, layer_idx, device):
        pass
        if 0 <= layer_idx < self.n_layer:
            self.attention_device_per_layer[layer_idx] = device
        else:
            raise ValueError(f"Layer index {layer_idx} out of range")

    def set_kv_cache_split_config(self, mode="hybrid", ratio=0.3):
        pass
        if mode not in ["gpu_only", "cpu_only", "hybrid"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of 'gpu_only', 'cpu_only', 'hybrid'")

        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"Invalid ratio: {ratio}. Must be between 0.0 and 1.0")

        self.kv_cache_split_mode = mode
        self.kv_cache_split_ratio = ratio

        if mode == "hybrid":
            pass

    def get_kv_cache_split_stats(self):
        pass
        stats = {
            'mode': self.kv_cache_split_mode,
            'gpu_ratio': self.kv_cache_split_ratio,
            'cpu_ratio': 1 - self.kv_cache_split_ratio
        }
        return stats

    def _record_attention_stats(self, layer_idx, gpu_transfer, gpu_compute,
                                cpu_compute, cpu_to_gpu_transfer):
        pass
        total_time = gpu_transfer + gpu_compute + cpu_compute + cpu_to_gpu_transfer

        self.kv_cache_stats['gpu_transfer_time'].append(gpu_transfer)
        self.kv_cache_stats['gpu_compute_time'].append(gpu_compute)
        self.kv_cache_stats['cpu_compute_time'].append(cpu_compute)
        self.kv_cache_stats['cpu_to_gpu_transfer_time'].append(cpu_to_gpu_transfer)

        self.kv_cache_stats['layer_attention_stats'][layer_idx]['gpu_transfer'] += gpu_transfer
        self.kv_cache_stats['layer_attention_stats'][layer_idx]['gpu_compute'] += gpu_compute
        self.kv_cache_stats['layer_attention_stats'][layer_idx]['cpu_compute'] += cpu_compute
        self.kv_cache_stats['layer_attention_stats'][layer_idx]['cpu_to_gpu'] += cpu_to_gpu_transfer

        if self.is_decode:
            pass

    def get_attention_stats_summary(self):
        pass
        if not self.kv_cache_stats['gpu_compute_time']:
            return {}

        summary = {
            'gpu_transfer_total': sum(self.kv_cache_stats['gpu_transfer_time']),
            'gpu_compute_total': sum(self.kv_cache_stats['gpu_compute_time']),
            'cpu_compute_total': sum(self.kv_cache_stats['cpu_compute_time']),
            'cpu_to_gpu_transfer_total': sum(self.kv_cache_stats['cpu_to_gpu_transfer_time']),
            'layer_stats': {}
        }

        for layer_idx in range(self.n_layer):
            stats = self.kv_cache_stats['layer_attention_stats'][layer_idx]
            summary['layer_stats'][layer_idx] = {
                'gpu_transfer': stats['gpu_transfer'],
                'gpu_compute': stats['gpu_compute'],
                'cpu_compute': stats['cpu_compute'],
                'cpu_to_gpu': stats['cpu_to_gpu'],
                'total': stats['gpu_transfer'] + stats['gpu_compute'] + stats['cpu_compute'] + stats['cpu_to_gpu']
            }

        return summary

    def print_attention_stats_summary(self):
        pass
        summary = self.get_attention_stats_summary()

        if not summary:
            print("_")
            return

        print("\n" + "=" * 60)
        print("_")
        print("=" * 60)

        total_gpu_transfer = summary['gpu_transfer_total']
        total_gpu_compute = summary['gpu_compute_total']
        total_cpu_compute = summary['cpu_compute_total']
        total_cpu_to_gpu = summary['cpu_to_gpu_transfer_total']
        total_time = total_gpu_transfer + total_gpu_compute + total_cpu_compute + total_cpu_to_gpu

        print(f"_")
        print(f"_"
              f"({total_gpu_transfer/total_time*100:.1f}%)")
        print(f"_"
              f"({total_gpu_compute/total_time*100:.1f}%)")
        print(f"_"
              f"({total_cpu_compute/total_time*100:.1f}%)")
        print(f"_"
              f"({total_cpu_to_gpu/total_time*100:.1f}%)")
        print(f"_")

        print(f"_")
        print(f"_")
        print("-" * 60)

        for layer_idx in range(self.n_layer):
            stats = summary['layer_stats'][layer_idx]
            print(f"{layer_idx:4d} | {stats['gpu_transfer']*1000:10.2f} | "
                  f"{stats['gpu_compute']*1000:10.2f} | {stats['cpu_compute']*1000:10.2f} | "
                  f"{stats['cpu_to_gpu']*1000:10.2f} | {stats['total']*1000:10.2f}")

        print("=" * 60)

    def compute_attention_with_split_kv(self, layer, hidden_states, attention_mask,
                                        position_embeddings, layer_idx, device):
        pass
        if self.kv_cache_split_mode == "gpu_only":

            if layer_idx < len(self.past_key_value.key_cache):
                self.past_key_value.key_cache[layer_idx] = self.past_key_value.key_cache[layer_idx].to(device)
                self.past_key_value.value_cache[layer_idx] = self.past_key_value.value_cache[layer_idx].to(device)
            attn_output, attn_weights = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=self.past_key_value,
                use_cache=True,
            )

            if layer_idx < len(self.past_key_value.key_cache):
                self.past_key_value.key_cache[layer_idx] = self.past_key_value.key_cache[layer_idx].cpu()
                self.past_key_value.value_cache[layer_idx] = self.past_key_value.value_cache[layer_idx].cpu()
            return attn_output, attn_weights

        if self.kv_cache_split_mode == "cpu_only":

            attn_module = self.cpu_attention_backups[layer_idx]
            current_past_key_value = self.get_kv_cache_for_layer(layer_idx, 'cpu')
            attn_output, attn_weights = attn_module(
                hidden_states=hidden_states.to('cpu'),
                attention_mask=attention_mask.to('cpu') if attention_mask is not None else None,
                position_embeddings=position_embeddings,
                past_key_values=current_past_key_value,
                use_cache=True,
            )
            attn_output = attn_output.to(device)
            return attn_output, attn_weights

        if (self.kv_cache_split_mode == "hybrid" and self.is_decode and
                self.past_key_value is not None and
                str(device) != 'cpu'):
            return self._compute_pipelined_attention(
                layer, hidden_states, attention_mask, position_embeddings,
                layer_idx, device
            )

        if not self.is_decode and self.kv_cache_split_mode == "hybrid":
            return self._compute_prefill_chunked(
                layer, hidden_states, attention_mask, position_embeddings, layer_idx, device
            )

        if layer_idx < len(self.past_key_value.key_cache):
            self.past_key_value.key_cache[layer_idx] = self.past_key_value.key_cache[layer_idx].to(device)
            self.past_key_value.value_cache[layer_idx] = self.past_key_value.value_cache[layer_idx].to(device)

        attn_output, attn_weights = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=self.past_key_value,
            use_cache=True,
        )

        if layer_idx < len(self.past_key_value.key_cache):
            self.past_key_value.key_cache[layer_idx] = self.past_key_value.key_cache[layer_idx].cpu()
            self.past_key_value.value_cache[layer_idx] = self.past_key_value.value_cache[layer_idx].cpu()
        return attn_output, attn_weights

    def _compute_prefill_chunked(self, layer, hidden_states, attention_mask,
                                  position_embeddings, layer_idx, device):
        pass
        batch_size, seq_len, hidden_dim = hidden_states.shape
        chunk_size = self.prefill_chunk_size

        attn = layer.self_attn
        cfg = self.model.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // num_heads)
        groups = num_heads // num_kv_heads
        scale = head_dim ** -0.5

        cos, sin = position_embeddings

        from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

        k_all = attn.k_norm(attn.k_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim)).transpose(1, 2)
        v_all = attn.v_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        from transformers.models.qwen3_moe.modeling_qwen3_moe import rotate_half
        cos_k = cos.unsqueeze(1)
        sin_k = sin.unsqueeze(1)
        k_all = (k_all * cos_k) + (rotate_half(k_all) * sin_k)

        k_all_cpu = k_all.to('cpu', non_blocking=True); del k_all
        v_all_cpu = v_all.to('cpu', non_blocking=True); del v_all
        torch.cuda.synchronize()

        if hasattr(self, 'inplace_kv'):
            self.inplace_kv.append_4d(layer_idx, k_all_cpu, v_all_cpu)
        else:

            if layer_idx < len(self.past_key_value.key_cache):
                self.past_key_value.key_cache[layer_idx] = k_all_cpu
                self.past_key_value.value_cache[layer_idx] = v_all_cpu
            else:
                self.past_key_value.update(k_all_cpu, v_all_cpu, layer_idx=layer_idx)

        for q_start in range(0, seq_len, chunk_size):
            q_end = min(q_start + chunk_size, seq_len)
            q_cs  = q_end - q_start

            hs_chunk = hidden_states[:, q_start:q_end, :]
            q_chunk = attn.q_norm(attn.q_proj(hs_chunk).view(batch_size, q_cs, num_heads, head_dim)).transpose(1, 2)

            cos_chunk = cos[:, q_start:q_end, :].unsqueeze(1) if cos.shape[1] > 1 else cos.unsqueeze(1)
            sin_chunk = sin[:, q_start:q_end, :].unsqueeze(1) if sin.shape[1] > 1 else sin.unsqueeze(1)
            q_chunk = (q_chunk * cos_chunk) + (rotate_half(q_chunk) * sin_chunk)

            out_acc = torch.zeros(batch_size, num_heads, q_cs, head_dim,
                                  device=device, dtype=hidden_states.dtype)
            max_acc = torch.full((batch_size, num_heads, q_cs, 1),
                                 float('-inf'), device=device, dtype=torch.float32)
            sum_acc = torch.zeros(batch_size, num_heads, q_cs, 1,
                                  device=device, dtype=torch.float32)

            for kv_start in range(0, seq_len, chunk_size):
                kv_end = min(kv_start + chunk_size, seq_len)

                k_chunk = k_all_cpu[:, :, kv_start:kv_end, :].to(device).repeat_interleave(groups, dim=1)
                v_chunk = v_all_cpu[:, :, kv_start:kv_end, :].to(device).repeat_interleave(groups, dim=1)

                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale

                m_chunk = scores.amax(dim=-1, keepdim=True).to(torch.float32)
                e_chunk = torch.exp(scores.to(torch.float32) - m_chunk)
                s_chunk = e_chunk.sum(dim=-1, keepdim=True)
                o_chunk = torch.matmul(e_chunk.to(hidden_states.dtype), v_chunk)

                max_new = torch.max(max_acc, m_chunk)
                w_acc   = torch.nan_to_num(torch.exp(max_acc - max_new), nan=0.0)
                w_chunk = torch.nan_to_num(torch.exp(m_chunk - max_new), nan=0.0)

                sum_acc = w_acc * sum_acc + w_chunk * s_chunk
                out_acc = (out_acc.to(torch.float32) * w_acc +
                           o_chunk.to(torch.float32) * w_chunk).to(hidden_states.dtype)
                max_acc = max_new

                del k_chunk, v_chunk, scores, e_chunk, s_chunk, o_chunk, m_chunk, max_new, w_acc, w_chunk

            sum_acc = sum_acc.clamp(min=1e-6)

            out_norm = (out_acc / sum_acc.to(hidden_states.dtype)).transpose(1, 2).contiguous()
            out_norm = out_norm.view(batch_size, q_cs, num_heads * head_dim)
            del out_acc, max_acc, sum_acc

            hidden_states[:, q_start:q_end, :] = attn.o_proj(out_norm)
            del out_norm, q_chunk

        del k_all_cpu, v_all_cpu

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return hidden_states, None

    def _compute_hybrid_attention(self, layer, hidden_states, attention_mask,
                                  position_embeddings, layer_idx, device):
        pass
        import threading
        batch_size, seq_len, _ = hidden_states.shape

        if (layer_idx < len(self.past_key_value.key_cache) and
                self.past_key_value.key_cache[layer_idx] is not None):
            past_seq_len = self.past_key_value.key_cache[layer_idx].shape[2]
        else:
            past_seq_len = 0

        total_seq_len = past_seq_len + seq_len
        gpu_seq_len = max(1, int(total_seq_len * self.kv_cache_split_ratio))
        cpu_seq_len = total_seq_len - gpu_seq_len

        gpu_transfer_time = 0.0
        gpu_compute_time = 0.0
        cpu_compute_time = 0.0
        cpu_to_gpu_transfer_time = 0.0

        if cpu_seq_len <= 0:
            transfer_start = time.time()
            current_past_key_value = self.get_kv_cache_for_layer(layer_idx, device)
            gpu_transfer_time = time.time() - transfer_start
            compute_start = time.time()
            attn_output, attn_weights = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=current_past_key_value,
                use_cache=True,
            )
            gpu_compute_time = time.time() - compute_start
            self._record_attention_stats(layer_idx, gpu_transfer_time, gpu_compute_time,
                                         cpu_compute_time, cpu_to_gpu_transfer_time)
            return attn_output, attn_weights

        attn = layer.self_attn
        cfg = self.model.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // num_heads)
        groups = num_heads // num_kv_heads

        q = attn.q_norm(attn.q_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim)).transpose(1, 2)
        k_new = attn.k_norm(attn.k_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim)).transpose(1, 2)
        v_new = attn.v_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb
        q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        q_cpu = q.to('cpu', non_blocking=True)
        k_new_cpu = k_new.to('cpu', non_blocking=True)
        v_new_cpu = v_new.to('cpu', non_blocking=True)

        torch.cuda.synchronize()
        k_new_f32 = k_new_cpu.float()
        v_new_f32 = v_new_cpu.float()

        past_key   = self.past_key_value.key_cache[layer_idx]   if layer_idx < len(self.past_key_value.key_cache) else None
        past_value = self.past_key_value.value_cache[layer_idx] if layer_idx < len(self.past_key_value.key_cache) else None
        if past_key is not None:
            full_key_cpu   = torch.cat([past_key.float(),   k_new_f32], dim=2)
            full_value_cpu = torch.cat([past_value.float(), v_new_f32], dim=2)
        else:
            full_key_cpu, full_value_cpu = k_new_f32, v_new_f32
        if layer_idx < len(self.past_key_value.key_cache):
            self.past_key_value.key_cache[layer_idx]   = full_key_cpu
            self.past_key_value.value_cache[layer_idx] = full_value_cpu
        else:
            self.past_key_value.update(full_key_cpu, full_value_cpu, layer_idx=layer_idx)

        key_cpu_part = full_key_cpu[:, :, :cpu_seq_len, :]
        val_cpu_part = full_value_cpu[:, :, :cpu_seq_len, :]
        key_gpu_part = full_key_cpu[:, :, cpu_seq_len:, :].to(device)
        val_gpu_part = full_value_cpu[:, :, cpu_seq_len:, :].to(device)

        scale = head_dim ** -0.5

        cpu_result = {}
        gpu_result = {}

        @torch.no_grad()
        def _cpu_compute():
            t0 = time.time()
            cpu_seq = key_cpu_part.shape[2]
            B_kv = batch_size * num_kv_heads
            q_g = q_cpu.float().view(B_kv, groups, head_dim).contiguous()

            if self._cpu_attn_ext is not None:

                if not getattr(self, '_cpu_attn_omp_set', False):
                    _n_omp = max(1, os.cpu_count() // 2)
                    self._cpu_attn_ext.set_num_threads(_n_omp)
                    self._cpu_attn_omp_set = True

                q_g_bf16 = q_cpu.bfloat16().view(B_kv, groups, head_dim).contiguous()
                k_f = key_cpu_part.reshape(B_kv, cpu_seq, head_dim).bfloat16()
                v_f = val_cpu_part.reshape(B_kv, cpu_seq, head_dim).bfloat16()
                out_acc, max_acc, sum_acc = self._cpu_attn_ext.fused_cpu_attn(q_g_bf16, k_f, v_f, scale)
                out_acc = out_acc * sum_acc
            else:

                k_f = key_cpu_part.reshape(B_kv, cpu_seq, head_dim).contiguous()
                v_f = val_cpu_part.reshape(B_kv, cpu_seq, head_dim).contiguous()
                out_acc = torch.zeros(B_kv, groups, head_dim)
                max_acc = torch.full((B_kv, groups, 1), float('-inf'))
                sum_acc = torch.zeros(B_kv, groups, 1)
                block_size = 1024
                for blk_start in range(0, cpu_seq, block_size):
                    blk_end = min(blk_start + block_size, cpu_seq)
                    k_blk = k_f[:, blk_start:blk_end, :]
                    v_blk = v_f[:, blk_start:blk_end, :]
                    scores = torch.bmm(q_g, k_blk.transpose(-2, -1)) * scale
                    m_blk = scores.amax(dim=-1, keepdim=True)
                    e_blk = torch.exp(scores - m_blk)
                    s_blk = e_blk.sum(dim=-1, keepdim=True)
                    o_blk = torch.bmm(e_blk, v_blk)
                    m_new = torch.max(max_acc, m_blk)
                    w_old = torch.exp(max_acc - m_new)
                    w_new = torch.exp(m_blk - m_new)
                    sum_acc = w_old * sum_acc + w_new * s_blk
                    out_acc = w_old * out_acc + w_new * o_blk
                    max_acc = m_new

            cpu_result['out'] = out_acc.view(batch_size, num_heads, seq_len, head_dim)
            cpu_result['max'] = max_acc.view(batch_size, num_heads, seq_len, 1)
            cpu_result['sum'] = sum_acc.view(batch_size, num_heads, seq_len, 1)
            cpu_result['time'] = time.time() - t0

        gpu_t0 = time.time()
        gpu_seq = key_gpu_part.shape[2]
        B_kv = batch_size * num_kv_heads
        q_g_gpu = q.float().view(B_kv, groups, head_dim)
        k_f_gpu = key_gpu_part.reshape(B_kv, gpu_seq, head_dim)
        v_f_gpu = val_gpu_part.reshape(B_kv, gpu_seq, head_dim)
        scores_gpu = torch.bmm(q_g_gpu, k_f_gpu.transpose(-2, -1)) * scale
        m_gpu = scores_gpu.amax(dim=-1, keepdim=True)
        e_gpu = torch.exp(scores_gpu - m_gpu)
        s_gpu = e_gpu.sum(dim=-1, keepdim=True)
        o_gpu = torch.bmm(e_gpu, v_f_gpu)

        _cpu_compute()

        torch.cuda.synchronize()
        gpu_compute_time = time.time() - gpu_t0 - cpu_result['time']
        gpu_result['out'] = o_gpu.view(batch_size, num_heads, seq_len, head_dim)
        gpu_result['max'] = m_gpu.view(batch_size, num_heads, seq_len, 1)
        gpu_result['sum'] = s_gpu.view(batch_size, num_heads, seq_len, 1)

        cpu_compute_time = cpu_result['time']

        transfer_start = time.time()
        out_cpu_d = cpu_result['out'].to(device)
        max_cpu_d = cpu_result['max'].to(device)
        sum_cpu_d = cpu_result['sum'].to(device)
        cpu_to_gpu_transfer_time = time.time() - transfer_start

        max_gpu = gpu_result['max'].float()
        sum_gpu = gpu_result['sum'].float()
        out_gpu = gpu_result['out'].float()

        max_global = torch.max(max_gpu, max_cpu_d)
        w_gpu = torch.exp(max_gpu - max_global)
        w_cpu = torch.exp(max_cpu_d - max_global)
        global_denom = w_gpu * sum_gpu + w_cpu * sum_cpu_d
        attn_out = (out_gpu * w_gpu + out_cpu_d * w_cpu) / global_denom

        attn_out = attn_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
        attn_output = attn.o_proj(attn_out)

        self._record_attention_stats(layer_idx, gpu_transfer_time, gpu_compute_time,
                                     cpu_compute_time, cpu_to_gpu_transfer_time)

        if getattr(self, '_split_ratio_adapt', False) and gpu_compute_time > 0 and cpu_compute_time > 0:
            total = gpu_compute_time + cpu_compute_time

            target_ratio = self.kv_cache_split_ratio * (total / (2.0 * gpu_compute_time))
            target_ratio = max(0.1, min(0.9, target_ratio))
            self.kv_cache_split_ratio = 0.85 * self.kv_cache_split_ratio + 0.15 * target_ratio

        return attn_output, None

    @staticmethod
    def _get_b_cpu(batch_size, cfg):
        pass
        bkv = batch_size * cfg.num_key_value_heads
        cpu_group_ratio = 0.5625

        if batch_size < 16:
            return None
        return max(1, min(bkv - 1, int(bkv * cpu_group_ratio)))

    def _build_attn_graph(self, gs, bkv, groups, head_dim, scale, device):
        pass
        q_s   = torch.zeros(bkv, groups, head_dim, dtype=torch.bfloat16, device=device)
        k_s   = torch.zeros(bkv, gs,     head_dim, dtype=torch.bfloat16, device=device)
        v_s   = torch.zeros(bkv, gs,     head_dim, dtype=torch.bfloat16, device=device)
        out_s = torch.zeros(bkv, groups, head_dim, dtype=torch.bfloat16, device=device)
        max_s = torch.full((bkv, groups, 1), float('-inf'), dtype=torch.bfloat16, device=device)
        sum_s = torch.zeros(bkv, groups, 1, dtype=torch.bfloat16, device=device)

        for _ in range(3):
            sc = torch.bmm(q_s, k_s.transpose(-2, -1)) * scale
            mc = sc.amax(-1, keepdim=True)
            ec = torch.exp(sc - mc)
            ssc = ec.sum(-1, keepdim=True)
            oc = torch.bmm(ec, v_s)
            mn = torch.max(max_s, mc)
            wa = torch.exp(max_s - mn); wc = torch.exp(mc - mn)
            sum_s.mul_(wa).add_(wc * ssc)
            out_s.mul_(wa).add_(wc * oc)
            max_s.copy_(mn)
        torch.cuda.synchronize(device)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sc = torch.bmm(q_s, k_s.transpose(-2, -1)) * scale
            mc = sc.amax(-1, keepdim=True)
            ec = torch.exp(sc - mc)
            ssc = ec.sum(-1, keepdim=True)
            oc = torch.bmm(ec, v_s)
            mn = torch.max(max_s, mc)
            wa = torch.exp(max_s - mn); wc = torch.exp(mc - mn)
            sum_s.mul_(wa).add_(wc * ssc)
            out_s.mul_(wa).add_(wc * oc)
            max_s.copy_(mn)

        return g, {'q': q_s, 'k': k_s, 'v': v_s, 'out': out_s, 'max': max_s, 'sum': sum_s, 'graph': g}

    @torch.no_grad()
    def _compute_pipelined_attention(self, layer, hidden_states, attention_mask,
                                     position_embeddings, layer_idx, device):
        pass
        batch_size, seq_len, _ = hidden_states.shape

        attn = layer.self_attn
        cfg  = self.model.config
        num_heads    = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim     = getattr(cfg, 'head_dim', cfg.hidden_size // num_heads)
        groups       = num_heads // num_kv_heads
        scale        = head_dim ** -0.5

        q     = attn.q_norm(attn.q_proj(hidden_states).view(batch_size, seq_len, num_heads,    head_dim)).transpose(1, 2)
        k_new = attn.k_norm(attn.k_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim)).transpose(1, 2)
        v_new = attn.v_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb
        q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        k_new_cpu = k_new.to('cpu', non_blocking=True)
        v_new_cpu = v_new.to('cpu', non_blocking=True)

        if hasattr(self, 'inplace_kv'):

            self.inplace_kv.append_4d(layer_idx, k_new_cpu, v_new_cpu)
            kv_ret = self.inplace_kv.get(layer_idx)

            if isinstance(kv_ret[0], tuple):
                (full_k_cpu, full_k_gpu), (full_v_cpu, full_v_gpu) = kv_ret
                full_k = full_k_cpu
                full_v = full_v_cpu
                _kv_split = True
            else:
                full_k, full_v = kv_ret
                _kv_split = False
        else:

            k_new_bkv = k_new_cpu.reshape(batch_size * num_kv_heads, seq_len, head_dim).contiguous()
            v_new_bkv = v_new_cpu.reshape(batch_size * num_kv_heads, seq_len, head_dim).contiguous()
            if (layer_idx < len(self.past_key_value.key_cache) and
                    self.past_key_value.key_cache[layer_idx] is not None):
                past_key   = self.past_key_value.key_cache[layer_idx]
                past_value = self.past_key_value.value_cache[layer_idx]
                if past_key.ndim == 4:
                    past_seq = past_key.shape[2]
                    past_k_bkv = past_key.reshape(batch_size * num_kv_heads, past_seq, head_dim).contiguous()
                    past_v_bkv = past_value.reshape(batch_size * num_kv_heads, past_seq, head_dim).contiguous()
                else:
                    past_k_bkv = past_key
                    past_v_bkv = past_value
                full_k = torch.cat([past_k_bkv, k_new_bkv], dim=1)
                full_v = torch.cat([past_v_bkv, v_new_bkv], dim=1)
            else:
                full_k = k_new_bkv
                full_v = v_new_bkv
            if layer_idx < len(self.past_key_value.key_cache):
                self.past_key_value.key_cache[layer_idx]   = full_k
                self.past_key_value.value_cache[layer_idx] = full_v
            else:
                self.past_key_value.update(full_k, full_v, layer_idx=layer_idx)

        total_seq_len = full_k.shape[1]

        if self.decode_group_size is not None:
            gs = self.decode_group_size
        else:
            target = (total_seq_len + 1) // 2
            gs = ((target + 63) // 64) * 64
            gs = max(128, min(2048, gs))

        bkv = batch_size * num_kv_heads
        cpu_group_ratio = getattr(self, 'cpu_group_ratio', 0.5625)

        if batch_size >= 32:
            kv_threshold = 2048
            if total_seq_len <= 2048:
                _ratio = 0.25
            elif total_seq_len <= 4096:
                _ratio = 0.375
            else:
                _ratio = cpu_group_ratio
        elif batch_size >= 16:
            kv_threshold = 4096
            if total_seq_len <= 4096:
                _ratio = 0.156
            elif total_seq_len <= 8192:
                _ratio = 0.3125
            elif total_seq_len <= 16384:
                _ratio = 0.4375
            else:
                _ratio = cpu_group_ratio
        elif batch_size >= 8:
            kv_threshold = 8192
            if total_seq_len <= 8192:
                _ratio = 0.156
            elif total_seq_len <= 16384:
                _ratio = 0.3125
            else:
                _ratio = cpu_group_ratio
        elif batch_size >= 4:
            kv_threshold = 16384
            _ratio = 0.0625 if total_seq_len <= 16384 else cpu_group_ratio
        else:
            kv_threshold = 32768
            _ratio = cpu_group_ratio
        if total_seq_len >= kv_threshold:
            b_cpu = max(1, min(bkv - 1, round(bkv * _ratio)))
        else:
            b_cpu = 0
        b_gpu = bkv - b_cpu

        if locals().get('_kv_split', False):

            _k_cpu_part = full_k_cpu
            _v_cpu_part = full_v_cpu
            _k_gpu_part = full_k_gpu
            _v_gpu_part = full_v_gpu
        else:
            _k_cpu_part = full_k[:b_cpu]
            _v_cpu_part = full_v[:b_cpu]
            _k_gpu_part = full_k[b_cpu:]
            _v_gpu_part = full_v[b_cpu:]

        if b_cpu == 0:

            if locals().get('_kv_split', False):
                full_k = torch.cat([full_k_cpu, full_k_gpu], dim=0)
                full_v = torch.cat([full_v_cpu, full_v_gpu], dim=0)

            k_gpu = full_k.to(device).view(batch_size, num_kv_heads, total_seq_len, head_dim).permute(0, 2, 1, 3).contiguous()
            v_gpu = full_v.to(device).view(batch_size, num_kv_heads, total_seq_len, head_dim).permute(0, 2, 1, 3).contiguous()
            if _FLASH_AVAILABLE:
                q_fa = q.bfloat16().transpose(1, 2)
                attn_out = _flash_attn_func(q_fa, k_gpu, v_gpu, causal=False)
                attn_out = attn_out.reshape(batch_size, seq_len, num_heads * head_dim)
            else:

                q_sdpa = q.bfloat16()
                k_sdpa = k_gpu.transpose(1, 2).unsqueeze(2).expand(
                    batch_size, num_kv_heads, groups, total_seq_len, head_dim
                ).reshape(batch_size, num_heads, total_seq_len, head_dim)
                v_sdpa = v_gpu.transpose(1, 2).unsqueeze(2).expand(
                    batch_size, num_kv_heads, groups, total_seq_len, head_dim
                ).reshape(batch_size, num_heads, total_seq_len, head_dim)
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, scale=head_dim ** -0.5)
                attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
            return attn.o_proj(attn_out.to(hidden_states.dtype)), None

        q_g = q.bfloat16().view(bkv, groups, head_dim)

        _gpu_starts = list(range(0, total_seq_len, gs))
        n_gpu_groups = len(_gpu_starts)

        _use_sdpa_single = (total_seq_len <= 16384 and b_gpu > 0)

        if not hasattr(self, '_attn_graphs'):
            self._attn_graphs = {}

        graph_key = (gs, b_gpu)

        if not _use_sdpa_single and graph_key not in self._attn_graphs and b_gpu > 0:
            _, self._attn_graphs[graph_key] = self._build_attn_graph(
                gs, b_gpu, groups, head_dim, scale, device)

        if not _use_sdpa_single and graph_key in self._attn_graphs:
            st = self._attn_graphs[graph_key]
            out_gpu = st['out']; max_gpu = st['max']; sum_gpu = st['sum']
            out_gpu.zero_()
            max_gpu.fill_(float('-inf'))
            sum_gpu.zero_()
            st['q'].copy_(q_g[b_cpu:])
        else:
            out_gpu = torch.zeros(b_gpu, groups, head_dim, device=device, dtype=torch.bfloat16)
            max_gpu = torch.full((b_gpu, groups, 1), float('-inf'), device=device, dtype=torch.bfloat16)
            sum_gpu = torch.zeros(b_gpu, groups, 1, device=device, dtype=torch.bfloat16)

        import threading
        cpu_result = {}

        def _cpu_pipeline():

            if self._numa1_cpus:
                try:
                    os.sched_setaffinity(0, self._numa1_cpus)
                except Exception:
                    pass
            t0 = time.time()

            q_cpu = q_g[:b_cpu].cpu()
            k_cpu = _k_cpu_part
            v_cpu = _v_cpu_part
            if self._cpu_attn_ext is not None:

                out_c, max_c, sum_c = self._cpu_attn_ext.fused_cpu_attn(q_cpu, k_cpu, v_cpu, scale)

            else:
                out_c = torch.zeros(b_cpu, groups, head_dim, dtype=torch.bfloat16)
                max_c = torch.full((b_cpu, groups, 1), float('-inf'), dtype=torch.bfloat16)
                sum_c = torch.zeros(b_cpu, groups, 1, dtype=torch.bfloat16)
                for cs in range(0, total_seq_len, gs):
                    ce    = min(cs + gs, total_seq_len)
                    k_f   = k_cpu[:, cs:ce, :].contiguous()
                    v_f   = v_cpu[:, cs:ce, :].contiguous()
                    scores  = torch.bmm(q_cpu, k_f.transpose(-2, -1)) * scale
                    m_chunk = scores.amax(dim=-1, keepdim=True)
                    e_chunk = torch.exp(scores - m_chunk)
                    s_chunk = e_chunk.sum(dim=-1, keepdim=True)
                    o_chunk = torch.bmm(e_chunk, v_f)
                    max_new = torch.max(max_c, m_chunk)
                    w_c     = torch.exp(max_c - max_new)
                    w_k     = torch.exp(m_chunk - max_new)
                    sum_c   = w_c * sum_c + w_k * s_chunk
                    out_c   = out_c * w_c + o_chunk * w_k
                    max_c   = max_new
                out_c = out_c / sum_c.clamp(min=1e-6)
            cpu_result.update({'out': out_c, 'time': time.time() - t0})

        if b_cpu > 0:
            cpu_thread = threading.Thread(target=_cpu_pipeline)
            cpu_thread.start()

        k_gpu_part = _k_gpu_part
        v_gpu_part = _v_gpu_part
        q_g_gpu    = q_g[b_cpu:]
        gpu_starts = _gpu_starts

        gpu_loop_start = time.time()
        use_graph = graph_key in self._attn_graphs

        if _use_sdpa_single:

            k_f = k_gpu_part.to(device)
            v_f = v_gpu_part.to(device)

            q_sdpa = q_g_gpu.unsqueeze(2)
            k_sdpa = k_f.unsqueeze(1).expand(b_gpu, groups, total_seq_len, head_dim)
            v_sdpa = v_f.unsqueeze(1).expand(b_gpu, groups, total_seq_len, head_dim)
            out_sdpa = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, scale=scale)
            out_gpu = out_sdpa.squeeze(2)

            sum_gpu.fill_(1.0)
            del k_f, v_f, q_sdpa, k_sdpa, v_sdpa, out_sdpa
        else:

            if self._decode_kv_buffers is None:
                self._decode_kv_buffers = {}
            buf_key = (gs, b_gpu)
            if buf_key not in self._decode_kv_buffers:

                buf_shape = (b_gpu, gs, head_dim)
                self._decode_kv_buffers[buf_key] = {
                    'k': [torch.empty(buf_shape, dtype=torch.bfloat16, device=device),
                          torch.empty(buf_shape, dtype=torch.bfloat16, device=device)],
                    'v': [torch.empty(buf_shape, dtype=torch.bfloat16, device=device),
                          torch.empty(buf_shape, dtype=torch.bfloat16, device=device)]
                }

            buf_k = self._decode_kv_buffers[buf_key]['k']
            buf_v = self._decode_kv_buffers[buf_key]['v']
            stream_xfer = torch.cuda.Stream(device=device)

            s0_end = min(gs, total_seq_len)
            with torch.cuda.stream(stream_xfer):

                buf_k[0][:, :s0_end, :].copy_(k_gpu_part[:, 0:s0_end, :], non_blocking=True)
                buf_v[0][:, :s0_end, :].copy_(v_gpu_part[:, 0:s0_end, :], non_blocking=True)

            for i, kv_start in enumerate(gpu_starts):
                kv_end    = min(kv_start + gs, total_seq_len)
                gs_actual = kv_end - kv_start
                cur = i % 2

                torch.cuda.current_stream(device).wait_stream(stream_xfer)

                if i + 1 < n_gpu_groups:
                    nxt = 1 - cur
                    ns  = gpu_starts[i + 1]
                    ne  = min(ns + gs, total_seq_len)
                    with torch.cuda.stream(stream_xfer):

                        buf_k[nxt][:, :ne-ns, :].copy_(k_gpu_part[:, ns:ne, :], non_blocking=True)
                        buf_v[nxt][:, :ne-ns, :].copy_(v_gpu_part[:, ns:ne, :], non_blocking=True)

                k_f = buf_k[cur][:, :gs_actual, :].contiguous()
                v_f = buf_v[cur][:, :gs_actual, :].contiguous()

                if use_graph and gs_actual == gs:
                    st = self._attn_graphs[graph_key]
                    st['k'].copy_(k_f)
                    st['v'].copy_(v_f)
                    st['graph'].replay()
                else:
                    scores  = torch.bmm(q_g_gpu, k_f.transpose(-2, -1)) * scale
                    m_chunk = scores.amax(dim=-1, keepdim=True)
                    e_chunk = torch.exp(scores - m_chunk)
                    s_chunk = e_chunk.sum(dim=-1, keepdim=True)
                    o_chunk = torch.bmm(e_chunk, v_f)
                    max_new = torch.max(max_gpu, m_chunk)
                    w_acc   = torch.exp(max_gpu - max_new)
                    w_chunk = torch.exp(m_chunk - max_new)
                    out_gpu.mul_(w_acc).add_(o_chunk * w_chunk)
                    sum_gpu.mul_(w_acc).add_(w_chunk * s_chunk)
                    max_gpu.copy_(max_new)
                    del scores, m_chunk, e_chunk, s_chunk, o_chunk, max_new, w_acc, w_chunk

        torch.cuda.synchronize(device)
        gpu_pipeline_time = time.time() - gpu_loop_start

        if b_cpu > 0:
            cpu_thread.join()

            out_gpu_norm = out_gpu / sum_gpu.clamp(min=1e-3)

            out_cpu_d = cpu_result['out'].to(device)

            out_acc = torch.cat([out_cpu_d, out_gpu_norm], dim=0)
        else:

            out_acc = out_gpu / sum_gpu.clamp(min=1e-3)

        attn_out = out_acc.view(batch_size, num_heads, seq_len, head_dim)
        attn_out = attn_out.to(hidden_states.dtype).transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, num_heads * head_dim)
        attn_output = attn.o_proj(attn_out)

        return attn_output, None

    def _compute_attention_chunk(self, layer, hidden_states, attention_mask,
                                 position_embeddings, key, value, seq_len, total_seq_len, device,layer_idx):
        pass
        from transformers.cache_utils import DynamicCache

        if str(device) == 'cpu':

            attn_module = self.cpu_attention_backups[layer_idx]
        else:

            attn_module = layer.self_attn

        temp_cache = DynamicCache()

        if key is not None and value is not None:
            temp_cache.update(key, value, layer_idx=layer.layer_idx)

        result = attn_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=temp_cache,
            use_cache=False,
            output_attentions=True
        )

        if isinstance(result, tuple) and len(result) >= 2:
            attn_output = result[0]
            attn_weights = result[1]
        else:
            attn_output = result
            attn_weights = None

        return attn_output, attn_weights

    def split_kv_cache(self, key, value):
        pass
        seq_len = key.shape[2]
        split_point = int(seq_len * (1 - self.kv_cache_split_ratio))
        key_cpu = key[:, :, :split_point, :].to('cpu')
        value_cpu = value[:, :, :split_point, :].to('cpu')
        key_gpu = key[:, :, split_point:, :]
        value_gpu = value[:, :, split_point:, :]
        return (key_gpu, value_gpu), (key_cpu, value_cpu)

    def compute_local_attention(self, query, key, value, attention_mask, device):
        pass
        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.shape[-1] ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask.to(device)
        max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(scores - max_scores)
        lse = max_scores.squeeze(-1) + torch.log(torch.sum(exp_scores, dim=-1))
        unnorm_output = torch.matmul(exp_scores, value)
        return unnorm_output, lse, exp_scores.sum(dim=-1)

    def merge_attention_outputs(self, gpu_output, gpu_lse, cpu_output, cpu_lse):
        pass
        gpu_output = gpu_output.to(self.dev)
        gpu_lse = gpu_lse.to(self.dev)
        cpu_output = cpu_output.to(self.dev)
        cpu_lse = cpu_lse.to(self.dev)
        max_lse = torch.max(gpu_lse, cpu_lse)
        global_lse = max_lse + torch.log(torch.exp(gpu_lse - max_lse) + torch.exp(cpu_lse - max_lse))
        gpu_weight = torch.exp(gpu_lse - global_lse).unsqueeze(-1)
        cpu_weight = torch.exp(cpu_lse - global_lse).unsqueeze(-1)
        final_output = gpu_output * gpu_weight + cpu_output * cpu_weight
        return final_output

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

        return bool(self.expert_loc[i_layer, i_expert] == 1)

    def calc_n_expert_on_gpu(self):
        pass
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].mlp.experts[0].parameters()
        )
        expert_size_bytes = n_param * 2

        max_gpu_bytes = 24 * 1024 ** 3

        current_allocated = torch.cuda.memory_allocated()

        reserved_bytes = 10 * 1024 ** 3

        available_bytes = max_gpu_bytes - current_allocated - reserved_bytes
        n_from_mem = int(available_bytes / expert_size_bytes)

        n_total = self.n_layer * self.n_expert
        return max(16, min(n_from_mem, n_total))

    def initial_beam_tensor(self, input_tensor):

        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor
    def _process_single_cpu_expert(self, i_layer, i_expert, combined_inps, combined_weights, combined_selected, expert_mask, combined_mask):
        expert_input = combined_inps[expert_mask]

        tick = time.time()
        expert_output = self.run_expert_at_cpu(i_layer, i_expert, expert_input)
        self.perf_stats['expert_compute-cpu'].append(time.time() - tick)

        expert_weights = combined_weights[expert_mask].gather(
            1, (combined_selected[expert_mask] == i_expert).long().argmax(dim=1, keepdim=True)
        )
        expert_output = expert_output * expert_weights

        mask_index = combined_mask.nonzero().squeeze(1)[expert_mask.nonzero().squeeze(1)]
        return mask_index, expert_output

    def generate_heatmap(self):
        pass
        import numpy as np
        import os

        os.makedirs('./log', exist_ok=True)

        np.savetxt('./log/expert_selection_count.csv',
                  self.expert_selection_count,
                  delimiter=',',
                  fmt='%d')

        np.savetxt('./log/expert_high_weight_count.csv',
                  self.expert_high_weight_count,
                  delimiter=',',
                  fmt='%d')

        print("_")
        return self.expert_selection_count

    def generate(self, text=None, output_token=20, input_token=None):
        torch.set_num_threads(16)
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0

        _max_seq = (input_token or 2048) + output_token + 16
        cfg = self.model.config
        self.inplace_kv = InplaceKVCache(
            self.n_layer, self.batch_size,
            cfg.num_key_value_heads,
            _max_seq,
            getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads),
            b_cpu=self._get_b_cpu(self.batch_size, cfg),
        )

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

            attention_mask = attention_mask[:, :, :, :input_token]

        tick = time.time()
        self.is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)
        self.token_decode_times = []
        self._cost_log_lock = threading.Lock()
        self._cost_log_file = open("cost.txt", "w", buffering=1)
        self.perf_stats = {
            'token_embedding': [],
            'self_attention': [],
            'attention_kv_cache_transfer': [],
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

                new_mask = torch.ones(
                    (attention_mask.shape[0], attention_mask.shape[1], 1, 1),
                    dtype=torch.bool,
                    device=self.dev
                )

                attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

            new_position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
            logits = self.qwen_forward(input_ids, new_position_ids, attention_mask )

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

                self.past_key_value = transformers.cache_utils.DynamicCache()
            self.is_decode = True
        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        if self.is_decode:
            self.print_attention_stats_summary()

        if self._linshi_buf or self._expert_stats_buf:
            os.makedirs('./log', exist_ok=True)
            if self._linshi_buf:
                with open('./log/linshi.txt', 'a') as f:
                    f.write(''.join(self._linshi_buf))
                self._linshi_buf.clear()
            if self._expert_stats_buf:
                with open('./log/expert_stats.txt', 'a') as f:
                    f.write(''.join(self._expert_stats_buf))
                self._expert_stats_buf.clear()

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
                'layer_time_details': self.layer_time_details,
                'expert_hot_stats': self.get_expert_stats(),
                'layer_time_avg': {
                    i: self.layer_time_accumulator[i] / max(1, len([x for x in self.layer_time_stats if x['layer_id'] == i]))
                    for i in range(self.n_layer)
                },
                'layer_time_avg_details': {
                    case: {
                        i: self.layer_time_accumulator_details[case][i] / max(1, len([x for x in self.layer_time_details[case] if x['layer_id'] == i]))
                        for i in range(self.n_layer)
                    }
                    for case in ['all_gpu', 'all_cpu', 'mixed']
                }
            }
        )
        self._cost_log_file.close()

    def tokenize(self, text, input_token):
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text should be str or list of str")

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

            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

            attention_mask = attention_mask.expand(-1, 32, -1, -1)

        return input_ids, position_ids, attention_mask

    @torch.no_grad()
    def qwen_forward(self, input_ids, position_ids, attention_mask ):
        hidden_dim = self.model.config.hidden_size
        inps = self.model.embed_tokens(input_ids)

        if self.is_decode:
            total_decode_start = time.time()
            layer_times = {i: 0.0 for i in range(self.n_layer)}
            layer_times_fwd = {i: 0.0 for i in range(self.n_layer)}
            layer_times_mid = {i: 0.0 for i in range(self.n_layer)}
            layer_times_final = {i: 0.0 for i in range(self.n_layer)}
            if not hasattr(self, 'layer_breakdown'):
                self.layer_breakdown = {k: 0.0 for k in ['attn', 'routing', 'expert_threads', 'other']}

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        layer_start_time = time.time()
        layer_total_time = 0.0
        isprefetch=False

        for i_layer, layer in enumerate(self.model.layers):
            layer_tick = time.time()

            self.release_placeholder(i_layer, 0)

            laymid = time.time() - layer_tick

            original_inps_shape = inps.shape
            inps_residual = inps

            inps = _rms_norm_bf16(inps, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)

            inps = inps.view(batch_size, seq_len, hidden_dim)
            tick = time.time()

            if not self.is_decode:

                attention_device = "cuda:0"
            else:

                attention_device = "cuda:0"
            if attention_device == "cuda:0":
                transfer_start = time.time()
                self.model.layers[i_layer].self_attn.to(self.dev)
            else:
                transfer_start = time.time()
                self.model.layers[i_layer].self_attn.to('cpu')

            if attention_device == "cpu":

                current_inps = inps.to('cpu')
                current_attention_mask = attention_mask.to('cpu')
                current_position_ids = position_ids.to('cpu')
                current_device = torch.device('cpu')
            else:

                current_inps = inps
                current_attention_mask = attention_mask
                current_position_ids = position_ids
                current_device = self.dev

            position_embeddings = self.model.rotary_emb(current_inps, current_position_ids)

            tick = time.time()
            attn_output, attn_weights = self.compute_attention_with_split_kv(
                layer=layer,
                hidden_states=current_inps,
                attention_mask=current_attention_mask,
                position_embeddings=position_embeddings,
                layer_idx=i_layer,
                device=current_device
            )

            torch.cuda.synchronize()
            attention_compute_time = time.time() - tick
            if self.is_decode:
                self.layer_breakdown['attn'] += attention_compute_time
            del position_embeddings

            if hasattr(attn_output, 'past_key_value') and attn_output.past_key_value is not None:
                self.past_key_value = attn_output.past_key_value

            if attention_device == "cpu":
                inps = attn_output.to(self.dev)
            else:
                inps = attn_output

            inps_residual.add_(inps)
            inps = inps_residual

            inps = _rms_norm_bf16(inps, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon)
            inps = inps.view(-1, hidden_dim)

            layer_idx=i_layer
            if layer_idx not in self.layer_data:
                self.layer_data[layer_idx] = {
                    "hidden_states": [],
                    "expert_indices": []
                }

            pre_expert_hidden_states = inps.view(batch_size, seq_len, -1)

            _routing_tick = time.time() if self.is_decode else 0
            router_logits = layer.mlp.gate(inps)
            del pre_expert_hidden_states

            routing_weights = F.softmax(router_logits, dim=1)

            routing_weights, selected_experts = torch.topk(routing_weights, 8, dim=-1)

            _selected_2d = selected_experts.tolist()
            _selected_flat = [e for row in _selected_2d for e in row]

            if self.is_decode:
                expert_pos_map = {}
                for _row_i, _row in enumerate(_selected_2d):
                    for _col_i, _eid in enumerate(_row):
                        if _eid not in expert_pos_map:
                            expert_pos_map[_eid] = (_row_i, _col_i)

            expert_token_counts = {}
            for _row in _selected_2d:
                for _eid in set(_row):
                    expert_token_counts[_eid] = expert_token_counts.get(_eid, 0) + 1

            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.mlp.experts

            if self.cpu_offload == 0:

                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=self.n_expert
                ).permute(2, 1, 0)

                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:

                        continue

                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if not is_cuda:
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )

                        current_state = self.expert_placeholder(current_state)
                    else:

                        current_state = current_state * routing_weights[top_2_list, idx_list, None]
                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )

                    if not is_cuda:
                        experts[i_expert] = experts[i_expert].to("cpu")

            else:

                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=self.n_expert
                ).permute(2, 1, 0)

                gpu_experts = []
                experts_remaining = []
                selected_expert_ids = selected_experts.unique().tolist()

                import collections as _collections
                _remaining_lock = threading.Lock()
                _scatter_lock   = threading.Lock()

                gpu_experts = []
                experts_remaining = []

                for i_expert in selected_expert_ids:
                    if self.is_expert_in_gpu_now(i_layer, i_expert):
                        gpu_experts.append(i_expert)
                    else:
                        tc = expert_token_counts.get(i_expert, 0)
                        experts_remaining.append((i_expert, tc))

                experts_remaining.sort(key=lambda x: x[1], reverse=True)

                if not hasattr(self, '_cpu_time_table'):
                    import csv
                    rows = list(csv.DictReader(open('microqwen2.txt')))
                    max_tok = int(rows[-1]['n_tok'])
                    table = [0.0] * (max_tok + 1)
                    for r in rows:
                        table[int(r['n_tok'])] = float(r['time_ms'])

                    prev = 0
                    for i in range(1, len(table)):
                        if table[i] == 0.0:
                            table[i] = table[prev] if prev > 0 else 0.0
                        else:
                            prev = i
                    self._cpu_time_table = table
                    self._cpu_time_table_max = max_tok

                _e = self._cm_e_decode if self.is_decode else self._cm_e_prefill
                _n_rem = len(experts_remaining)
                _n_od = 0
                _cm_log_lines = []
                if _n_rem > 0:
                    _cpu_times = []
                    for eid, tc in experts_remaining:
                        idx = min(tc, self._cpu_time_table_max)
                        if tc <= 8:
                            c = 1.4
                        elif tc <= 64:
                            c = 2.5
                        elif tc <= 512:
                            c = 4.0
                        else:
                            c = 4.5
                        _cpu_times.append(self._cpu_time_table[idx] * c)

                    _T_cpu = sum(_cpu_times)
                    _T_gpu = 0.0
                    for k in range(_n_rem):
                        _T_gpu += _e
                        _T_cpu -= _cpu_times[k]
                        eid_k, tc_k = experts_remaining[k]
                        _cm_log_lines.append(
                            f"_"
                            f"_"
                        )
                        if _T_gpu >= _T_cpu:
                            _n_od = k + 1
                            break
                    else:
                        _n_od = _n_rem

                od_experts_list  = [eid for eid, _ in experts_remaining[:_n_od]]
                cpu_experts_list = [eid for eid, _ in experts_remaining[_n_od:]]

                if not self.is_decode and cpu_experts_list:
                    od_experts_list.append(cpu_experts_list.pop())

                remaining_deque = _collections.deque()

                if self.is_decode:
                    self.layer_breakdown['routing'] += time.time() - _routing_tick

                inps_after_experts = torch.zeros_like(inps, device=self.dev)

                @torch.no_grad()
                def process_gpu_resident():
                    _t0 = time.time()
                    for i_expert in gpu_experts:
                        mask = (selected_experts == i_expert).any(dim=1)
                        if not mask.any():
                            continue
                        expert_input = inps[mask]
                        expert_output = self.run_expert_at_gpu(i_layer, i_expert, expert_input)
                        mask_index = mask.nonzero().squeeze(1)
                        if self.is_decode:
                            _row_i, _col_i = expert_pos_map[i_expert]
                            weights = routing_weights[_row_i, _col_i].item()
                        else:
                            expert_positions = (selected_experts[mask_index] == i_expert).long().argmax(dim=1, keepdim=True)
                            weights = routing_weights[mask_index].gather(1, expert_positions)
                        expert_output = expert_output * weights
                        with _scatter_lock:
                            inps_after_experts.view(-1, hidden_dim).index_add_(
                                0, mask_index, expert_output.view(-1, hidden_dim).to(inps_after_experts.dtype)
                            )
                    _elapsed = time.time() - _t0
                    if self.is_decode:
                        self.detail_gpu_gemm[i_layer] += _elapsed
                    else:
                        self.pf_gpu_gemm[i_layer] += _elapsed
                        self.pf_n_gpu[i_layer] += len(gpu_experts)

                @torch.no_grad()
                def process_gpu_od():
                    _gpu_wall_tick = time.time()
                    _od_stream_h2d  = torch.cuda.Stream()
                    _od_stream_gemm = torch.cuda.Stream()
                    _od_placeholders = [self.expert_placeholder5, self.expert_placeholder6]

                    _n_od = 0
                    _total_h2d_wall = 0.0
                    _total_gemm = 0.0
                    _pending_tasks = []

                    _ph_idx = 0
                    for _expert_id in od_experts_list:
                        mask = (selected_experts == _expert_id).any(dim=1)
                        if not mask.any():
                            continue

                        expert_input = inps[mask]
                        _cur_ph = _od_placeholders[_ph_idx % 2]
                        _h2d_start = time.time()
                        self._async_ondemand(i_layer, _expert_id, _cur_ph, stream=_od_stream_h2d)
                        _h2d_event = torch.cuda.Event()
                        _h2d_event.record(_od_stream_h2d)
                        _pending_tasks.append((_expert_id, _h2d_event, _cur_ph, mask, expert_input, _h2d_start))
                        _ph_idx += 1

                        if len(_pending_tasks) >= 2:
                            _exp_id, _event, _ph, _mask, _input, _h2d_t0 = _pending_tasks.pop(0)
                            _od_stream_gemm.wait_event(_event)
                            _total_h2d_wall += time.time() - _h2d_t0
                            with torch.cuda.stream(_od_stream_gemm):
                                _gemm_tick = time.time()
                                expert_output = _ph(_input)
                                _od_stream_gemm.synchronize()
                                _total_gemm += time.time() - _gemm_tick
                            mask_index = _mask.nonzero().squeeze(1)
                            if self.is_decode:
                                _row_i, _col_i = expert_pos_map[_exp_id]
                                weights = routing_weights[_row_i, _col_i].item()
                            else:
                                expert_positions = (selected_experts[mask_index] == _exp_id).long().argmax(dim=1, keepdim=True)
                                weights = routing_weights[mask_index].gather(1, expert_positions)
                            expert_output = (expert_output * weights).view(-1, hidden_dim)
                            with _scatter_lock:
                                inps_after_experts.view(-1, hidden_dim).index_add_(0, mask_index, expert_output.to(inps_after_experts.dtype))
                            _n_od += 1

                    for _exp_id, _event, _ph, _mask, _input, _h2d_t0 in _pending_tasks:
                        _od_stream_gemm.wait_event(_event)
                        _total_h2d_wall += time.time() - _h2d_t0
                        with torch.cuda.stream(_od_stream_gemm):
                            _gemm_tick = time.time()
                            expert_output = _ph(_input)
                            _od_stream_gemm.synchronize()
                            _total_gemm += time.time() - _gemm_tick
                        mask_index = _mask.nonzero().squeeze(1)
                        if self.is_decode:
                            _row_i, _col_i = expert_pos_map[_exp_id]
                            weights = routing_weights[_row_i, _col_i].item()
                        else:
                            expert_positions = (selected_experts[mask_index] == _exp_id).long().argmax(dim=1, keepdim=True)
                            weights = routing_weights[mask_index].gather(1, expert_positions)
                        expert_output = (expert_output * weights).view(-1, hidden_dim)
                        with _scatter_lock:
                            inps_after_experts.view(-1, hidden_dim).index_add_(0, mask_index, expert_output.to(inps_after_experts.dtype))
                        _n_od += 1

                    if self.is_decode:
                        self.detail_od_h2d[i_layer]  += _total_h2d_wall
                        self.detail_od_gemm[i_layer] += _total_gemm
                        self.detail_n_od[i_layer]    += _n_od
                    else:
                        self.pf_od_h2d[i_layer]  += _total_h2d_wall
                        self.pf_od_gemm[i_layer] += _total_gemm
                        self.pf_n_od[i_layer]    += _n_od

                    _gpu_wall = time.time() - _gpu_wall_tick
                    if self.is_decode:
                        self.detail_gpu_wall[i_layer] += _gpu_wall
                        if _n_od > 0:
                            _new_e = (_total_h2d_wall + _total_gemm) / _n_od
                            self._cm_e_decode = (1 - self._cm_alpha) * self._cm_e_decode + self._cm_alpha * _new_e
                    else:
                        self.pf_gpu_wall[i_layer] += _gpu_wall
                        if _n_od > 0:
                            _new_e = (_total_h2d_wall + _total_gemm) / _n_od
                            self._cm_e_prefill = (1 - self._cm_alpha) * self._cm_e_prefill + self._cm_alpha * _new_e

                    _phase = "decode" if self.is_decode else "prefill"
                    _gpu_wall_ms = _gpu_wall * 1000
                    with self._cost_log_lock:
                        _lines = [f"_"]
                        _lines += _cm_log_lines
                        self._cost_log_file.write("\n".join(_lines) + "\n")
                        self._cost_log_file.flush()

                @torch.no_grad()
                def process_cpu_experts():
                    _cpu_wall_tick = time.time()
                    nonlocal inps_after_experts

                    if self.is_decode:
                        _d2h_tick = time.time()
                        inps_cpu = inps.to("cpu")
                        selected_experts_cpu = selected_experts.to("cpu")
                        routing_weights_cpu = routing_weights.to("cpu")
                        self.detail_cpu_d2h[i_layer] += time.time() - _d2h_tick
                    else:
                        inps_cpu = None
                        selected_experts_cpu = selected_experts
                        routing_weights_cpu = routing_weights

                    cpu_results_local = []
                    _cpu_gemm_tick = time.time()
                    _pf_cpu_detail = []

                    for i_expert in cpu_experts_list:

                        if self.is_decode:
                            mask = (selected_experts_cpu == i_expert).any(dim=1)
                        else:
                            mask = (selected_experts == i_expert).any(dim=1)
                        if not mask.any():
                            continue

                        if self.is_decode:
                            expert_input = inps_cpu[mask]
                        else:
                            _d2h_t0 = time.time()
                            expert_input = inps[mask].to("cpu")
                            _d2h_ms = (time.time() - _d2h_t0) * 1000

                        _expert_t0 = time.time()
                        expert_output = self.run_expert_at_cpu(i_layer, i_expert, expert_input)
                        _expert_elapsed = (time.time() - _expert_t0) * 1000
                        if not self.is_decode and i_layer == 0:
                            _pf_cpu_detail.append((expert_input.shape[0], _expert_elapsed,
                                                   _d2h_ms if not self.is_decode else 0.0))

                        mask_index = mask.nonzero().squeeze(1)
                        if len(mask_index) > 0:
                            if self.is_decode:
                                expert_positions = (selected_experts_cpu[mask_index] == i_expert).long().argmax(dim=1, keepdim=True)
                                weights = routing_weights_cpu[mask_index].gather(1, expert_positions)
                            else:
                                expert_positions = (selected_experts[mask_index] == i_expert).long().argmax(dim=1, keepdim=True)
                                weights = routing_weights[mask_index].gather(1, expert_positions).to("cpu")
                            expert_output = expert_output * weights
                        cpu_results_local.append((mask_index, expert_output))
                        if self.is_decode:
                            self.detail_n_cpu[i_layer] += 1
                        else:
                            self.pf_n_cpu[i_layer] += 1

                    if cpu_results_local:
                        if not self.is_decode and i_layer == 0 and _pf_cpu_detail:
                            print(f"[pf_cpu_detail L0] {len(_pf_cpu_detail)} experts: " +
                                  ", ".join(f"{n}tok/d2h={d:.1f}ms/gemm={t:.1f}ms" for n, t, d in _pf_cpu_detail), flush=True)
                        if self.is_decode:
                            self.detail_cpu_gemm[i_layer] += time.time() - _cpu_gemm_tick
                        else:
                            self.pf_cpu_gemm[i_layer] += time.time() - _cpu_gemm_tick
                        _h2d_tick = time.time()
                        all_indices = torch.cat([idx for idx, _ in cpu_results_local], dim=0).to(self.dev)
                        all_outputs = torch.cat([out.view(-1, hidden_dim) for _, out in cpu_results_local], dim=0).to(self.dev).to(inps_after_experts.dtype)
                        if self.is_decode:
                            self.detail_cpu_h2d[i_layer] += time.time() - _h2d_tick
                        with _scatter_lock:
                            inps_after_experts.view(-1, hidden_dim).index_add_(0, all_indices, all_outputs)

                        if self.is_decode:
                            self.detail_n_cpu[i_layer] += 1
                        else:
                            self.pf_n_cpu[i_layer] += 1

                    if self.is_decode and cpu_results_local:
                        self.detail_cpu_gemm[i_layer] += time.time() - _cpu_gemm_tick

                        _h2d_tick = time.time()
                        all_indices = torch.cat([idx for idx, _ in cpu_results_local], dim=0).to(self.dev)
                        all_outputs = torch.cat([out.view(-1, hidden_dim) for _, out in cpu_results_local], dim=0).to(self.dev).to(inps_after_experts.dtype)
                        self.detail_cpu_h2d[i_layer] += time.time() - _h2d_tick
                        inps_after_experts = inps_after_experts.view(-1, hidden_dim)
                        inps_after_experts.index_add_(0, all_indices, all_outputs)
                    else:
                        if cpu_results_local:
                            self.pf_cpu_gemm[i_layer] += time.time() - _cpu_gemm_tick
                        for mask_index, expert_output in cpu_results_local:
                            expert_output = expert_output.view(-1, hidden_dim)
                            inps_after_experts = inps_after_experts.view(-1, hidden_dim)
                            inps_after_experts.index_add_(
                                0,
                                mask_index.to(self.dev),
                                expert_output.to(self.dev).to(inps_after_experts.dtype)
                            )

                    _cpu_wall = time.time() - _cpu_wall_tick
                    if self.is_decode:
                        self.detail_cpu_wall[i_layer] += _cpu_wall
                    else:
                        self.pf_cpu_wall[i_layer] += _cpu_wall

                    _phase = "decode" if self.is_decode else "prefill"
                    _n_cpu_actual = len(cpu_experts_list)
                    with self._cost_log_lock:
                        self._cost_log_file.write(
                            f"_"
                        )
                        self._cost_log_file.flush()

                parallel_start = time.time()

                resident_thread = threading.Thread(target=process_gpu_resident)
                gpu_od_thread   = threading.Thread(target=process_gpu_od)
                cpu_thread      = threading.Thread(target=process_cpu_experts)
                _thread_tick = time.time() if self.is_decode else 0
                resident_thread.start()
                gpu_od_thread.start()
                cpu_thread.start()

                resident_thread.join()
                gpu_od_thread.join()
                cpu_thread.join()
                if self.is_decode:
                    self.layer_breakdown['expert_threads'] += time.time() - _thread_tick

            inps = inps_residual + inps_after_experts.reshape(batch_size, seq_len, hidden_dim)
            del inps_after_experts, inps_residual

            if not self.is_decode and (i_layer + 1) % 4 == 0:
                torch.cuda.empty_cache()
            if self.is_decode:
                layer_time = time.time() - layer_tick
                self.layer_time_accumulator[i_layer] += layer_time
                layer_times[i_layer] += layer_time
                layer_times_mid[i_layer] += laymid
                layer_times_final[i_layer] += layer_time
        hot_experts = self.get_hot_expert()

        if self.is_decode:

            avg_layer_time = sum(layer_times.values()) / self.n_layer
            total_cpu_time = sum(self.cpu_expert_time_per_layer.values())
            avg_cpu_ratio = (total_cpu_time / avg_layer_time * 100) if avg_layer_time > 0 else 0

            os.makedirs('./log', exist_ok=True)

            stats_lines = "_"
            for i in range(self.n_layer):
                cpu_time = self.cpu_expert_time_per_layer[i] * 1000
                layer_time = layer_times[i] * 1000
                layer_time_mids = layer_times_mid[i] * 1000
                layer_time_finals = layer_times_final[i] * 1000
                stats_lines += f"_"
            self._expert_stats_buf.append(stats_lines)

        other_ops_start = time.time()

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        if self.is_decode:
            other_ops_time = time.time() - other_ops_start
            total_decode_time = time.time() - total_decode_start
        return lm_logis

        """Run the expert at CPU"""

    def run_expert_at_gpu(self, i_layer, i_expert, inps ):
        pass
        return self.model.layers[i_layer].mlp.experts[i_expert](inps)

    def run_expert_at_cpu(self, i_layer, i_expert, inps):
        pass
        start_time = time.time()
        result = self.model.layers[i_layer].mlp.experts[i_expert](inps)
        elapsed = time.time() - start_time
        if self.is_decode:
            self.cpu_expert_time_per_layer[i_layer] += elapsed
            self.detail_cpu_kernel[i_layer] += elapsed
            self.detail_cpu_kernel_ntok[i_layer].append(inps.shape[0])
        else:
            self.pf_cpu_kernel[i_layer] += elapsed
            self.pf_cpu_kernel_ntok[i_layer].append(inps.shape[0])
        return result
    def get_expert_stats(self):
        pass
        stats = {
            'hot_experts': {i: {'count': 0, 'hot': 0, 'veryhot': 0} for i in range(self.n_layer)},
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

        layer_hot_counts = {i: 0 for i in range(self.n_layer)}
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

