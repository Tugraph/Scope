pass
import sys, os, argparse, time, threading
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    'qwen3':   dict(name='Qwen3-30B-A3B',     NH=4, GROUPS=8, QHD=128, VHD=128, NL=48, cpu_fn='fused_cpu_attn'),
    'ds2':     dict(name='DeepSeek-V2',       NH=16, GROUPS=1, QHD=192, VHD=128, NL=27, cpu_fn='fused_cpu_attn_mla'),
    'mixtral': dict(name='Mixtral-8x7B-v0.1', NH=8, GROUPS=4, QHD=128, VHD=128, NL=32, cpu_fn='fused_cpu_attn'),
}

DS2_TEST1 = {
    (4, 2048): (6, 1536), (4, 4096): (32, 2048), (4, 8192): (48, 2048),
    (4, 16384): (60, 4096), (4, 32768): (48, 8192), (4, 65536): (48, 16384),
    (8, 2048): (32, 1024), (8, 4096): (64, 2048), (8, 8192): (112, 2048),
    (8, 16384): (96, 4096), (8, 32768): (112, 8192), (8, 65536): (112, 16384),
    (16, 2048): (128, 256), (16, 4096): (224, 256), (16, 8192): (192, 1024),
    (16, 16384): (192, 4096), (16, 32768): (240, 8192), (16, 65536): (224, 16384),
    (32, 2048): (480, 256), (32, 4096): (448, 256), (32, 8192): (480, 512),
    (32, 16384): (448, 1024), (32, 32768): (448, 4096), (32, 65536): (448, 8192),
}

QWEN_TEST1 = {
    (4, 2048): (2, 1536), (4, 4096): (2, 3072), (4, 8192): (2, 6144),
    (4, 16384): (2, 12288), (4, 32768): (4, 24576), (4, 65536): (8, 32768),
    (8, 2048): (3, 512), (8, 4096): (3, 1024), (8, 8192): (8, 4096),
    (8, 16384): (8, 1024), (8, 32768): (16, 819), (8, 65536): (16, 32768),
    (16, 2048): (48, 256), (16, 4096): (6, 256), (16, 8192): (16, 256),
    (16, 16384): (16, 410), (16, 32768): (16, 24576), (16, 65536): (32, 49152),
    (32, 2048): (13, 256), (32, 4096): (120, 256), (32, 8192): (13, 256),
    (32, 16384): (32, 12288), (32, 32768): (64, 16384), (32, 65536): (64, 32768),
}

MIXTRAL_TEST1 = {
    (4, 2048): (3, 1536), (4, 4096): (3, 3072), (4, 8192): (3, 6144),
    (4, 16384): (8, 8192), (4, 32768): (16, 16384), (4, 65536): (16, 32768),
    (8, 2048): (6, 512), (8, 4096): (6, 256), (8, 8192): (32, 6144),
    (8, 16384): (32, 12288), (8, 32768): (32, 16384), (8, 65536): (32, 32768),
    (16, 2048): (13, 256), (16, 4096): (32, 256), (16, 8192): (32, 256),
    (16, 16384): (64, 410), (16, 32768): (64, 16384), (16, 65536): (64, 32768),
    (32, 2048): (240, 256), (32, 4096): (240, 256), (32, 8192): (192, 256),
    (32, 16384): (128, 8192), (32, 32768): (128, 16384), (32, 65536): (128, 16384),
}

TABLES = {'ds2': DS2_TEST1, 'qwen3': QWEN_TEST1, 'mixtral': MIXTRAL_TEST1}

DS2_TEST1_MS = {
    (4, 2048): (6, 3.97, 1536, 3.92), (4, 4096): (32, 5.08, 2048, 4.92),
    (4, 8192): (48, 7.19, 2048, 6.87), (4, 16384): (60, 11.20, 4096, 10.97),
    (4, 32768): (48, 17.67, 8192, 18.41), (4, 65536): (48, 30.03, 16384, 28.98),
    (8, 2048): (32, 5.29, 1024, 4.66), (8, 4096): (64, 7.10, 2048, 7.16),
    (8, 8192): (112, 10.64, 2048, 10.00), (8, 16384): (96, 18.38, 4096, 18.43),
    (8, 32768): (112, 30.36, 8192, 28.81), (8, 65536): (112, 53.95, 16384, 54.58),
    (16, 2048): (128, 7.04, 256, 10.00), (16, 4096): (224, 10.81, 256, 14.44),
    (16, 8192): (192, 17.81, 1024, 21.19), (16, 16384): (192, 30.01, 4096, 36.41),
    (16, 32768): (240, 54.51, 8192, 57.45), (16, 65536): (224, 100.22, 16384, 108.91),
    (32, 2048): (480, 13.76, 256, 12.62), (32, 4096): (448, 19.89, 256, 19.27),
    (32, 8192): (480, 33.12, 512, 30.92), (32, 16384): (448, 65.45, 1024, 54.62),
    (32, 32768): (448, 108.83, 4096, 103.29), (32, 65536): (448, 194.92, 8192, 171.20),
}

QWEN_TEST1_MS = {
    (4, 2048): (2, 4.07, 1536, 3.80), (4, 4096): (2, 4.08, 3072, 4.19),
    (4, 8192): (2, 4.70, 6144, 4.74), (4, 16384): (2, 5.28, 12288, 5.17),
    (4, 32768): (4, 8.83, 24576, 8.69), (4, 65536): (8, 16.43, 32768, 15.08),
    (8, 2048): (3, 4.07, 512, 2.38), (8, 4096): (3, 4.36, 1024, 3.54),
    (8, 8192): (8, 5.58, 4096, 5.32), (8, 16384): (8, 9.41, 1024, 10.63),
    (8, 32768): (16, 16.40, 819, 23.07), (8, 65536): (16, 26.99, 32768, 36.70),
    (16, 2048): (48, 3.97, 256, 3.94), (16, 4096): (6, 5.80, 256, 6.82),
    (16, 8192): (16, 9.44, 256, 11.86), (16, 16384): (16, 18.46, 410, 22.43),
    (16, 32768): (16, 36.59, 24576, 41.40), (16, 65536): (32, 62.54, 49152, 68.05),
    (32, 2048): (13, 5.69, 256, 5.51), (32, 4096): (120, 10.72, 256, 9.53),
    (32, 8192): (13, 21.92, 256, 18.55), (32, 16384): (32, 39.42, 12288, 37.10),
    (32, 32768): (64, 72.50, 16384, 60.90), (32, 65536): (64, 118.33, 32768, 106.63),
}

MIXTRAL_TEST1_MS = {
    (4, 2048): (3, 4.20, 1536, 4.11), (4, 4096): (3, 4.27, 3072, 4.48),
    (4, 8192): (3, 5.20, 6144, 5.27), (4, 16384): (8, 8.52, 8192, 8.38),
    (4, 32768): (16, 13.10, 16384, 13.58), (4, 65536): (16, 22.27, 32768, 22.50),
    (8, 2048): (6, 4.42, 512, 3.53), (8, 4096): (6, 5.45, 256, 5.88),
    (8, 8192): (32, 8.69, 6144, 8.53), (8, 16384): (32, 13.09, 12288, 19.61),
    (8, 32768): (32, 22.33, 16384, 24.44), (8, 65536): (32, 44.13, 32768, 50.58),
    (16, 2048): (13, 5.43, 256, 5.87), (16, 4096): (32, 8.78, 256, 9.93),
    (16, 8192): (32, 17.25, 256, 18.04), (16, 16384): (64, 32.26, 410, 33.82),
    (16, 32768): (64, 54.33, 16384, 59.05), (16, 65536): (64, 97.08, 32768, 90.85),
    (32, 2048): (240, 8.20, 256, 10.76), (32, 4096): (240, 15.58, 256, 17.73),
    (32, 8192): (192, 32.16, 256, 30.23), (32, 16384): (128, 54.47, 8192, 50.27),
    (32, 32768): (128, 93.54, 16384, 89.73), (32, 65536): (128, 182.75, 16384, 165.54),
}

TABLES_MS = {'ds2': DS2_TEST1_MS, 'qwen3': QWEN_TEST1_MS, 'mixtral': MIXTRAL_TEST1_MS}

BATCH_SIZES = [4, 8, 16, 32]
KV_LENS = [2048, 4096, 8192, 16384, 32768, 65536]

_RAM_SANITY_CAP_GB = 150.0

DEV   = 'cuda:0'
dtype = torch.bfloat16
_NUMA1_CPUS = list(range(24, 48)) + list(range(72, 96))

GiB = 1024 ** 3
BYTES_BF16 = 2
BYTES_F32 = 4
BUDGET_BYTES = 4 * GiB
CHUNK = 512

_CPU_EXT = None
try:
    from cpu_attn_ext.build import build as _build
    _CPU_EXT = _build(verbose=False)
    _CPU_EXT.set_num_threads(16)
    print("C++ kernel loaded")
except Exception as e:
    print(f"[warn] C++ kernel: {e}")

NH = GROUPS = QHD = VHD = NL = None
_CPU_FN_NAME = None

def set_model(model_key):
    global NH, GROUPS, QHD, VHD, NL, _CPU_FN_NAME
    cfg = MODELS[model_key]
    NH, GROUPS, QHD, VHD, NL = cfg['NH'], cfg['GROUPS'], cfg['QHD'], cfg['VHD'], cfg['NL']
    _CPU_FN_NAME = cfg['cpu_fn']

def _cpu_attn(q, k, v, scale):
    pass
    if _CPU_EXT is not None:
        fn = getattr(_CPU_EXT, _CPU_FN_NAME)
        return fn(q, k, v, scale)
    sc    = torch.bmm(q, k.contiguous().transpose(1, 2)) * scale
    max_c = sc.amax(-1, keepdim=True).float()
    ec    = torch.exp(sc.float() - max_c)
    sum_c = ec.sum(-1, keepdim=True)
    out_u = torch.bmm(ec.bfloat16(), v.contiguous()).float()
    return out_u, max_c, sum_c

def _pairwise_merge(o1, m1, s1, o2, m2, s2):
    pass
    mx = torch.maximum(m1, m2)
    w1 = torch.exp(m1 - mx)
    w2 = torch.exp(m2 - mx)
    s  = w1 * s1 + w2 * s2
    o  = w1 * o1 + w2 * o2
    return o, mx, s

def _local_bmm_softmax(q, k, v, scale):
    pass
    sc  = torch.bmm(q, k.transpose(1, 2)) * scale
    mc  = sc.amax(-1, keepdim=True).float()
    ec  = torch.exp(sc.float() - mc)
    sc2 = ec.sum(-1, keepdim=True)
    oc  = torch.bmm(ec.bfloat16(), v).float()
    return oc, mc, sc2

def _pingpong_stream(q, k_chunks, v_chunks, n_heads, scale):
    pass
    chunk = CHUNK
    n_chunks = len(k_chunks)
    key = (n_heads, chunk)
    if key not in _pp_bufs:
        _pp_bufs[key] = {
            'k': [torch.empty(n_heads, chunk, QHD, dtype=dtype, device=DEV),
                  torch.empty(n_heads, chunk, QHD, dtype=dtype, device=DEV)],
            'v': [torch.empty(n_heads, chunk, VHD, dtype=dtype, device=DEV),
                  torch.empty(n_heads, chunk, VHD, dtype=dtype, device=DEV)],
            'stream': torch.cuda.Stream(device=DEV),
        }
    bufs = _pp_bufs[key]
    buf_k, buf_v = bufs['k'], bufs['v']
    h2d_st = bufs['stream']

    oa = torch.zeros(n_heads, GROUPS, VHD, device=DEV, dtype=torch.float32)
    ma = torch.full((n_heads, GROUPS, 1), float('-inf'), device=DEV, dtype=torch.float32)
    sa = torch.zeros(n_heads, GROUPS, 1, device=DEV, dtype=torch.float32)
    if n_chunks == 0:
        return oa, ma, sa

    sizes = [t.shape[1] for t in k_chunks]
    s0 = sizes[0]
    with torch.cuda.stream(h2d_st):
        buf_k[0][:, :s0].copy_(k_chunks[0], non_blocking=True)
        buf_v[0][:, :s0].copy_(v_chunks[0], non_blocking=True)

    for ci in range(n_chunks):
        csz = sizes[ci]
        cur = ci % 2; nxt = 1 - cur
        torch.cuda.current_stream().wait_stream(h2d_st)
        if ci + 1 < n_chunks:
            nsz = sizes[ci + 1]
            with torch.cuda.stream(h2d_st):
                buf_k[nxt][:, :nsz].copy_(k_chunks[ci + 1], non_blocking=True)
                buf_v[nxt][:, :nsz].copy_(v_chunks[ci + 1], non_blocking=True)
        kc = buf_k[cur][:, :csz]
        vc = buf_v[cur][:, :csz]
        oc, mc, sc3 = _local_bmm_softmax(q, kc, vc, scale)
        oa, ma, sa = _pairwise_merge(oa, ma, sa, oc, mc, sc3)
    return oa, ma, sa

def _prep_pingpong_chunks(n_heads, length, head_dim):
    pass
    if length <= 0:
        return []
    chunk = CHUNK
    n_chunks = (length + chunk - 1) // chunk
    return [torch.empty(n_heads, min(chunk, length - i * chunk), head_dim,
                         dtype=dtype).pin_memory() for i in range(n_chunks)]

_pp_bufs = {}

def scratch_bytes(n_heads, transferred_len):
    pass
    if n_heads <= 0:
        return 0
    pingpong = 2 * n_heads * CHUNK * (QHD + VHD) * BYTES_BF16
    softmax  = n_heads * (2 + VHD) * BYTES_F32
    return pingpong + softmax

def n_resident_bkv(b_cpu, b_gpu, kv):
    scr = scratch_bytes(b_gpu, kv)
    budget = BUDGET_BYTES - scr
    if budget <= 0 or b_cpu <= 0:
        return 0
    per_head = NL * kv * (QHD + VHD) * BYTES_BF16
    return max(0, min(b_cpu, int(budget // per_head))) if per_head else 0

def n_resident_kvlen(bkv_total, gs, kv):
    scr = scratch_bytes(bkv_total, gs)
    budget = BUDGET_BYTES - scr
    cpu_tokens = kv - gs
    if budget <= 0 or cpu_tokens <= 0:
        return 0
    per_tok = NL * bkv_total * (QHD + VHD) * BYTES_BF16
    return max(0, min(cpu_tokens, int(budget // per_tok))) if per_tok else 0

def align_gs_to_chunk(gs, n_resident):
    pass
    if gs <= CHUNK or gs % CHUNK == 0:
        return gs, n_resident
    deficit = CHUNK - (gs % CHUNK)
    if n_resident >= deficit:
        return gs + deficit, n_resident - deficit
    return gs, n_resident

_resident_bufs = {}

def _get_resident_buf(n_heads, cap):
    key = (n_heads, cap, 'resident')
    if key not in _resident_bufs:
        _resident_bufs[key] = dict(
            k=torch.randn(n_heads, cap, QHD, dtype=dtype, device=DEV),
            v=torch.randn(n_heads, cap, VHD, dtype=dtype, device=DEV),
            new_k=torch.randn(n_heads, 1, QHD, dtype=dtype, device=DEV),
            new_v=torch.randn(n_heads, 1, VHD, dtype=dtype, device=DEV),
        )
    return _resident_bufs[key]

def _resident_layer(q, n_heads, cap, scale):
    pass
    if n_heads <= 0 or cap <= 0:
        z_o = torch.zeros(max(n_heads, 0), GROUPS, VHD, device=DEV, dtype=torch.float32)
        z_m = torch.full((max(n_heads, 0), GROUPS, 1), float('-inf'), device=DEV, dtype=torch.float32)
        z_s = torch.zeros(max(n_heads, 0), GROUPS, 1, device=DEV, dtype=torch.float32)
        return z_o, z_m, z_s
    buf = _get_resident_buf(n_heads, cap)
    buf['k'][:, -1:, :].copy_(buf['new_k'])
    buf['v'][:, -1:, :].copy_(buf['new_v'])
    return _local_bmm_softmax(q, buf['k'], buf['v'], scale)

def run_bkv_resident(B, kv, q_gpu, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin, b_cpu, b_gpu, n_resident):
    pass
    scale = QHD ** -0.5
    bkv = B * NH
    q_g = q_gpu.view(bkv, GROUPS, QHD)
    b_cpu_remain = b_cpu - n_resident

    q_cpu = q_g[:b_cpu_remain].cpu().contiguous() if b_cpu_remain > 0 else None
    q_res = q_g[b_cpu_remain:b_cpu] if n_resident > 0 else None
    q_str = q_g[b_cpu:]

    cpu_res = {}
    def _cpu_fn():
        try: os.sched_setaffinity(0, _NUMA1_CPUS)
        except Exception: pass
        out_u, _, lse = _cpu_attn(q_cpu, k_cpu_n1, v_cpu_n1, scale)
        cpu_res['out'] = (out_u / lse.clamp(min=1e-6)).bfloat16()

    thr = None
    if b_cpu_remain > 0:
        thr = threading.Thread(target=_cpu_fn, daemon=True); thr.start()

    out_res = None
    if n_resident > 0:
        oa_r, ma_r, sa_r = _resident_layer(q_res, n_resident, kv, scale)
        out_res = (oa_r / sa_r.clamp(min=1e-6)).bfloat16()

    oa_s, ma_s, sa_s = _pingpong_stream(q_str, k_gpu_pin, v_gpu_pin, b_gpu, scale)
    out_str = (oa_s / sa_s.clamp(min=1e-6)).bfloat16()

    torch.cuda.synchronize(DEV)
    parts = []
    if b_cpu_remain > 0:
        thr.join()
        parts.append(cpu_res['out'].to(DEV))
    if n_resident > 0:
        parts.append(out_res)
    parts.append(out_str)
    out_all = torch.cat(parts, dim=0)
    return out_all.view(B, NH * GROUPS, 1, VHD)

def run_kvlen_resident(B, kv, q_gpu, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin, gs, n_resident):
    pass
    scale = QHD ** -0.5
    bkv = B * NH
    q_g = q_gpu.view(bkv, GROUPS, QHD)
    cpu_tokens = kv - gs - n_resident
    q_cpu = q_g.cpu().contiguous() if cpu_tokens > 0 else None

    cpu_res = {}
    def _cpu_fn():
        try: os.sched_setaffinity(0, _NUMA1_CPUS)
        except Exception: pass
        out_c, max_c, sum_c = _cpu_attn(q_cpu, k_cpu_n1, v_cpu_n1, scale)
        cpu_res.update({'out': out_c, 'max': max_c, 'sum': sum_c})

    thr = None
    if cpu_tokens > 0:
        thr = threading.Thread(target=_cpu_fn, daemon=True); thr.start()

    oa_r, ma_r, sa_r = _resident_layer(q_g, bkv, n_resident, scale) if n_resident > 0 else (
        torch.zeros(bkv, GROUPS, VHD, device=DEV, dtype=torch.float32),
        torch.full((bkv, GROUPS, 1), float('-inf'), device=DEV, dtype=torch.float32),
        torch.zeros(bkv, GROUPS, 1, device=DEV, dtype=torch.float32),
    )
    oa_s, ma_s, sa_s = _pingpong_stream(q_g, k_gpu_pin, v_gpu_pin, bkv, scale)
    oa_g, ma_g, sa_g = _pairwise_merge(oa_r, ma_r, sa_r, oa_s, ma_s, sa_s)

    torch.cuda.synchronize(DEV)
    if cpu_tokens > 0:
        thr.join()
        max_c = cpu_res['max'].to(DEV)
        sum_c = cpu_res['sum'].to(DEV)
        out_c = cpu_res['out'].to(DEV)
        o_all, m_all, s_all = _pairwise_merge(oa_g, ma_g, sa_g, out_c, max_c, sum_c)
        out = (o_all / s_all.clamp(min=1e-6)).bfloat16()
    else:
        out = (oa_g / sa_g.clamp(min=1e-6)).bfloat16()

    return out.view(B, NH * GROUPS, 1, VHD)

def run_bkv_baseline(B, kv, q_gpu, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin, b_cpu, b_gpu):
    return run_bkv_resident(B, kv, q_gpu, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin, b_cpu, b_gpu, 0)

def run_kvlen_baseline(B, kv, q_gpu, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin, gs):
    return run_kvlen_resident(B, kv, q_gpu, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin, gs, 0)

def _check_close(out_a, out_b, tag, atol=2e-2, rtol=2e-2):
    ok = torch.allclose(out_a.float(), out_b.float(), atol=atol, rtol=rtol)
    diff = (out_a.float() - out_b.float()).abs().max().item()
    print(f"    [check {tag}] allclose={ok}  max_abs_diff={diff:.4f}")
    return ok

def _clear_gpu_buf_caches():
    pass
    _pp_bufs.clear()
    _resident_bufs.clear()
    torch.cuda.empty_cache()

def measure(fn, warmup=3, meas=8):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.cuda.synchronize(DEV)
        t0 = time.perf_counter()
        for _ in range(meas): fn()
        torch.cuda.synchronize(DEV)
    return (time.perf_counter() - t0) / meas * 1000

def _numa1_alloc(shape):
    orig = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, _NUMA1_CPUS)
        t = torch.randn(shape, dtype=dtype)
    finally:
        os.sched_setaffinity(0, orig)
    return t

def _prep_bkv(B, kv, b_cpu):
    bkv = B * NH
    b_gpu = bkv - b_cpu
    q_gpu = torch.randn(bkv, GROUPS, QHD, dtype=dtype, device=DEV)
    k_cpu_n1 = _numa1_alloc((b_cpu, kv, QHD)) if b_cpu > 0 else None
    v_cpu_n1 = _numa1_alloc((b_cpu, kv, VHD)) if b_cpu > 0 else None
    k_gpu_pin = _prep_pingpong_chunks(b_gpu, kv, QHD)
    v_gpu_pin = _prep_pingpong_chunks(b_gpu, kv, VHD)
    return q_gpu, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin

def _prep_kvlen(B, kv, gs):
    bkv = B * NH
    cpu_seq = kv - gs
    q_gpu = torch.randn(bkv, GROUPS, QHD, dtype=dtype, device=DEV)
    k_gpu_pin = _prep_pingpong_chunks(bkv, gs, QHD)
    v_gpu_pin = _prep_pingpong_chunks(bkv, gs, VHD)
    k_cpu_n1 = _numa1_alloc((bkv, cpu_seq, QHD)) if cpu_seq > 0 else None
    v_cpu_n1 = _numa1_alloc((bkv, cpu_seq, VHD)) if cpu_seq > 0 else None
    return q_gpu, k_cpu_n1, v_cpu_n1, k_gpu_pin, v_gpu_pin

def correctness_check(model_key, mode):
    set_model(model_key)
    torch.manual_seed(0)
    B, kv = 2, 4096
    bkv = B * NH
    scale = QHD ** -0.5

    if mode == 'bkv':
        b_cpu, b_gpu = bkv // 2, bkv - bkv // 2
        n_res = max(1, b_cpu // 2)
        q = torch.randn(bkv, GROUPS, QHD, dtype=dtype, device=DEV)
        k_full = torch.randn(bkv, kv, QHD, dtype=dtype, device=DEV)
        v_full = torch.randn(bkv, kv, VHD, dtype=dtype, device=DEV)

        ref_o, ref_m, ref_s = _local_bmm_softmax(q, k_full, v_full, scale)
        ref = (ref_o / ref_s.clamp(min=1e-6)).bfloat16()

        b_cpu_remain = b_cpu - n_res
        parts = []
        if b_cpu_remain > 0:
            o, m, s = _local_bmm_softmax(q[:b_cpu_remain], k_full[:b_cpu_remain], v_full[:b_cpu_remain], scale)
            parts.append((o / s.clamp(min=1e-6)).bfloat16())
        if n_res > 0:
            o, m, s = _local_bmm_softmax(q[b_cpu_remain:b_cpu], k_full[b_cpu_remain:b_cpu], v_full[b_cpu_remain:b_cpu], scale)
            parts.append((o / s.clamp(min=1e-6)).bfloat16())
        o, m, s = _local_bmm_softmax(q[b_cpu:], k_full[b_cpu:], v_full[b_cpu:], scale)
        parts.append((o / s.clamp(min=1e-6)).bfloat16())
        got = torch.cat(parts, dim=0)
    else:
        gs = kv // 4
        n_res = kv // 4
        cpu_len = kv - gs - n_res
        q = torch.randn(bkv, GROUPS, QHD, dtype=dtype, device=DEV)
        k_full = torch.randn(bkv, kv, QHD, dtype=dtype, device=DEV)
        v_full = torch.randn(bkv, kv, VHD, dtype=dtype, device=DEV)

        ref_o, ref_m, ref_s = _local_bmm_softmax(q, k_full, v_full, scale)
        ref = (ref_o / ref_s.clamp(min=1e-6)).bfloat16()

        o_s, m_s, s_s = _local_bmm_softmax(q, k_full[:, :gs], v_full[:, :gs], scale)
        o_r, m_r, s_r = _local_bmm_softmax(q, k_full[:, gs:gs + n_res], v_full[:, gs:gs + n_res], scale)
        o_c, m_c, s_c = _local_bmm_softmax(q, k_full[:, gs + n_res:], v_full[:, gs + n_res:], scale)
        o1, m1, s1 = _pairwise_merge(o_s, m_s, s_s, o_r, m_r, s_r)
        o2, m2, s2 = _pairwise_merge(o1, m1, s1, o_c, m_c, s_c)
        got = (o2 / s2.clamp(min=1e-6)).bfloat16()

    ok = _check_close(got, ref, f"correctness/{model_key}/{mode}")
    if not ok:
        raise AssertionError(f"_")
    print(f"_")

def smoke_test(model_key, mode):
    set_model(model_key)
    table = TABLES[model_key]
    B, kv = 4, 4096
    b_cpu, gs = table[(B, kv)]
    bkv = B * NH
    b_gpu = bkv - b_cpu

    print(f"_")

    if mode == 'bkv':
        n_res = n_resident_bkv(b_cpu, b_gpu, kv)
        print(f"_")
        q_gpu, k_n1, v_n1, k_pin, v_pin = _prep_bkv(B, kv, b_cpu)

        out_r0 = run_bkv_baseline(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, b_cpu, b_gpu)
        out_rn = run_bkv_resident(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, b_cpu, b_gpu, n_res)

        print(f"  out_r0 shape={tuple(out_r0.shape)} finite={torch.isfinite(out_r0).all().item()}")
        print(f"  out_rn shape={tuple(out_rn.shape)} finite={torch.isfinite(out_rn).all().item()}")

        ms_r0 = measure(lambda: run_bkv_baseline(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, b_cpu, b_gpu))
        ms_rn = measure(lambda: run_bkv_resident(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, b_cpu, b_gpu, n_res))
        print(f"  R=0        : {ms_r0:.3f} ms")
        print(f"  R={n_res:<4d}    : {ms_rn:.3f} ms  (speedup={ms_r0/ms_rn:.3f}x)")
    else:
        n_res = n_resident_kvlen(bkv, gs, kv)
        gs, n_res = align_gs_to_chunk(gs, n_res)
        cpu_tokens = kv - gs
        print(f"_")
        q_gpu, k_n1, v_n1, k_pin, v_pin = _prep_kvlen(B, kv, gs)

        out_r0 = run_kvlen_baseline(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, gs)
        out_rn = run_kvlen_resident(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, gs, n_res)
        print(f"  out_r0 shape={tuple(out_r0.shape)} finite={torch.isfinite(out_r0).all().item()}")
        print(f"  out_rn shape={tuple(out_rn.shape)} finite={torch.isfinite(out_rn).all().item()}")

        ms_r0 = measure(lambda: run_kvlen_baseline(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, gs))
        ms_rn = measure(lambda: run_kvlen_resident(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, gs, n_res))
        print(f"  R=0        : {ms_r0:.3f} ms")
        print(f"  R={n_res:<4d}    : {ms_rn:.3f} ms  (speedup={ms_r0/ms_rn:.3f}x)")

    print("_")

def full_sweep(model_key, mode, out_path=None):
    set_model(model_key)
    table = TABLES[model_key]
    table_ms = TABLES_MS[model_key]
    rows = []

    for B in BATCH_SIZES:
        bkv = B * NH
        for kv in KV_LENS:
            b_cpu, gs = table[(B, kv)]
            b_cpu_ms_ref, bkv_ms_ref, gs_ms_ref, kvlen_ms_ref = table_ms[(B, kv)]
            b_gpu = bkv - b_cpu

            if mode == 'bkv':
                n_res = n_resident_bkv(b_cpu, b_gpu, kv)
                ram_gb = (b_cpu * kv * (QHD + VHD) * 2 + b_gpu * kv * (QHD + VHD) * 2) / GiB
                if ram_gb > _RAM_SANITY_CAP_GB:
                    print(f"  [SKIP B={B} kv={kv}: pinned/numa RAM {ram_gb:.1f}GB > {_RAM_SANITY_CAP_GB}GB]")
                    rows.append((B, kv, b_cpu, n_res, bkv_ms_ref, float('nan')))
                    continue
                q_gpu, k_n1, v_n1, k_pin, v_pin = _prep_bkv(B, kv, b_cpu)
                ms_rn = measure(lambda: run_bkv_resident(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, b_cpu, b_gpu, n_res))
                del q_gpu, k_n1, v_n1, k_pin, v_pin
                rows.append((B, kv, b_cpu, n_res, bkv_ms_ref, ms_rn))
                print(f"  B={B:2d} kv={kv:6d}  b_cpu={b_cpu:4d}  n_res={n_res:4d}  "
                      f"Q48_ref={bkv_ms_ref:8.2f}ms  resident={ms_rn:8.2f}ms  "
                      f"speedup={bkv_ms_ref/ms_rn:.3f}x", flush=True)
            else:
                n_res = n_resident_kvlen(bkv, gs, kv)
                gs, n_res = align_gs_to_chunk(gs, n_res)
                cpu_tokens_final = kv - gs - n_res
                ram_gb = (bkv * gs * (QHD + VHD) * 2 + bkv * (kv - gs) * (QHD + VHD) * 2) / GiB
                if ram_gb > _RAM_SANITY_CAP_GB:
                    print(f"  [SKIP B={B} kv={kv}: pinned/numa RAM {ram_gb:.1f}GB > {_RAM_SANITY_CAP_GB}GB]")
                    rows.append((B, kv, gs, n_res, kvlen_ms_ref, float('nan')))
                    continue
                q_gpu, k_n1, v_n1, k_pin, v_pin = _prep_kvlen(B, kv, gs)
                ms_rn = measure(lambda: run_kvlen_resident(B, kv, q_gpu, k_n1, v_n1, k_pin, v_pin, gs, n_res))
                del q_gpu, k_n1, v_n1, k_pin, v_pin
                rows.append((B, kv, gs, n_res, kvlen_ms_ref, ms_rn))
                print(f"  B={B:2d} kv={kv:6d}  gs={gs:6d}  n_res={n_res:6d}  "
                      f"Q48_ref={kvlen_ms_ref:8.2f}ms  resident={ms_rn:8.2f}ms  "
                      f"speedup={kvlen_ms_ref/ms_rn:.3f}x", flush=True)

            _clear_gpu_buf_caches()

    lines = []
    lines.append(f"_")
    if mode == 'bkv':
        lines.append(f"{'BS':>4} {'KV':>7} | {'b_cpu':>6} {'n_res':>6} | {'Q48_ms':>9} {'resid_ms':>9} {'speedup':>8}")
    else:
        lines.append(f"{'BS':>4} {'KV':>7} | {'gs':>7} {'n_res':>7} | {'Q48_ms':>9} {'resid_ms':>9} {'speedup':>8}")
    lines.append("─" * 70)
    for B, kv, split, n_res, ref_ms, resid_ms in rows:
        if resid_ms != resid_ms:
            lines.append(f"{B:>4} {kv:>7} | {split:>6} {n_res:>6} | {ref_ms:>8.2f}ms {'SKIP':>9} {'?':>8}")
        else:
            speedup = ref_ms / resid_ms
            lines.append(f"{B:>4} {kv:>7} | {split:>6} {n_res:>6} | {ref_ms:>8.2f}ms {resid_ms:>8.2f}ms {speedup:>7.3f}x")
    table_str = "\n".join(lines)
    print("\n" + table_str)

    if out_path:
        with open(out_path, 'a') as f:
            f.write(table_str + "\n\n")
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS), default='ds2')
    parser.add_argument('--mode', choices=['bkv', 'kvlen'], default='kvlen')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--out', default='perf_log_resident_kv.md')
    args = parser.parse_args()

    if args.smoke:
        correctness_check(args.model, args.mode)
        smoke_test(args.model, args.mode)
    elif args.full:
        full_sweep(args.model, args.mode, out_path=args.out)
    else:
        smoke_test(args.model, args.mode)

