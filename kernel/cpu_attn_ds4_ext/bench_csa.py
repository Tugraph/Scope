
pass
import torch
import time
import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(__file__))
from build import build_csa_attn

DEV = torch.device("cuda:0")
BSZ = 16
N_HEADS = 64
SEQLEN = 256
HEAD_DIM = 512
GROUP = 512
WIN = 128

def cuda_sync_time(fn, warmup=3, iters=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def cpu_wall_time(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000

def make_data(bs, n_compress):
    scale = 1.0 / (HEAD_DIM ** 0.5)
    q_gpu = torch.randn(bs, SEQLEN, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    kv_win = torch.randn(bs, WIN, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    kv_chunk = torch.randn(bs, SEQLEN, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    cpu_buf = torch.randn(bs, n_compress, HEAD_DIM, dtype=torch.bfloat16).pin_memory()
    pp = [
        torch.zeros(bs, GROUP, HEAD_DIM, dtype=torch.bfloat16, device=DEV),
        torch.zeros(bs, GROUP, HEAD_DIM, dtype=torch.bfloat16, device=DEV),
    ]
    stream_h2d = torch.cuda.Stream(device=DEV)
    return scale, q_gpu, kv_win, kv_chunk, cpu_buf, pp, stream_h2d

def gpu_window_fn(q_gpu, kv_win, kv_chunk, scale):
    kv_gpu = torch.cat([kv_win, kv_chunk], dim=1)
    q_perm = q_gpu.permute(0, 2, 1, 3)
    kv_gpu_b = kv_gpu.unsqueeze(1)
    attn_bias = torch.zeros(SEQLEN, WIN + SEQLEN, device=DEV, dtype=torch.float32)
    intra_mask = torch.ones(SEQLEN, SEQLEN, device=DEV, dtype=torch.bool).triu(diagonal=1)
    attn_bias[:, WIN:].masked_fill_(intra_mask, float('-inf'))
    scores_g = torch.einsum('bnqd,bnkd->bnqk', q_perm, kv_gpu_b).float() * scale + attn_bias
    lse_g = torch.logsumexp(scores_g, dim=-1)
    exp_g = torch.exp(scores_g - lse_g.unsqueeze(-1))
    o_gpu = torch.einsum('bnqk,bnkd->bnqd', exp_g, kv_gpu_b.float()).permute(0, 2, 1, 3)
    return o_gpu, lse_g

def gpu_compress_fn(q_gpu, cpu_buf, pp, stream_h2d, scale, n_compress, bs):
    q_perm_c = q_gpu.permute(0, 2, 1, 3).float()
    lse_run = torch.full((bs, N_HEADS, SEQLEN), float('-inf'), device=DEV, dtype=torch.float32)
    o_run   = torch.zeros(bs, N_HEADS, SEQLEN, HEAD_DIM, device=DEV, dtype=torch.float32)
    n_groups = (n_compress + GROUP - 1) // GROUP
    g0_end = min(GROUP, n_compress)
    with torch.cuda.stream(stream_h2d):
        pp[0][:bs, :g0_end].copy_(cpu_buf[:bs, :g0_end].contiguous())
    for gi in range(n_groups):
        g_s = gi * GROUP
        g_e = min(g_s + GROUP, n_compress)
        gs = g_e - g_s
        cur = gi % 2
        torch.cuda.current_stream(DEV).wait_stream(stream_h2d)
        if gi + 1 < n_groups:
            nxt = 1 - cur
            ns = (gi + 1) * GROUP
            ne = min(ns + GROUP, n_compress)
            with torch.cuda.stream(stream_h2d):
                pp[nxt][:bs, :ne - ns].copy_(cpu_buf[:bs, ns:ne].contiguous())
        kv_tile = pp[cur][:bs, :gs].unsqueeze(1)
        scores_tile = torch.einsum('bnqd,bnkd->bnqk', q_perm_c, kv_tile.float()) * scale
        m_tile = scores_tile.amax(dim=-1)
        lse_new = m_tile + torch.log(torch.exp(scores_tile - m_tile.unsqueeze(-1)).sum(-1))
        m_max = torch.maximum(lse_run, lse_new)
        o_run = o_run * torch.exp(lse_run - m_max).unsqueeze(-1) +\
                torch.einsum('bnqk,bnkd->bnqd',
                             torch.exp(scores_tile - m_max.unsqueeze(-1)),
                             kv_tile.float())
        lse_run = m_max + torch.log(torch.exp(lse_run - m_max) + torch.exp(lse_new - m_max))
        del scores_tile, m_tile, lse_new, m_max, kv_tile
    return o_run, lse_run

def main():
    print(f"DS4 CSA Attention Benchmark  bs={BSZ} sq={SEQLEN} nh={N_HEADS} hd={HEAD_DIM}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    ext = build_csa_attn()
    ext.set_num_threads(48)

    q_gpu_tmp = torch.randn(BSZ, SEQLEN, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    q_cpu_pin = torch.empty(BSZ, SEQLEN, N_HEADS, HEAD_DIM, dtype=torch.bfloat16).pin_memory()
    def d2h():
        q_cpu_pin.copy_(q_gpu_tmp)
        torch.cuda.synchronize()
    t_d2h = cuda_sync_time(d2h)
    size_mb = q_gpu_tmp.numel() * 2 / 1e6
    print(f"D2H Q (bf16): {size_mb:.0f} MB → {t_d2h:.2f} ms  ({size_mb/t_d2h*1000:.0f} MB/s)")
    del q_gpu_tmp, q_cpu_pin
    print()

    print(f"{'nc':>6} {'sp':>7} | {'GPU_win':>8} {'GPU_cmp':>8} {'GPU_tot':>8} | {'CPU_krn':>8} | {'mixed*':>7} | {'mixed+D2H':>10} | {'speedup':>8}")
    print("-" * 100)

    n_compress_cases = [64, 128, 256, 512, 1024, 2048, 4096]

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    for nc in n_compress_cases:
        scale, q_gpu, kv_win, kv_chunk, cpu_buf, pp, stream_h2d = make_data(BSZ, nc)

        q_cpu_fp32 = torch.randn(BSZ, N_HEADS, SEQLEN, HEAD_DIM, dtype=torch.float32)

        t_win   = cuda_sync_time(lambda: gpu_window_fn(q_gpu, kv_win, kv_chunk, scale))
        t_cmp   = cuda_sync_time(lambda: gpu_compress_fn(q_gpu, cpu_buf, pp, stream_h2d, scale, nc, BSZ))
        t_full  = cuda_sync_time(lambda: (gpu_window_fn(q_gpu, kv_win, kv_chunk, scale),
                                           gpu_compress_fn(q_gpu, cpu_buf, pp, stream_h2d, scale, nc, BSZ)))
        t_cpu   = cpu_wall_time(lambda: ext.fused_csa_attn(q_cpu_fp32, cpu_buf, scale))

        def mixed_fn():
            fut = pool.submit(ext.fused_csa_attn, q_cpu_fp32, cpu_buf, scale)
            res_g = gpu_window_fn(q_gpu, kv_win, kv_chunk, scale)
            torch.cuda.synchronize()
            fut.result()
            return res_g
        t_mixed = cpu_wall_time(mixed_fn)

        t_mixed_with_d2h = t_mixed + t_d2h

        ratio = t_full / t_mixed_with_d2h if t_mixed_with_d2h > 0 else 0

        sp = nc * 4
        print(f"{nc:6d} {sp:7d} | {t_win:8.2f} {t_cmp:8.2f} {t_full:8.2f} | {t_cpu:8.2f} | {t_mixed:7.2f} | {t_mixed_with_d2h:10.2f} | {ratio:8.2f}x")

    pool.shutdown(wait=False)
    print()
    print("_")
    print("_")
    print("speedup = GPU_tot / mixed+D2H")

if __name__ == '__main__':
    main()

