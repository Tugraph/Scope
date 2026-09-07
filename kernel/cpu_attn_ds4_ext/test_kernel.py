
pass
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from build import build_csa_attn

def ref_csa_attn(q, kv, scale):
    pass
    bs, n_heads, seqlen, head_dim = q.shape
    n_compress = kv.size(1)

    scores = torch.einsum('bhsd,bcd->bhsc', q, kv.float()) * scale

    attn_weights = torch.softmax(scores, dim=-1)

    o = torch.einsum('bhsc,bcd->bhsd', attn_weights, kv.float())

    max_scores = scores.max(dim=-1, keepdim=True).values
    lse = max_scores.squeeze(-1) + torch.log(
        torch.exp(scores - max_scores).sum(dim=-1)
    )

    return o, lse

def test_correctness():
    print("_")
    torch.manual_seed(42)

    bs, n_heads, seqlen, head_dim = 2, 4, 16, 512
    n_compress = 64
    scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(bs, n_heads, seqlen, head_dim, dtype=torch.float32)
    kv = torch.randn(bs, n_compress, head_dim, dtype=torch.bfloat16)

    o_ref, lse_ref = ref_csa_attn(q, kv, scale)

    ext = build_csa_attn()
    o_cpu, lse_cpu = ext.fused_csa_attn(q, kv, scale)

    o_diff = (o_cpu - o_ref).abs().max().item()
    lse_diff = (lse_cpu - lse_ref).abs().max().item()

    print(f"q shape: {q.shape}, kv shape: {kv.shape}")
    print(f"max |o_cpu - o_ref|: {o_diff:.6f}")
    print(f"max |lse_cpu - lse_ref|: {lse_diff:.6f}")

    if o_diff < 0.01 and lse_diff < 0.01:
        print("_")
        return True
    else:
        print("_")
        print("\no_ref sample:")
        print(o_ref[0, 0, 0, :8])
        print("\no_cpu sample:")
        print(o_cpu[0, 0, 0, :8])
        print("\nlse_ref sample:")
        print(lse_ref[0, 0, :8])
        print("\nlse_cpu sample:")
        print(lse_cpu[0, 0, :8])
        return False

def bench_kernel(ext, bs, n_heads, seqlen, n_compress, head_dim=512, n_warmup=3, n_iter=10):
    pass
    scale = 1.0 / (head_dim ** 0.5)
    q = torch.randn(bs, n_heads, seqlen, head_dim, dtype=torch.float32)
    kv = torch.randn(bs, n_compress, head_dim, dtype=torch.bfloat16)

    for _ in range(n_warmup):
        o, lse = ext.fused_csa_attn(q, kv, scale)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n_iter):
        o, lse = ext.fused_csa_attn(q, kv, scale)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / n_iter * 1000

    flops = bs * n_heads * seqlen * (2 * head_dim * n_compress + 2 * n_compress * head_dim)
    gflops = flops / avg_ms / 1e6

    return avg_ms, gflops

def test_performance():
    print("_")
    ext = build_csa_attn()
    print(f"_")

    configs = [

        (1, 64, 256, 64),
        (8, 64, 256, 64),
        (16, 64, 256, 64),
        (32, 64, 256, 64),
        (16, 64, 256, 512),
        (16, 64, 256, 2048),
    ]

    print(f"\n{'bs':>3} {'nh':>3} {'sq':>4} {'nc':>5} | {'time(ms)':>9} {'GFLOPS':>8}")
    print("-" * 60)

    for bs, nh, sq, nc in configs:
        avg_ms, gflops = bench_kernel(ext, bs, nh, sq, nc)
        print(f"{bs:3d} {nh:3d} {sq:4d} {nc:5d} | {avg_ms:9.2f} {gflops:8.1f}")

if __name__ == '__main__':
    if not test_correctness():
        sys.exit(1)

    test_performance()

