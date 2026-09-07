pass
import os
import torch
from torch.utils.cpp_extension import load

_ext = None

def build(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    src_dir = os.path.dirname(os.path.abspath(__file__))
    _ext = load(
        name='cpu_attn_ext',
        sources=[os.path.join(src_dir, 'cpu_attn.cpp')],
        extra_cflags=[
            '-O3',
            '-march=native',
            '-mavx512f', '-mavx512bw', '-mavx512vnni',
            '-fopenmp',
            '-ffast-math',
        ],
        extra_ldflags=['-fopenmp'],
        verbose=verbose,
    )
    return _ext

if __name__ == '__main__':
    ext = build(verbose=True)
    print('Build OK:', ext)

    import torch, time
    B_kv, groups, head_dim, cpu_seq = 512, 4, 128, 3072
    scale = head_dim ** -0.5
    torch.manual_seed(42)
    q = torch.randn(B_kv, groups, head_dim)
    k = torch.randn(B_kv, cpu_seq, head_dim)
    v = torch.randn(B_kv, cpu_seq, head_dim)

    def ref_blocked(block_size=1024):
        out_acc = torch.zeros(B_kv, groups, head_dim)
        max_acc = torch.full((B_kv, groups, 1), float('-inf'))
        sum_acc = torch.zeros(B_kv, groups, 1)
        for s in range(0, cpu_seq, block_size):
            e = min(s + block_size, cpu_seq)
            kb = k[:, s:e, :]; vb = v[:, s:e, :]
            sc = torch.bmm(q, kb.transpose(-2, -1)) * scale
            mb = sc.amax(dim=-1, keepdim=True)
            eb = torch.exp(sc - mb); sb = eb.sum(dim=-1, keepdim=True)
            ob = torch.bmm(eb, vb)
            mn = torch.max(max_acc, mb)
            wo = torch.exp(max_acc - mn); wn = torch.exp(mb - mn)
            sum_acc = wo * sum_acc + wn * sb
            out_acc = wo * out_acc + wn * ob
            max_acc = mn
        out_acc = out_acc / sum_acc
        return out_acc, max_acc, sum_acc

    ref_out, ref_max, ref_sum = ref_blocked()
    cpp_out, cpp_max, cpp_sum = ext.fused_cpu_attn(q, k, v, scale)

    print(f'max_diff out: {(ref_out - cpp_out).abs().max().item():.6f}')
    print(f'max_diff max: {(ref_max - cpp_max).abs().max().item():.6f}')
    print(f'max_diff sum: {(ref_sum - cpp_sum).abs().max().item():.6f}')

    for fn, name in [(lambda: ref_blocked(), 'python_blocked'),
                     (lambda: ext.fused_cpu_attn(q, k, v, scale), 'cpp_fused')]:
        for _ in range(3): fn()
        N = 20
        t0 = time.time()
        for _ in range(N): fn()
        t1 = time.time()
        print(f'{name}: {(t1-t0)/N*1000:.2f}ms')

