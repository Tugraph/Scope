pass
import sys, os, time, json, statistics
sys.path.insert(0, '/home/yed/software/cpuvision3/code/decode_sim_v100')
import torch
import bench_resident_kv as rkv

rkv.set_model('qwen3')
rkv.CHUNK = 2048
OUT = '/home/yed/software/cpuvision3/exp53_results/qwen3_r0219_vs_puregpu.json'
BS = 8
B_CPU = 7
KV_LENS = [2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]
ITERS = 10
WARMUP = 3

def measure(bs, kv, b_cpu):
    bkv = bs * rkv.NH
    q, k_cpu, v_cpu, k_gpu, v_gpu = rkv._prep_bkv(bs, kv, b_cpu)

    def one():
        rkv.run_bkv_resident(bs, kv, q, k_cpu, v_cpu, k_gpu, v_gpu,
                             b_cpu, bkv - b_cpu, 0)
    for _ in range(WARMUP): one()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); one(); times.append((time.perf_counter() - t0) * 1000)
    del q, k_cpu, v_cpu, k_gpu, v_gpu
    torch.cuda.empty_cache()
    return statistics.median(times)

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = {}
    print(f"Qwen3 BS={BS} bkv={BS*rkv.NH} CHUNK={rkv.CHUNK} ITERS={ITERS}")
    print(f"_")
    print("-" * 52)
    for kv in KV_LENS:
        ms_r = measure(BS, kv, B_CPU)
        ms_g = measure(BS, kv, 0)
        ratio = ms_r / ms_g if ms_g > 0 else 0
        results[str(kv)] = {'r0219_ms': round(ms_r, 3), 'puregpu_ms': round(ms_g, 3),
                            'degradation': round(ratio, 3)}
        json.dump(results, open(OUT, 'w'), indent=1)
        print(f"{kv:>6} | {ms_r:>12.3f}ms | {ms_g:>12.3f}ms | {ratio:>6.2f}x", flush=True)
    print(f"_")

if __name__ == '__main__':
    main()

