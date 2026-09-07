pass
import sys, os, time, json, statistics, random
sys.path.insert(0, '/home/yed/software/cpuvision3/code/decode_sim_v100')
import torch
import bench_resident_kv as rkv

rkv.set_model('mixtral')
rkv.CHUNK = 2048
OUT = '/home/yed/software/cpuvision3/exp53_results/mixtral_rstar_bigkv_corrected.json'
KV_LENS = [16384, 32768]
BATCHES = [4, 8, 16]
ITERS = 10
WARMUP = 5
random.seed(0)

def b_cpu_range(bkv):
    pass
    lo = max(1, int(round(0.60 * bkv)))
    hi = min(bkv - 1, int(round(0.80 * bkv)))
    return list(range(lo, hi + 1))

def measure(bs, kv, b_cpu):
    bkv = bs * rkv.NH
    q, k_cpu, v_cpu, k_gpu, v_gpu = rkv._prep_bkv(bs, kv, b_cpu)
    def one():
        rkv.run_bkv_resident(bs, kv, q, k_cpu, v_cpu, k_gpu, v_gpu, b_cpu, bkv - b_cpu, 0)
    for _ in range(WARMUP): one()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); one(); ts.append((time.perf_counter() - t0) * 1000)
    del q, k_cpu, v_cpu, k_gpu, v_gpu
    torch.cuda.empty_cache()
    return statistics.median(ts)

def main():
    results = {}
    print(f"_")
    print(f"_")
    for bs in BATCHES:
        bkv = bs * rkv.NH
        for kv in KV_LENS:
            key = f'{bs},{kv}'
            scan = [0] + b_cpu_range(bkv)
            random.shuffle(scan)
            curve = {}
            print(f"_", flush=True)
            for bc in scan:
                try:
                    ms = measure(bs, kv, bc)
                    curve[bc] = round(ms, 3)
                    print(f"  b_cpu={bc:>3} r*={bc/bkv:.3f}: {ms:.3f}ms", flush=True)
                except (RuntimeError, MemoryError) as e:
                    print(f'  [OOM b_cpu={bc}] {e}', flush=True)
                    torch.cuda.empty_cache()
                    curve[bc] = 'OOM'
            inner = {bc: ms for bc, ms in curve.items() if bc > 0 and ms != 'OOM'}
            opt = min(inner, key=inner.get) if inner else 0
            results[key] = {
                'bkv': bkv, 'curve': curve, 'gpu0_ms': curve.get(0),
                'opt_b_cpu': opt, 'opt_r_star': round(opt / bkv, 3) if opt else None,
                'opt_ms': curve.get(opt) if opt else None,
            }
            json.dump(results, open(OUT, 'w'), indent=1)
            opt_ms_v = curve.get(opt) if opt else None
            opt_r_v = opt/bkv if opt else 0
            print(f"_", flush=True)
    print("_")
    print(f"{'BS':>4} | {'16k':>10} {'32k':>10}")
    for bs in BATCHES:
        row = []
        for kv in KV_LENS:
            e = results[f'{bs},{kv}']
            if e['opt_r_star'] is not None:
                row.append(f"{e['opt_r_star']:.3f}({e['opt_ms']:.1f}ms)")
            else:
                row.append("OOM")
        print(f"{bs:>4} | {row[0]:>10} {row[1]:>10}")
    print(f"_")

if __name__ == '__main__':
    main()

