pass
import sys, os, time, json, statistics, random
sys.path.insert(0, '/home/yed/software/cpuvision3/code/decode_sim_v100')
import torch
import bench_resident_kv as rkv

rkv.set_model('mixtral')
rkv.CHUNK = 2048
OLD = '/home/yed/software/cpuvision3/exp53_results/mixtral_rstar_sweep.json'
OUT = '/home/yed/software/cpuvision3/exp53_results/mixtral_rstar_revalidate.json'
BATCHES = [4, 8, 16]
KV_LENS = [2048, 4096, 8192]
ITERS = 10
WARMUP = 5
random.seed(0)

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
    old = json.load(open(OLD))
    results = {}
    print(f"_")
    print(f"_")
    print("-" * 70)
    for bs in BATCHES:
        bkv = bs * rkv.NH
        for kv in KV_LENS:
            key = f'{bs},{kv}'
            old_e = old.get(key, {})
            old_curve = {int(k): v for k, v in old_e.get('curve', {}).items() if v is not None}\
                if old_e else {}
            b_cpus = list(range(0, bkv + 1))
            random.shuffle(b_cpus)
            new_curve = {}
            inflated = 0
            for bc in b_cpus:
                try:
                    ms = measure(bs, kv, bc)
                except (RuntimeError, MemoryError) as e:
                    print(f'  [OOM bs={bs} kv={kv} b_cpu={bc}] {e}', flush=True)
                    torch.cuda.empty_cache()
                    ms = float('nan')
                new_curve[bc] = None if ms != ms else round(ms, 3)
            valid = {k: v for k, v in new_curve.items() if v is not None}
            if valid:
                b_opt = min(valid, key=valid.get)
                r_star = b_opt / bkv

                for bc in valid:
                    o = old_curve.get(bc)
                    if o and valid[bc] / o < 0.8:
                        inflated += 1
                old_r = old_e.get('r_star') if old_e else None
                old_ms = old_e.get('ms_opt') if old_e else None
                results[key] = {'bkv': bkv, 'b_cpu_opt': b_opt, 'r_star': round(r_star, 3),
                                'ms_opt': valid[b_opt], 'curve': new_curve, 'inflated': inflated}
                print(f"{bs:>3} {kv:>6} {bkv:>4} | {b_opt:>6} {r_star:>6.3f} {valid[b_opt]:>7.2f}ms | "
                      f"{old_r if old_r is not None else '-':>5} {old_ms if old_ms is not None else '-':>5} {inflated}/{len(valid)}", flush=True)
            else:
                results[key] = {'bkv': bkv, 'b_cpu_opt': None, 'curve': new_curve}
                print(f"_", flush=True)
            json.dump(results, open(OUT, 'w'), indent=1)
    print(f"_")

if __name__ == '__main__':
    main()

