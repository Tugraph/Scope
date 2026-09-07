pass
import sys, os, time, json, statistics, random
sys.path.insert(0, '/home/yed/software/cpuvision3/code/decode_sim_v100')
import torch
import bench_resident_kv as rkv

rkv.set_model('qwen3')
rkv.CHUNK = 2048
OLD = '/home/yed/software/cpuvision3/exp53_results/qwen3_rstar_sweep.json'
OUT = '/home/yed/software/cpuvision3/exp53_results/qwen3_rstar_revalidate.json'
BS = 8
KV_LENS = [2048, 4096, 8192]
ITERS = 10
WARMUP = 3
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
    print("-" * 55)
    for kv in KV_LENS:
        bkv = BS * rkv.NH
        old_curve = old[f'{BS},{kv}']['curve']
        b_cpus = list(range(0, bkv + 1))
        random.shuffle(b_cpus)
        new_curve = {}
        for bc in b_cpus:
            ms = measure(BS, kv, bc)
            new_curve[bc] = round(ms, 3)

        inflated = 0
        for bc in range(0, bkv + 1):
            o = old_curve[str(bc)]
            n = new_curve[bc]
            r = n / o if o > 0 else 0
            flag = ""
            if r < 0.8: flag = "_"; inflated += 1
            elif r < 0.95: flag = "_"
            r_star = bc / bkv
            print(f"{kv:>5} {bc:>5} {r_star*100:>5.1f}% | {o:>8.3f} {n:>8.3f} {r:>6.2f} {flag:>8}", flush=True)
        b_opt = min(new_curve, key=new_curve.get)
        old_opt = old[f'{BS},{kv}']['b_cpu_opt']
        results[f'{BS},{kv}'] = {
            'new_curve': new_curve,
            'new_b_cpu_opt': b_opt,
            'new_r_star': round(b_opt / bkv, 3),
            'new_ms_opt': new_curve[b_opt],
            'old_b_cpu_opt': old_opt,
            'old_ms_opt': old[f'{BS},{kv}']['ms_opt'],
            'inflated_points': inflated,
        }
        json.dump(results, open(OUT, 'w'), indent=1)
        print(f"_"
              f"_", flush=True)
    print(f"_")

if __name__ == '__main__':
    main()

