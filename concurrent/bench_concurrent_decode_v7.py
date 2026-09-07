pass
import sys, os, time, json, statistics, threading, copy
sys.path.insert(0, '/home/yed/software/cpuvision3/code/decode_sim_v100')
os.environ.setdefault('OMP_NUM_THREADS', '16')
import torch
torch.set_num_threads(16)
import bench_resident_kv as rkv
rkv.set_model('qwen3')

DEV = rkv.DEV
WARMUP = 5
ITERS = 128
ROUNDS = 3
CONFIGS = [
    {'tag': '8k',  'bs': 8, 'kv': 8192},
    {'tag': '16k', 'bs': 8, 'kv': 16384},
]

def build_expert_pool():
    pass
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        '/home/share/data/model/Qwen3-30B-A3B', torch_dtype=torch.bfloat16)
    experts = model.model.layers[1].mlp.experts
    import intel_extension_for_pytorch as ipex
    cpu_exp1 = ipex.optimize(copy.deepcopy(experts[1]), dtype=torch.bfloat16, auto_kernel_selection=True)
    cpu_exp2 = ipex.optimize(copy.deepcopy(experts[2]), dtype=torch.bfloat16, auto_kernel_selection=True)
    pin_state = {k: v.pin_memory() if not v.is_pinned() else v
                 for k, v in copy.deepcopy(experts[3]).state_dict().items()}
    gpu_expert = copy.deepcopy(experts[0]).to(DEV)
    gpu_od_expert = copy.deepcopy(experts[4]).to(DEV)
    hidden = model.config.hidden_size
    del model
    return {'cpu': [cpu_exp1, cpu_exp2], 'pin': pin_state,
            'gpu': gpu_expert, 'gpu_od': gpu_od_expert, 'hidden': hidden, 'pp': 0}

def load_expert_dist_scaled(target_bs):
    import json as _json
    ed = _json.load(open('/home/yed/software/cpuvision3/code/expert_dist_qwen3.json'))
    ed['steps'] = [s for s in ed['steps'] if s.get('step') != 'prefill']
    src_bs = ed.get('batch_size', 32)
    scale = target_bs / src_bs
    ed = copy.deepcopy(ed)
    for s in ed['steps']:
        for l in s.get('layers', []):
            l['hit'] = [max(0, round(x * scale)) for x in l['hit']]
            l['miss'] = [max(0, round(x * scale)) for x in l['miss']]
    return ed

def simulate_expert(pool, expert_dist, step_idx, hidden):
    pass
    steps = expert_dist['steps']
    step_rec = steps[step_idx % len(steps)]
    t_total = time.perf_counter()
    hit_t, miss_cpu_t, miss_gpu_t = 0.0, 0.0, 0.0
    for i_layer in range(48):
        layer_rec = step_rec['layers'][i_layer]
        hit_counts = layer_rec['hit']
        miss_counts = layer_rec['miss']

        t0 = time.perf_counter()
        for n_tok in hit_counts:
            if n_tok <= 0: continue
            x = torch.randn(n_tok, hidden, dtype=torch.bfloat16, device=DEV)
            pool['gpu'](x)
        torch.cuda.synchronize(DEV)
        hit_t += time.perf_counter() - t0

        miss_active = [(i, n) for i, n in enumerate(miss_counts) if n > 0]
        if not miss_active: continue
        queue = sorted([n for _, n in miss_active], reverse=False)

        gpu_od_done = [False]; cpu_done = [False]
        gpu_el = [0.0]; cpu_el = [0.0]
        lock = threading.Lock()
        def gpu_od():
            t0 = time.perf_counter()
            while True:
                with lock:
                    if not queue: break
                    n = queue.pop()
                if n <= 0: continue
                pool['gpu_od'].load_state_dict(pool['pin'])
                x = torch.randn(n, hidden, dtype=torch.bfloat16, device=DEV)
                pool['gpu_od'](x)
            torch.cuda.synchronize(DEV)
            gpu_el[0] = time.perf_counter() - t0
        def cpu_fn():
            t0 = time.perf_counter()
            while True:
                with lock:
                    if not queue: break
                    n = queue.pop(0)
                if n <= 0: continue
                x = torch.randn(n, hidden, dtype=torch.bfloat16, device='cpu')
                pool['pp'] = (pool['pp'] + 1) % 2
                pool['cpu'][pool['pp']](x)
            cpu_el[0] = time.perf_counter() - t0
        tg = threading.Thread(target=gpu_od); tc = threading.Thread(target=cpu_fn)
        tg.start(); tc.start(); tg.join(); tc.join()
        miss_gpu_t += gpu_el[0]
        miss_cpu_t += cpu_el[0]
    expert_ms = (time.perf_counter() - t_total) * 1000
    return expert_ms, hit_t*1000, miss_cpu_t*1000, miss_gpu_t*1000

def decode_step(rkv_data, pool, ed, step_idx, stream=None):
    pass
    if stream is not None:
        ctx = torch.cuda.stream(stream)
    else:
        from contextlib import nullcontext; ctx = nullcontext()
    hidden = pool['hidden']
    t0 = time.perf_counter()
    with ctx:

        t_attn = time.perf_counter()
        for i_layer in range(48):
            d = rkv_data[i_layer]
            rkv.run_kvlen_resident(d['bs'], d['kv'], d['q'], d['k_cpu'], d['v_cpu'],
                                    d['k_gpu'], d['v_gpu'], d['gs'], d['n_res'])
        torch.cuda.synchronize(DEV)
        attn_ms = (time.perf_counter() - t_attn) * 1000

        expert_ms, hit_ms, miss_cpu_ms, miss_gpu_ms = simulate_expert(pool, ed, step_idx, hidden)
    wall_ms = (time.perf_counter() - t0) * 1000
    return wall_ms, attn_ms, expert_ms, hit_ms, miss_cpu_ms, miss_gpu_ms

def prep_attention(cfg):
    pass
    bs = cfg['bs']; kv = cfg['kv']
    bkv = bs * rkv.NH

    r_star = 0.469
    gs = int(kv * r_star)
    gs = max(0, min(gs, kv))
    n_res = rkv.n_resident_kvlen(bkv, gs, kv) if hasattr(rkv, 'n_resident_kvlen') else 0

    if hasattr(rkv, 'align_gs_to_chunk'):
        gs, n_res = rkv.align_gs_to_chunk(gs, n_res)
    data = []
    for i_layer in range(48):
        q, k_cpu, v_cpu, k_gpu, v_gpu = rkv._prep_kvlen(bs, kv, gs)
        data.append({'bs': bs, 'kv': kv, 'q': q, 'k_cpu': k_cpu, 'v_cpu': v_cpu,
                     'k_gpu': k_gpu, 'v_gpu': v_gpu, 'gs': gs, 'n_res': n_res})
    return data

def measure(rkv_data, pool, ed, iters, stream=None):
    walls, attns, exps = [], [], []
    for s in range(iters):
        w, a, e, *_ = decode_step(rkv_data, pool, ed, s, stream)
        walls.append(w); attns.append(a); exps.append(e)
    def st(xs): return {'median': statistics.median(xs), 'mean': statistics.mean(xs),
                        'std': statistics.stdev(xs), 'min': min(xs), 'max': max(xs)}
    return {'wall': st(walls), 'attn': st(attns), 'expert': st(exps)}

def main():
    results = {}
    print("building expert pool...", flush=True)
    pool = build_expert_pool()
    print(f"  hidden={pool['hidden']}, experts ready", flush=True)

    s1 = torch.cuda.Stream(device=DEV); s2 = torch.cuda.Stream(device=DEV)
    ed = load_expert_dist_scaled(8)

    print("=== solo 8k ===", flush=True)
    d8 = prep_attention(CONFIGS[0])
    for s in range(WARMUP): decode_step(d8, pool, ed, s)
    solo8 = measure(d8, pool, ed, ITERS)
    results['solo_8k'] = solo8
    print(f"  wall={solo8['wall']['median']:.1f}±{solo8['wall']['std']:.1f} attn={solo8['attn']['median']:.1f} expert={solo8['expert']['median']:.1f}", flush=True)

    print("=== solo 16k ===", flush=True)
    d16 = prep_attention(CONFIGS[1])
    for s in range(WARMUP): decode_step(d16, pool, ed, s)
    solo16 = measure(d16, pool, ed, ITERS)
    results['solo_16k'] = solo16
    print(f"  wall={solo16['wall']['median']:.1f}±{solo16['wall']['std']:.1f} attn={solo16['attn']['median']:.1f} expert={solo16['expert']['median']:.1f}", flush=True)

    print("\n=== concurrent ===", flush=True)
    for s in range(WARMUP):
        decode_step(d8, pool, ed, s, s1)
        decode_step(d16, pool, ed, s, s2)

    all_c8 = {'wall':[], 'attn':[], 'expert':[]}
    all_c16 = {'wall':[], 'attn':[], 'expert':[]}
    for rnd in range(ROUNDS):
        c8w, c8a, c8e = [], [], []
        c16w, c16a, c16e = [], [], []
        for s in range(ITERS):
            r8, r16 = [None], [None]
            def t1():
                w, a, e, *_ = decode_step(d8, pool, ed, s, s1); r8[0] = (w, a, e)
            def t2():
                w, a, e, *_ = decode_step(d16, pool, ed, s, s2); r16[0] = (w, a, e)
            th1 = threading.Thread(target=t1); th2 = threading.Thread(target=t2)
            th1.start(); th2.start(); th1.join(); th2.join()
            c8w.append(r8[0][0]); c8a.append(r8[0][1]); c8e.append(r8[0][2])
            c16w.append(r16[0][0]); c16a.append(r16[0][1]); c16e.append(r16[0][2])
        all_c8['wall'] += c8w; all_c8['attn'] += c8a; all_c8['expert'] += c8e
        all_c16['wall'] += c16w; all_c16['attn'] += c16a; all_c16['expert'] += c16e
        print(f"  round {rnd+1}: 8k wall={statistics.median(c8w):.1f} attn={statistics.median(c8a):.1f} exp={statistics.median(c8e):.1f} | "
              f"16k wall={statistics.median(c16w):.1f} attn={statistics.median(c16a):.1f} exp={statistics.median(c16e):.1f}", flush=True)

    def st(xs): return {'median': statistics.median(xs), 'mean': statistics.mean(xs),
                        'std': statistics.stdev(xs), 'min': min(xs), 'max': max(xs)}
    for tag, conc, solo in [('8k', all_c8, solo8), ('16k', all_c16, solo16)]:
        result = {k: st(conc[k]) for k in ['wall','attn','expert']}
        result['slowdown'] = {k: round(result[k]['median']/solo[k]['median'], 2) for k in ['wall','attn','expert']}
        results['concurrent_'+tag] = result
        sd = result['slowdown']
        print(f"  concurrent {tag}: wall={result['wall']['median']:.1f}±{result['wall']['std']:.1f}(×{sd['wall']}) "
              f"attn={result['attn']['median']:.1f}(×{sd['attn']}) expert={result['expert']['median']:.1f}(×{sd['expert']})", flush=True)

    OUT = '/home/yed/software/cpuvision3/exp53_results/concurrent_decode_qwen3_v7.json'
    json.dump(results, open(OUT, 'w'), indent=1, default=str)
    print(f"_")

if __name__ == '__main__':
    main()

