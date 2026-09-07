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
    {'tag': 'bs4_8k',  'bs': 4,  'kv': 8192,  'r_star': 0.438},
    {'tag': 'bs16_16k','bs': 16, 'kv': 16384, 'r_star': 0.547},
]

def build_expert_pool():
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
    ed = json.load(open('/home/yed/software/cpuvision3/code/expert_dist_qwen3.json'))
    ed['steps'] = [s for s in ed['steps'] if s.get('step') != 'prefill']
    src_bs = ed.get('batch_size', 32)
    scale = target_bs / src_bs
    ed = copy.deepcopy(ed)
    for s in ed['steps']:
        for l in s.get('layers', []):
            l['hit'] = [max(0, round(x * scale)) for x in l['hit']]
            l['miss'] = [max(0, round(x * scale)) for x in l['miss']]
    ed['batch_size'] = target_bs
    return ed

def simulate_expert(pool, expert_dist, step_idx, hidden):
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
        miss_active = [n for n in miss_counts if n > 0]
        if not miss_active: continue
        queue = sorted(miss_active)
        gpu_el = [0.0]; cpu_el = [0.0]; lock = threading.Lock()
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
        miss_gpu_t += gpu_el[0]; miss_cpu_t += cpu_el[0]
    expert_ms = (time.perf_counter() - t_total) * 1000
    return expert_ms, hit_t*1000, miss_cpu_t*1000, miss_gpu_t*1000

def decode_step(rkv_data, pool, ed, step_idx, stream=None):
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
    return wall_ms, attn_ms, expert_ms

def prep_attention(cfg):
    bs = cfg['bs']; kv = cfg['kv']; r_star = cfg['r_star']
    bkv = bs * rkv.NH
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

def st(xs): return {'median': statistics.median(xs), 'mean': statistics.mean(xs),
                    'std': statistics.stdev(xs), 'min': min(xs), 'max': max(xs)}

def measure(rkv_data, pool, ed, iters, stream=None):
    walls, attns, exps = [], [], []
    for s in range(iters):
        w, a, e = decode_step(rkv_data, pool, ed, s, stream)
        walls.append(w); attns.append(a); exps.append(e)
    return {'wall': st(walls), 'attn': st(attns), 'expert': st(exps)}

def main():
    results = {}
    print("building expert pool...", flush=True)
    pool = build_expert_pool()
    print(f"  hidden={pool['hidden']}, experts ready", flush=True)

    s1 = torch.cuda.Stream(device=DEV); s2 = torch.cuda.Stream(device=DEV)
    ed1 = load_expert_dist_scaled(CONFIGS[0]['bs'])
    ed2 = load_expert_dist_scaled(CONFIGS[1]['bs'])

    for cfg in CONFIGS:
        tag = cfg['tag']
        print(f"=== solo {tag} (BS={cfg['bs']}, kv={cfg['kv']}, r*={cfg['r_star']}) ===", flush=True)
        d = prep_attention(cfg)
        ed = ed1 if cfg['bs'] == 4 else ed2
        for s in range(WARMUP): decode_step(d, pool, ed, s)
        solo = measure(d, pool, ed, ITERS)
        results['solo_'+tag] = solo
        print(f"  wall={solo['wall']['median']:.1f}±{solo['wall']['std']:.1f} attn={solo['attn']['median']:.1f} expert={solo['expert']['median']:.1f}", flush=True)

    print("\n=== concurrent ===", flush=True)
    d1 = prep_attention(CONFIGS[0]); d2 = prep_attention(CONFIGS[1])
    for s in range(WARMUP):
        decode_step(d1, pool, ed1, s, s1)
        decode_step(d2, pool, ed2, s, s2)
    all_c1 = {'wall':[], 'attn':[], 'expert':[]}
    all_c2 = {'wall':[], 'attn':[], 'expert':[]}
    for rnd in range(ROUNDS):
        c1w, c1a, c1e = [], [], []
        c2w, c2a, c2e = [], [], []
        for s in range(ITERS):
            r1, r2 = [None], [None]
            def t1():
                w, a, e = decode_step(d1, pool, ed1, s, s1); r1[0] = (w, a, e)
            def t2():
                w, a, e = decode_step(d2, pool, ed2, s, s2); r2[0] = (w, a, e)
            th1 = threading.Thread(target=t1); th2 = threading.Thread(target=t2)
            th1.start(); th2.start(); th1.join(); th2.join()
            c1w.append(r1[0][0]); c1a.append(r1[0][1]); c1e.append(r1[0][2])
            c2w.append(r2[0][0]); c2a.append(r2[0][1]); c2e.append(r2[0][2])
        all_c1['wall'] += c1w; all_c1['attn'] += c1a; all_c1['expert'] += c1e
        all_c2['wall'] += c2w; all_c2['attn'] += c2a; all_c2['expert'] += c2e
        print(f"  round {rnd+1}: bs4_8k wall={statistics.median(c1w):.1f} attn={statistics.median(c1a):.1f} exp={statistics.median(c1e):.1f} | "
              f"bs16_16k wall={statistics.median(c2w):.1f} attn={statistics.median(c2a):.1f} exp={statistics.median(c2e):.1f}", flush=True)

    for tag, conc, solo in [(CONFIGS[0]['tag'], all_c1, results['solo_'+CONFIGS[0]['tag']]),
                            (CONFIGS[1]['tag'], all_c2, results['solo_'+CONFIGS[1]['tag']])]:
        result = {k: st(conc[k]) for k in ['wall','attn','expert']}
        result['slowdown'] = {k: round(result[k]['median']/solo[k]['median'], 2) for k in ['wall','attn','expert']}
        results['concurrent_'+tag] = result
        sd = result['slowdown']
        print(f"  concurrent {tag}: wall={result['wall']['median']:.1f}±{result['wall']['std']:.1f}(×{sd['wall']}) "
              f"attn={result['attn']['median']:.1f}(×{sd['attn']}) expert={result['expert']['median']:.1f}(×{sd['expert']})", flush=True)

    OUT = '/home/yed/software/cpuvision3/exp53_results/concurrent_decode_qwen3_bs4_bs16.json'
    json.dump(results, open(OUT, 'w'), indent=1, default=str)
    print(f"_")

if __name__ == '__main__':
    main()

