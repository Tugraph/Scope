pass
import sys, os, argparse, threading
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import bench_resident_kv as rkv

rkv.BUDGET_BYTES = 2 * rkv.GiB

torch.set_num_threads(16)

BATCH_SIZES = rkv.BATCH_SIZES
KV_LENS_NOMINAL = [2048, 4096, 8192, 16384, 32768]
HYBRIDGEN_RATIO = 1.0
HYBRIDGEN_BUDGET_GIB = 2

def effective_kv(nominal_kv):
    return round(nominal_kv * HYBRIDGEN_RATIO)

def run_hybridgen_kvlen(B, nominal_kv):
    pass
    ekv = effective_kv(nominal_kv)
    bkv = B * rkv.NH
    scale = rkv.QHD ** -0.5

    n_res = rkv.n_resident_kvlen(bkv, 0, ekv)
    cpu_tokens = ekv - n_res

    q_gpu = torch.randn(bkv, rkv.GROUPS, rkv.QHD, dtype=rkv.dtype, device=rkv.DEV)
    k_cpu_n1 = rkv._numa1_alloc((bkv, cpu_tokens, rkv.QHD)) if cpu_tokens > 0 else None
    v_cpu_n1 = rkv._numa1_alloc((bkv, cpu_tokens, rkv.VHD)) if cpu_tokens > 0 else None

    def _run():
        q_g = q_gpu.view(bkv, rkv.GROUPS, rkv.QHD)
        q_cpu = q_g.cpu().contiguous() if cpu_tokens > 0 else None

        cpu_res = {}
        def _cpu_fn():
            try: os.sched_setaffinity(0, rkv._NUMA1_CPUS)
            except Exception: pass
            out_c, max_c, sum_c = rkv._local_bmm_softmax(q_cpu, k_cpu_n1, v_cpu_n1, scale)
            cpu_res.update(out=out_c, max=max_c, sum=sum_c)

        thr = None
        if cpu_tokens > 0:
            thr = threading.Thread(target=_cpu_fn, daemon=True); thr.start()

        if n_res > 0:
            oa_r, ma_r, sa_r = rkv._resident_layer(q_g, bkv, n_res, scale)
        else:
            oa_r = torch.zeros(bkv, rkv.GROUPS, rkv.VHD, device=rkv.DEV, dtype=torch.float32)
            ma_r = torch.full((bkv, rkv.GROUPS, 1), float('-inf'), device=rkv.DEV, dtype=torch.float32)
            sa_r = torch.zeros(bkv, rkv.GROUPS, 1, device=rkv.DEV, dtype=torch.float32)

        torch.cuda.synchronize(rkv.DEV)
        if cpu_tokens > 0:
            thr.join()
            max_c = cpu_res['max'].to(rkv.DEV)
            sum_c = cpu_res['sum'].to(rkv.DEV)
            out_c = cpu_res['out'].to(rkv.DEV)
            o_all, m_all, s_all = rkv._pairwise_merge(oa_r, ma_r, sa_r, out_c, max_c, sum_c)
            out = (o_all / s_all.clamp(min=1e-6)).bfloat16()
        else:
            out = (oa_r / sa_r.clamp(min=1e-6)).bfloat16()
        return out.view(B, rkv.NH * rkv.GROUPS, 1, rkv.VHD)

    ms = rkv.measure(_run)
    del q_gpu, k_cpu_n1, v_cpu_n1
    rkv._clear_gpu_buf_caches()
    return ekv, n_res, cpu_tokens, ms

def smoke_test(model_key):
    rkv.set_model(model_key)
    B, kv = 4, 4096
    ekv, n_res, cpu_tokens, ms = run_hybridgen_kvlen(B, kv)
    print(f"model={model_key} B={B} nominal_kv={kv} eff_kv={ekv} "
          f"n_res={n_res} cpu_tokens={cpu_tokens}")
    print(f"latency={ms:.3f} ms")
    assert ekv == kv, "_"
    assert n_res + cpu_tokens == ekv, "_"
    print("_")

def full_sweep(model_key, out_path=None):
    rkv.set_model(model_key)
    rows = []
    for B in BATCH_SIZES:
        for kv in KV_LENS_NOMINAL:
            ekv, n_res, cpu_tokens, ms = run_hybridgen_kvlen(B, kv)
            rows.append((B, kv, ekv, n_res, cpu_tokens, ms))
            print(f"  B={B:2d} kv={kv:6d} (eff={ekv:6d})  n_res={n_res:6d}  cpu={cpu_tokens:6d}  "
                  f"latency={ms:8.2f}ms", flush=True)

    lines = []
    lines.append(f"### {rkv.MODELS[model_key]['name']} / HybridGen-adapt "
                  f"(kvlen, budget={HYBRIDGEN_BUDGET_GIB}GiB, ratio={HYBRIDGEN_RATIO})\n")
    lines.append(f"{'BS':>4} {'KV':>7} | {'eff_KV':>7} | {'n_res':>7} {'cpu':>7} | {'latency_ms':>10}")
    lines.append("─" * 60)
    for B, kv, ekv, n_res, cpu_tokens, ms in rows:
        lines.append(f"{B:>4} {kv:>7} {ekv:>7} | {n_res:>7} {cpu_tokens:>7} | {ms:>9.2f}ms")
    table_str = "\n".join(lines)
    print("\n" + table_str)

    if out_path:
        with open(out_path, 'a') as f:
            f.write(table_str + "\n\n")
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(rkv.MODELS), default='ds2')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline.md'))
    args = parser.parse_args()

    if args.smoke:
        smoke_test(args.model)
    elif args.full:
        full_sweep(args.model, out_path=args.out)
    else:
        smoke_test(args.model)

