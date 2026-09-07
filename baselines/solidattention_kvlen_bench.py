pass
import sys, os, argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import bench_resident_kv as rkv

rkv.BUDGET_BYTES = 2 * rkv.GiB

BATCH_SIZES = rkv.BATCH_SIZES
KV_LENS_NOMINAL = [2048, 4096, 8192, 16384, 32768]
SOLID_RATIO = 0.8
SOLID_BUDGET_GIB = 2

def effective_kv(nominal_kv):
    return round(nominal_kv * SOLID_RATIO)

def _prep_solid(B, ekv, gs_final):
    pass
    bkv = B * rkv.NH
    q_gpu = torch.randn(bkv, rkv.GROUPS, rkv.QHD, dtype=rkv.dtype, device=rkv.DEV)
    k_pin = rkv._prep_pingpong_chunks(bkv, gs_final, rkv.QHD)
    v_pin = rkv._prep_pingpong_chunks(bkv, gs_final, rkv.VHD)
    return q_gpu, k_pin, v_pin

def run_solid_kvlen(B, nominal_kv):
    pass
    ekv = effective_kv(nominal_kv)
    bkv = B * rkv.NH

    n_res = rkv.n_resident_kvlen(bkv, 0, ekv)
    gs_final = ekv - n_res
    gs_final, n_res = rkv.align_gs_to_chunk(gs_final, n_res)

    q_gpu, k_pin, v_pin = _prep_solid(B, ekv, gs_final)
    ms = rkv.measure(lambda: rkv.run_kvlen_resident(
        B, ekv, q_gpu, None, None, k_pin, v_pin, gs_final, n_res))
    del q_gpu, k_pin, v_pin
    rkv._clear_gpu_buf_caches()
    return ekv, n_res, gs_final, ms

def smoke_test(model_key):
    rkv.set_model(model_key)
    B, kv = 4, 4096
    ekv, n_res, gs_final, ms = run_solid_kvlen(B, kv)
    print(f"model={model_key} B={B} nominal_kv={kv} eff_kv={ekv} "
          f"n_res={n_res} gs={gs_final} cpu_tokens={ekv - gs_final - n_res}")
    print(f"latency={ms:.3f} ms")
    assert ekv - gs_final - n_res == 0, "_"
    print("_")

def full_sweep(model_key, out_path=None):
    rkv.set_model(model_key)
    rows = []
    for B in BATCH_SIZES:
        for kv in KV_LENS_NOMINAL:
            ekv, n_res, gs_final, ms = run_solid_kvlen(B, kv)
            rows.append((B, kv, ekv, n_res, gs_final, ms))
            print(f"  B={B:2d} kv={kv:6d} (eff={ekv:6d})  n_res={n_res:6d}  gs={gs_final:6d}  "
                  f"latency={ms:8.2f}ms", flush=True)

    lines = []
    lines.append(f"### {rkv.MODELS[model_key]['name']} / SolidAttention-adapt "
                  f"(kvlen, budget={SOLID_BUDGET_GIB}GiB, ratio={SOLID_RATIO})\n")
    lines.append(f"{'BS':>4} {'KV':>7} {'eff_KV':>7} | {'n_res':>7} {'gs':>7} | {'latency_ms':>10}")
    lines.append("─" * 60)
    for B, kv, ekv, n_res, gs_final, ms in rows:
        lines.append(f"{B:>4} {kv:>7} {ekv:>7} | {n_res:>7} {gs_final:>7} | {ms:>9.2f}ms")
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

