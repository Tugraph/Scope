pass
import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_attn_overhead_v2 as bv2
import bench_resident_kv as rkv

BATCH_SIZES = [4, 8, 16, 32]
KV_LENS = [2048, 4096, 8192, 16384, 32768]
BASELINES = ["ours-bkv", "ours-kvlen", "solid", "scout", "hybridgen", "fastdecode"]
PREFILL_TOKENS = 64
GPU_MEM_GB = 28
WARMUP = 5
ITERS = 5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="/home/share/data/model/Qwen3-30B-A3B")
    ap.add_argument("--model_key", type=str, default=None,
                    help="_"
                         "_")
    ap.add_argument("--out", type=str, default=None,
                    help="_"
                         "sweep_baseline_compare_{model_key}[_no_overlap].json")
    ap.add_argument("--no_overlap", action="store_true",
                    help="_"
                         "_"
                         "_")
    args = ap.parse_args()

    if args.model_key is None:
        model_path_lower = args.model.lower()
        if "qwen" in model_path_lower:
            args.model_key = "qwen3"
        elif "deepseek" in model_path_lower or "ds2" in model_path_lower:
            args.model_key = "ds2"
        elif "mixtral" in model_path_lower:
            args.model_key = "mixtral"
        else:
            raise ValueError(f"_")
        print(f"_")

    if args.out is None:
        suffix = "_no_overlap" if args.no_overlap else ""
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"sweep_baseline_compare_{args.model_key}{suffix}.json")

    rkv.set_model(args.model_key)

    print(f"_")
    expert_dist = bv2.load_expert_dist(args.model_key)

    mixtral_style = (args.model_key == "mixtral")

    for bl in BASELINES:
        bv2.BASELINE_CONFIG[bl]["model_key"] = args.model_key

    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        print(f"_")
    done = {(r["batch_size"], r["target_kv_len"], r["baseline"]) for r in results}

    for bs in BATCH_SIZES:
        print(f"_", flush=True)
        ns = argparse.Namespace(model=args.model, batch_size=bs, gpu_mem_gb=GPU_MEM_GB)
        model = bv2.build_model(ns)

        for kv_len in KV_LENS:
            remaining_baselines = [bl for bl in BASELINES if (bs, kv_len, bl) not in done]
            if not remaining_baselines:
                print(f"_")
                continue

            for baseline in remaining_baselines:
                if bv2.BASELINE_CONFIG.get(baseline, {}).get("arch") == "twolayer":

                    torch.set_num_threads(16)

                print(f"_", flush=True)
                t_setup = time.time()
                bv2.setup_kv(model, PREFILL_TOKENS, kv_len)
                print(f"_", flush=True)

                prep_mode = bv2.BASELINE_TO_PREP_MODE.get(baseline)
                if prep_mode is not None:
                    print(f"_", flush=True)
                    t_prep = time.time()
                    baseline_cfg = bv2.BASELINE_CONFIG[baseline]
                    bv2.preprocess_kv_for_decode(model, baseline_cfg["ratio"],
                                                baseline_cfg["budget_gib"],
                                                baseline_cfg["model_key"], prep_mode)
                    print(f"_", flush=True)

                print(f"  bs={bs} kv={kv_len} baseline={baseline} ...", flush=True)
                t0 = time.time()
                try:
                    result = bv2.run_single(
                        model, bs, kv_len, PREFILL_TOKENS, "bkv", baseline,
                        expert_dist, mixtral_style, WARMUP, ITERS, setup=False,
                        no_overlap=args.no_overlap,
                    )
                except Exception as e:
                    print(f"_", flush=True)
                    result = {"batch_size": bs, "target_kv_len": kv_len, "baseline": baseline,
                              "error": str(e)}
                finally:
                    if prep_mode is not None:
                        bv2.free_preprocessed_kv(model)
                result["wall_s"] = time.time() - t0
                results.append(result)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                if "error" not in result:
                    print(f"_"
                          f"attn={result['attn_ms']:.2f}ms "
                          f"expert={result['expert_ms']:.2f}ms "
                          f"_", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"_")

if __name__ == "__main__":
    main()

