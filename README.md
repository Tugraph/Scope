# WriteScope — Artifact

Artifact for the paper **"WriteScope: Physically Partitioned KVCache Format for CPU-GPU Load Balancing in MoE Inference"**.

WriteScope is a KV cache abstraction that encodes physical device affinity into the cache format itself, with a roofline-guided scheduler that steers the CPU/GPU share as the sequence grows.

## Directory Layout

```
artifact/
├── kernel/        CPU attention kernels (GQA, MLA-compressed, DeepSeek-V4 CSA) + build scripts
├── system/        WriteScope runtime on the three evaluated MoE models
├── baselines/     Adapted attention baselines (SolidAttention / ScoutAttention / HybridGen / FastDecode)
├── benchmarks/    End-to-end prefill and decode sweeps against the baselines
├── roofline/      r*(L) roofline calibration and validation micro-benchmarks
└── concurrent/    Two-request concurrent decode interference benchmark
```

## Requirements

- GPU: NVIDIA A100 40GB PCIe (primary), V100 32GB PCIe (portability study)
- CPU: Intel Xeon Gold 5318Y (24 cores, AVX-512), 502 GB DRAM
- OS: Linux (RHEL / AlmaLinux 9)
- Python 3.11, PyTorch >= 2.6 (cu124), transformers >= 4.40
- Intel Extension for PyTorch (IPEX), libnuma
- Model weights (not included, downloaded separately): DeepSeek-V2-Lite, Qwen3-30B-A3B, Mixtral-8x7B, DeepSeek-V4-Flash-INT8

## Build

```bash
cd kernel
python3.11 build.py                    # GQA / MLA-compressed kernels
python3.11 build_tile.py               # MLA tile variant
python3.11 cpu_attn_ds4_ext/build.py   # DeepSeek-V4 CSA kernel
```

## Run

All experiments assume the working directory is the repository root (`artifact/`).

| Paper section | Artifact entry |
|---|---|
| §5.3 roofline calibration (`r*(L)`) | `python3.11 roofline/bench_rstar_revalidate_{ds2,qwen3,mixtral}.py` |
| §5.3 large-KV convergence points | `python3.11 roofline/bench_{ds2,qwen3,mixtral}_rstar_bigkv.py` |
| §7.2 prefill throughput | `python3.11 benchmarks/sweep_baseline_compare.py --model qwen3` |
| §7.3 end-to-end decode | `python3.11 benchmarks/sweep_baseline_compare.py --model {ds2,qwen3,mixtral}` |
| §7.4 ablation | baseline variants inside `benchmarks/sweep_baseline_compare.py` |
| §7.5 V100 portability | `python3.11 roofline/bench_resident_kv_v100.py` |
| §7.6 DeepSeek-V4 case study | `system/ds4offload.py` + `kernel/cpu_attn_ds4_ext/bench_csa.py` |
| §7.8 concurrent decode | `python3.11 concurrent/bench_concurrent_decode_v7.py` |

Each script prints its sweep table to stdout; results can be cross-checked against the corresponding figure or table in the paper.

## Notes

- Model paths are set as constants at the top of each entry script; adjust them to your local weight directory.
- The first run of each entry compiles the CPU kernels (cached afterwards in the PyTorch extension cache).
- `system/ds4offload_gpuonly.py` is the GPU-only ablation variant used in the case study.

## License

This artifact is provided solely for Artifact Evaluation by the program committee. It may not be redistributed, modified, or used to build derivative works without the authors' written permission. All rights reserved.
