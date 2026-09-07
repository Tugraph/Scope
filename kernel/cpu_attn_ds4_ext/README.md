# DS4 CSA CPU Attention Kernel

## 状态：Stage 1 完成 ✓

已实现优化的 CPU attention kernel，用于 DS4 CSA 层的 prefill chunk 计算。

## 文件

- `csa_attn.cpp`: AVX512 优化的 C++ kernel
- `build.py`: torch.utils.cpp_extension 编译脚本
- `test_kernel.py`: 正确性 + 性能测试

## 验证结果

### 正确性
- ✓ 与 PyTorch einsum 参考实现误差 < 1e-6
- 测试配置：bs=2, n_heads=4, seqlen=16, n_compress=64, head_dim=512

### 性能（48核 OpenMP, AVX512）

```
 bs  nh   sq    nc |  time(ms)   GFLOPS
------------------------------------------------------------
  1  64  256    64 |      8.58    250.3
  8  64  256    64 |     63.00    272.7
 16  64  256    64 |    121.20    283.5
 32  64  256    64 |    239.81    286.6
 16  64  256   512 |    548.44    501.2   ← seq=2k CSA KV
 16  64  256  2048 |   2150.96    511.2   ← seq=8k
```

## 核心优化

1. **AVX512 BF16 widening**: load 16 bf16 → widen to fp32，内存带宽减半
2. **Online softmax**: scores 不落内存，寄存器内累积 exp 和 weighted sum
3. **3D 并行**: bs × n_heads × seqlen 独立任务，低 bs 时仍有高并行度
   - bs=16, nh=64, sq=256 → 262k 任务，48核充分利用
4. **Register blocking**: 每次处理 2 KV token，Q 常驻寄存器
5. **Prefetch**: 提前 4 token 预取 KV，隐藏 DRAM 延迟

## API

```cpp
std::tuple<torch::Tensor, torch::Tensor> fused_csa_attn(
    torch::Tensor q,    // [bs, n_heads, seqlen, head_dim] fp32
    torch::Tensor kv,   // [bs, n_compress, head_dim] bf16
    float scale         // 1/sqrt(head_dim)
);
// 返回: (o_cpu, lse_cpu)
//   o_cpu:   [bs, n_heads, seqlen, head_dim] fp32
//   lse_cpu: [bs, n_heads, seqlen] fp32 (用于与 GPU LSE merge)
```

## 下一步（Stage 2）

集成到 ds4offload.py，实现 CPU-GPU 并行 CSA attention：

1. 修改 `_patched_attn_forward` CSA streaming 路径
2. 引入 CPU 并行计算：部分 compress KV 在 CPU 处理
3. GPU-CPU LSE merge（类似 qwenoffload.py 的 mixed attention）
4. 三线程并行：GPU window + GPU CSA partial + CPU CSA partial
5. Benchmark 不同 CPU/GPU ratio 配置

参考设计：
- `/home/yed/software/cpuvision3/perf_log.md` Round Q-8 到 Q-10
- qwenoffload.py 的 ping-pong + LSE merge 机制
