
#include <torch/extension.h>
#include <immintrin.h>
#include <cmath>
#include <limits>
#include <tuple>
#include <cstdint>
#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

struct ThreadPool {
    int n_threads;
    std::vector<std::thread> workers;

    std::mutex mtx;
    std::condition_variable cv_work, cv_done;

    std::function<void(int)> task_fn;
    std::atomic<int> task_gen{0};
    std::atomic<int> done_count{0};
    int expected_done = 0;
    bool shutdown = false;

    explicit ThreadPool(int n) : n_threads(n), workers(n) {
        for (int i = 0; i < n; i++) {
            workers[i] = std::thread([this, i]() {
                int last_gen = 0;
                while (true) {
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv_work.wait(lk, [&]{ return shutdown || task_gen.load() != last_gen; });
                    }
                    if (shutdown) break;
                    last_gen = task_gen.load();
                    task_fn(i);
                    if (done_count.fetch_add(1) + 1 == expected_done)
                        cv_done.notify_one();
                }
            });
        }
    }

    void run(std::function<void(int)> fn) {
        {
            std::unique_lock<std::mutex> lk(mtx);
            task_fn = std::move(fn);
            expected_done = n_threads;
            done_count.store(0);
            task_gen.fetch_add(1);
        }
        cv_work.notify_all();
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_done.wait(lk, [&]{ return done_count.load() == expected_done; });
        }
    }

    ~ThreadPool() {
        { std::unique_lock<std::mutex> lk(mtx); shutdown = true; }
        cv_work.notify_all();
        for (auto& w : workers) w.join();
    }
};

static ThreadPool* g_pool = nullptr;
static int g_pool_threads = 0;

static ThreadPool& get_pool(int n_threads) {
    if (g_pool == nullptr || g_pool_threads != n_threads) {
        delete g_pool;
        g_pool = new ThreadPool(n_threads);
        g_pool_threads = n_threads;
    }
    return *g_pool;
}

static inline float fast_expf(float x) {

    if (x < -87.3f) return 0.f;

    const float inv_ln2 = 1.4426950408889634f;
    const float ln2     = 0.6931471805599453f;
    int n   = (int)(x * inv_ln2);
    float r = x - n * ln2;

    float p = 1.f/120.f;
    p = p * r + 1.f/24.f;
    p = p * r + 1.f/6.f;
    p = p * r + 0.5f;
    p = p * r + 1.f;
    p = p * r + 1.f;

    union { float f; int32_t i; } u;
    u.i = (n + 127) << 23;
    return p * u.f;
}

static inline __m512 fast_exp_ps(__m512 x) {
    const __m512 inv_ln2 = _mm512_set1_ps(1.4426950408889634f);
    const __m512 ln2     = _mm512_set1_ps(0.6931471805599453f);

    __m512i n = _mm512_cvtps_epi32(_mm512_roundscale_ps(
        _mm512_mul_ps(x, inv_ln2), 0x01 ));
    __m512 fn = _mm512_cvtepi32_ps(n);
    __m512 r  = _mm512_fnmadd_ps(fn, ln2, x);

    const __m512 c5 = _mm512_set1_ps(1.f/120.f), c4 = _mm512_set1_ps(1.f/24.f);
    const __m512 c3 = _mm512_set1_ps(1.f/6.f),   c2 = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.f);
    __m512 p = _mm512_fmadd_ps(c5, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, one);
    p = _mm512_fmadd_ps(p, r, one);

    __m512i exp2n = _mm512_slli_epi32(_mm512_add_epi32(n, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(exp2n));
}

static inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

static inline float dot_fp32_bf16(const float* __restrict__ q_fp32,
                                   const uint16_t* __restrict__ k_bf16,
                                   int head_dim) {
    __m512 acc = _mm512_setzero_ps();
    for (int d = 0; d < head_dim; d += 16) {
        __m256i k16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_bf16 + d));
        __m512i k32i = _mm512_cvtepu16_epi32(k16);
        k32i = _mm512_slli_epi32(k32i, 16);
        __m512 kv = _mm512_castsi512_ps(k32i);
        __m512 qv = _mm512_loadu_ps(q_fp32 + d);
        acc = _mm512_fmadd_ps(qv, kv, acc);
    }
    return hsum_avx512(acc);
}

static inline void update_bf16(float* __restrict__ out,
                                const uint16_t* __restrict__ v_bf16,
                                float exp_old, float exp_new,
                                int head_dim) {
    __m512 eo = _mm512_set1_ps(exp_old);
    __m512 en = _mm512_set1_ps(exp_new);
    for (int d = 0; d < head_dim; d += 16) {
        __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v_bf16 + d));
        __m512i v32i = _mm512_cvtepu16_epi32(v16);
        v32i = _mm512_slli_epi32(v32i, 16);
        __m512 vv = _mm512_castsi512_ps(v32i);
        __m512 o = _mm512_loadu_ps(out + d);
        _mm512_storeu_ps(out + d, _mm512_fmadd_ps(eo, o, _mm512_mul_ps(en, vv)));
    }
}

static inline void assign_bf16(float* __restrict__ out,
                                const uint16_t* __restrict__ v_bf16,
                                float en,
                                int head_dim) {
    __m512 ev = _mm512_set1_ps(en);
    for (int d = 0; d < head_dim; d += 16) {
        __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v_bf16 + d));
        __m512i v32i = _mm512_cvtepu16_epi32(v16);
        v32i = _mm512_slli_epi32(v32i, 16);
        _mm512_storeu_ps(out + d, _mm512_mul_ps(ev, _mm512_castsi512_ps(v32i)));
    }
}

static inline void normalize_fp32(float* __restrict__ out, float sum, int head_dim) {
    __m512 inv = _mm512_set1_ps(1.0f / sum);
    for (int d = 0; d < head_dim; d += 16) {
        __m512 o = _mm512_loadu_ps(out + d);
        _mm512_storeu_ps(out + d, _mm512_mul_ps(o, inv));
    }
}

static inline void bf16_row_to_fp32(float* dst, const uint16_t* src, int head_dim) {
    for (int d = 0; d < head_dim; d += 16) {
        __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + d));
        __m512i v32i = _mm512_cvtepu16_epi32(v16);
        v32i = _mm512_slli_epi32(v32i, 16);
        _mm512_storeu_ps(dst + d, _mm512_castsi512_ps(v32i));
    }
}

static constexpr int PREFETCH_DIST = 4;

static inline void prefetch_kv_bf16(const uint16_t* kv, int head_dim) {
    for (int d = 0; d < head_dim * 2; d += 64)
        _mm_prefetch(reinterpret_cast<const char*>(kv) + d, _MM_HINT_T1);
}

std::tuple<torch::Tensor, torch::Tensor> fused_csa_attn(
        torch::Tensor q,
        torch::Tensor kv,
        float scale) {

    TORCH_CHECK(q.dim() == 4, "q must be 4D");
    TORCH_CHECK(kv.dim() == 3, "kv must be 3D");
    TORCH_CHECK(q.scalar_type() == at::kFloat, "q must be fp32");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bf16");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");

    const int bs = q.size(0);
    const int n_heads = q.size(1);
    const int seqlen = q.size(2);
    const int head_dim = q.size(3);
    const int n_compress = kv.size(1);

    TORCH_CHECK(kv.size(0) == bs, "kv batch mismatch");
    TORCH_CHECK(kv.size(2) == head_dim, "kv head_dim mismatch");
    TORCH_CHECK(head_dim % 16 == 0, "head_dim must be multiple of 16");

    auto o_cpu   = torch::empty({bs, n_heads, seqlen, head_dim}, q.options());
    auto lse_cpu = torch::empty({bs, n_heads, seqlen}, q.options());

    const float* q_ptr = q.data_ptr<float>();
    const uint16_t* kv_ptr = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float* o_ptr   = o_cpu.data_ptr<float>();
    float* lse_ptr = lse_cpu.data_ptr<float>();

    #pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < bs; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int sq = 0; sq < seqlen; sq++) {
                const float* q_bhsq = q_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                const uint16_t* kv_b = kv_ptr + b * n_compress * head_dim;
                float* o_bhsq   = o_ptr   + ((b * n_heads + h) * seqlen + sq) * head_dim;
                float* lse_bhsq = lse_ptr + (b * n_heads + h) * seqlen + sq;

                float max_val = -std::numeric_limits<float>::infinity();
                float sum_val = 0.0f;
                for (int d = 0; d < head_dim; d++) o_bhsq[d] = 0.0f;

                int i = 0;
                for (; i + 1 < n_compress; i += 2) {
                    const uint16_t* k0 = kv_b + i * head_dim;
                    const uint16_t* k1 = kv_b + (i + 1) * head_dim;

                    if (i + PREFETCH_DIST + 1 < n_compress) {
                        prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST)     * head_dim, head_dim);
                        prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST + 1) * head_dim, head_dim);
                    }

                    float s0 = dot_fp32_bf16(q_bhsq, k0, head_dim) * scale;
                    float nm0 = (s0 > max_val) ? s0 : max_val;
                    float eo0 = std::exp(max_val - nm0), en0 = std::exp(s0 - nm0);
                    sum_val = eo0 * sum_val + en0; max_val = nm0;
                    update_bf16(o_bhsq, k0, eo0, en0, head_dim);

                    float s1 = dot_fp32_bf16(q_bhsq, k1, head_dim) * scale;
                    float nm1 = (s1 > max_val) ? s1 : max_val;
                    float eo1 = std::exp(max_val - nm1), en1 = std::exp(s1 - nm1);
                    sum_val = eo1 * sum_val + en1; max_val = nm1;
                    update_bf16(o_bhsq, k1, eo1, en1, head_dim);
                }
                for (; i < n_compress; i++) {
                    const uint16_t* ki = kv_b + i * head_dim;
                    float score = dot_fp32_bf16(q_bhsq, ki, head_dim) * scale;
                    float nm = (score > max_val) ? score : max_val;
                    float eo = std::exp(max_val - nm), en = std::exp(score - nm);
                    sum_val = eo * sum_val + en; max_val = nm;
                    update_bf16(o_bhsq, ki, eo, en, head_dim);
                }

                normalize_fp32(o_bhsq, sum_val, head_dim);
                *lse_bhsq = max_val + std::log(sum_val);
            }
        }
    }

    return std::make_tuple(o_cpu, lse_cpu);
}

std::tuple<torch::Tensor, torch::Tensor> fused_csa_attn_pool(
        torch::Tensor q,
        torch::Tensor kv,
        float scale,
        int n_threads = 0) {

    TORCH_CHECK(q.dim() == 4, "q must be 4D");
    TORCH_CHECK(kv.dim() == 3, "kv must be 3D");
    TORCH_CHECK(q.scalar_type() == at::kFloat, "q must be fp32");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bf16");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "must be contiguous");

    const int bs = q.size(0);
    const int n_heads = q.size(1);
    const int seqlen = q.size(2);
    const int head_dim = q.size(3);
    const int n_compress = kv.size(1);
    TORCH_CHECK(kv.size(0) == bs && kv.size(2) == head_dim, "kv shape mismatch");
    TORCH_CHECK(head_dim % 16 == 0, "head_dim must be multiple of 16");

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    auto o_cpu   = torch::empty({bs, n_heads, seqlen, head_dim}, q.options());
    auto lse_cpu = torch::empty({bs, n_heads, seqlen}, q.options());

    const float*    q_ptr  = q.data_ptr<float>();
    const uint16_t* kv_ptr = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float* o_ptr   = o_cpu.data_ptr<float>();
    float* lse_ptr = lse_cpu.data_ptr<float>();

    const int total_tasks = bs * n_heads * seqlen;

    ThreadPool& pool = get_pool(n_threads);
    pool.run([&](int tid) {

        int chunk = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        for (int task = t_start; task < t_end; task++) {
            int sq = task % seqlen;
            int bh = task / seqlen;
            int h  = bh % n_heads;
            int b  = bh / n_heads;

            const float*    q_row = q_ptr  + ((b * n_heads + h) * seqlen + sq) * head_dim;
            const uint16_t* kv_b  = kv_ptr + b * n_compress * head_dim;
            float*          o_row = o_ptr  + ((b * n_heads + h) * seqlen + sq) * head_dim;
            float*       lse_row  = lse_ptr + (b * n_heads + h) * seqlen + sq;

            float max_val = -std::numeric_limits<float>::infinity();
            float sum_val = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            int i = 0;
            for (; i + 1 < n_compress; i += 2) {
                const uint16_t* k0 = kv_b + i * head_dim;
                const uint16_t* k1 = kv_b + (i + 1) * head_dim;
                if (i + PREFETCH_DIST + 1 < n_compress) {
                    prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST)     * head_dim, head_dim);
                    prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST + 1) * head_dim, head_dim);
                }
                float s0 = dot_fp32_bf16(q_row, k0, head_dim) * scale;
                float nm0 = s0 > max_val ? s0 : max_val;
                float eo0 = std::exp(max_val - nm0), en0 = std::exp(s0 - nm0);
                sum_val = eo0 * sum_val + en0; max_val = nm0;
                update_bf16(o_row, k0, eo0, en0, head_dim);

                float s1 = dot_fp32_bf16(q_row, k1, head_dim) * scale;
                float nm1 = s1 > max_val ? s1 : max_val;
                float eo1 = std::exp(max_val - nm1), en1 = std::exp(s1 - nm1);
                sum_val = eo1 * sum_val + en1; max_val = nm1;
                update_bf16(o_row, k1, eo1, en1, head_dim);
            }
            for (; i < n_compress; i++) {
                const uint16_t* ki = kv_b + i * head_dim;
                float score = dot_fp32_bf16(q_row, ki, head_dim) * scale;
                float nm = score > max_val ? score : max_val;
                float eo = std::exp(max_val - nm), en = std::exp(score - nm);
                sum_val = eo * sum_val + en; max_val = nm;
                update_bf16(o_row, ki, eo, en, head_dim);
            }
            normalize_fp32(o_row, sum_val, head_dim);
            *lse_row = max_val + std::log(sum_val);
        }
    });

    return std::make_tuple(o_cpu, lse_cpu);
}

std::tuple<torch::Tensor, torch::Tensor> fused_csa_attn_pool_bf16(
        torch::Tensor q,
        torch::Tensor kv,
        float scale,
        int n_threads = 0) {

    TORCH_CHECK(q.dim() == 4 && kv.dim() == 3, "q must be 4D, kv must be 3D");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bf16");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "must be contiguous");

    const int bs = q.size(0), n_heads = q.size(1), seqlen = q.size(2), head_dim = q.size(3);
    const int n_compress = kv.size(1);
    TORCH_CHECK(kv.size(0) == bs && kv.size(2) == head_dim, "kv shape mismatch");
    TORCH_CHECK(head_dim % 16 == 0, "head_dim must be multiple of 16");

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    auto o_cpu   = torch::empty({bs, n_heads, seqlen, head_dim},
                                torch::TensorOptions().dtype(torch::kFloat32));
    auto lse_cpu = torch::empty({bs, n_heads, seqlen},
                                torch::TensorOptions().dtype(torch::kFloat32));

    const uint16_t* q_ptr  = reinterpret_cast<const uint16_t*>(q.data_ptr<at::BFloat16>());
    const uint16_t* kv_ptr = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float* o_ptr   = o_cpu.data_ptr<float>();
    float* lse_ptr = lse_cpu.data_ptr<float>();

    const int total_tasks = bs * n_heads * seqlen;
    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        std::vector<float> q_fp32(head_dim);

        for (int task = t_start; task < t_end; task++) {
            int sq = task % seqlen;
            int bh = task / seqlen;
            int h  = bh % n_heads;
            int b  = bh / n_heads;

            const uint16_t* q_row  = q_ptr  + ((b * n_heads + h) * seqlen + sq) * head_dim;
            const uint16_t* kv_b   = kv_ptr + b * n_compress * head_dim;
            float*          o_row  = o_ptr  + ((b * n_heads + h) * seqlen + sq) * head_dim;
            float*          lse_row = lse_ptr + (b * n_heads + h) * seqlen + sq;

            bf16_row_to_fp32(q_fp32.data(), q_row, head_dim);

            float max_val = -std::numeric_limits<float>::infinity();
            float sum_val = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            int i = 0;
            for (; i + 1 < n_compress; i += 2) {
                const uint16_t* k0 = kv_b + i * head_dim;
                const uint16_t* k1 = kv_b + (i + 1) * head_dim;
                if (i + PREFETCH_DIST + 1 < n_compress) {
                    prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST)     * head_dim, head_dim);
                    prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST + 1) * head_dim, head_dim);
                }
                float s0 = dot_fp32_bf16(q_fp32.data(), k0, head_dim) * scale;
                float nm0 = s0 > max_val ? s0 : max_val;
                float eo0 = std::exp(max_val - nm0), en0 = std::exp(s0 - nm0);
                sum_val = eo0 * sum_val + en0; max_val = nm0;
                update_bf16(o_row, k0, eo0, en0, head_dim);

                float s1 = dot_fp32_bf16(q_fp32.data(), k1, head_dim) * scale;
                float nm1 = s1 > max_val ? s1 : max_val;
                float eo1 = std::exp(max_val - nm1), en1 = std::exp(s1 - nm1);
                sum_val = eo1 * sum_val + en1; max_val = nm1;
                update_bf16(o_row, k1, eo1, en1, head_dim);
            }
            for (; i < n_compress; i++) {
                const uint16_t* ki = kv_b + i * head_dim;
                float score = dot_fp32_bf16(q_fp32.data(), ki, head_dim) * scale;
                float nm = score > max_val ? score : max_val;
                float eo = std::exp(max_val - nm), en = std::exp(score - nm);
                sum_val = eo * sum_val + en; max_val = nm;
                update_bf16(o_row, ki, eo, en, head_dim);
            }
            normalize_fp32(o_row, sum_val, head_dim);
            *lse_row = max_val + std::log(sum_val);
        }
    });

    return std::make_tuple(o_cpu, lse_cpu);
}

void fused_csa_attn_bsq(
        torch::Tensor q,
        torch::Tensor kv,
        float scale,
        torch::Tensor o_out,
        torch::Tensor lse_out,
        int n_threads = 0) {

    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "q/kv must be contiguous");
    TORCH_CHECK(o_out.is_contiguous() && lse_out.is_contiguous(), "o/lse must be contiguous");

    const int bs        = q.size(0);
    const int n_heads   = q.size(1);
    const int seqlen    = q.size(2);
    const int head_dim  = q.size(3);
    const int n_compress = kv.size(1);
    TORCH_CHECK(head_dim % 16 == 0);

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    const float*    q_ptr   = q.data_ptr<float>();
    const uint16_t* kv_ptr  = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float*          o_ptr   = o_out.data_ptr<float>();
    float*          lse_ptr = lse_out.data_ptr<float>();

    const int total_tasks = bs * seqlen;
    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        std::vector<float> max_v(n_heads);
        std::vector<float> sum_v(n_heads);
        float k_fp32[512];

        for (int task = t_start; task < t_end; task++) {
            int sq = task % seqlen;
            int b  = task / seqlen;

            const uint16_t* kv_b = kv_ptr + b * n_compress * head_dim;

            constexpr int HEAD_BLOCK = 16;

            for (int h0 = 0; h0 < n_heads; h0 += HEAD_BLOCK) {
                int h1 = std::min(h0 + HEAD_BLOCK, n_heads);

                for (int h = h0; h < h1; h++) {
                    max_v[h] = -std::numeric_limits<float>::infinity();
                    sum_v[h] = 0.0f;
                    float* o_h = o_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    for (int d = 0; d < head_dim; d++) o_h[d] = 0.0f;
                }

                for (int i = 0; i < n_compress; i++) {
                    const uint16_t* ki = kv_b + i * head_dim;
                    if (i + PREFETCH_DIST < n_compress)
                        prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST) * head_dim, head_dim);
                    bf16_row_to_fp32(k_fp32, ki, head_dim);

                    for (int h = h0; h < h1; h++) {
                        const float* q_h = q_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                        float*       o_h = o_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;

                        __m512 acc = _mm512_setzero_ps();
                        for (int d = 0; d < head_dim; d += 16)
                            acc = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d),
                                                  _mm512_loadu_ps(k_fp32 + d), acc);
                        float score = hsum_avx512(acc) * scale;

                        float nm = score > max_v[h] ? score : max_v[h];
                        float eo = std::exp(max_v[h] - nm);
                        float en = std::exp(score    - nm);
                        sum_v[h] = eo * sum_v[h] + en;
                        max_v[h] = nm;

                        __m512 veo = _mm512_set1_ps(eo);
                        __m512 ven = _mm512_set1_ps(en);
                        for (int d = 0; d < head_dim; d += 16) {
                            _mm512_storeu_ps(o_h + d,
                                _mm512_fmadd_ps(veo, _mm512_loadu_ps(o_h + d),
                                                _mm512_mul_ps(ven, _mm512_loadu_ps(k_fp32 + d))));
                        }
                    }
                }

                for (int h = h0; h < h1; h++) {
                    float* o_h   = o_ptr   + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    float* lse_h = lse_ptr + (b * n_heads + h) * seqlen + sq;
                    normalize_fp32(o_h, sum_v[h], head_dim);
                    *lse_h = max_v[h] + std::log(sum_v[h]);
                }
            }
        }
    });
}

void fused_csa_attn_bsq_bf16(
        torch::Tensor q,
        torch::Tensor kv,
        float scale,
        torch::Tensor o_out,
        torch::Tensor lse_out,
        int n_threads = 0) {

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bf16");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "q/kv must be contiguous");
    TORCH_CHECK(o_out.is_contiguous() && lse_out.is_contiguous(), "o/lse must be contiguous");

    const int bs        = q.size(0);
    const int n_heads   = q.size(1);
    const int seqlen    = q.size(2);
    const int head_dim  = q.size(3);
    const int n_compress = kv.size(1);
    TORCH_CHECK(head_dim % 16 == 0);

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    const uint16_t* q_ptr   = reinterpret_cast<const uint16_t*>(q.data_ptr<at::BFloat16>());
    const uint16_t* kv_ptr  = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float*          o_ptr   = o_out.data_ptr<float>();
    float*          lse_ptr = lse_out.data_ptr<float>();

    const int total_tasks = bs * seqlen;
    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        std::vector<float> max_v(n_heads);
        std::vector<float> sum_v(n_heads);
        float k_fp32[512];

        constexpr int HEAD_BLOCK = 8;
        float q_blk  [HEAD_BLOCK * 512];
        float o_local[HEAD_BLOCK * 512];

        for (int task = t_start; task < t_end; task++) {
            int sq = task % seqlen;
            int b  = task / seqlen;

            const uint16_t* kv_b = kv_ptr + b * n_compress * head_dim;

            for (int h0 = 0; h0 < n_heads; h0 += HEAD_BLOCK) {
                int h1 = std::min(h0 + HEAD_BLOCK, n_heads);

                for (int h = h0; h < h1; h++) {
                    const uint16_t* q_h_bf16 = q_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    bf16_row_to_fp32(q_blk   + (h - h0) * head_dim, q_h_bf16, head_dim);
                    float* ol = o_local + (h - h0) * head_dim;
                    for (int d = 0; d < head_dim; d++) ol[d] = 0.0f;
                    max_v[h] = -std::numeric_limits<float>::infinity();
                    sum_v[h] = 0.0f;
                }

                for (int i = 0; i < n_compress; i++) {
                    const uint16_t* ki = kv_b + i * head_dim;
                    if (i + PREFETCH_DIST < n_compress)
                        prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST) * head_dim, head_dim);
                    bf16_row_to_fp32(k_fp32, ki, head_dim);

                    for (int h = h0; h < h1; h++) {
                        const float* q_h = q_blk   + (h - h0) * head_dim;
                        float*       ol  = o_local  + (h - h0) * head_dim;

                        __m512 acc = _mm512_setzero_ps();
                        for (int d = 0; d < head_dim; d += 16)
                            acc = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d),
                                                  _mm512_loadu_ps(k_fp32 + d), acc);
                        float score = hsum_avx512(acc) * scale;

                        float nm = score > max_v[h] ? score : max_v[h];
                        float eo = std::exp(max_v[h] - nm);
                        float en = std::exp(score    - nm);
                        sum_v[h] = eo * sum_v[h] + en;
                        max_v[h] = nm;

                        __m512 veo = _mm512_set1_ps(eo);
                        __m512 ven = _mm512_set1_ps(en);
                        for (int d = 0; d < head_dim; d += 16) {
                            _mm512_storeu_ps(ol + d,
                                _mm512_fmadd_ps(veo, _mm512_loadu_ps(ol + d),
                                                _mm512_mul_ps(ven, _mm512_loadu_ps(k_fp32 + d))));
                        }
                    }
                }

                for (int h = h0; h < h1; h++) {
                    float* ol    = o_local + (h - h0) * head_dim;
                    float* o_h   = o_ptr   + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    float* lse_h = lse_ptr + (b * n_heads + h) * seqlen + sq;
                    normalize_fp32(ol, sum_v[h], head_dim);
                    std::memcpy(o_h, ol, head_dim * sizeof(float));
                    *lse_h = max_v[h] + std::log(sum_v[h]);
                }
            }
        }
    });
}

static inline int32_t hsum_epi32_avx512(__m512i v) {
    __m256i lo = _mm512_castsi512_si256(v);
    __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    __m256i s  = _mm256_add_epi32(lo, hi);
    __m128i a  = _mm256_castsi256_si128(s);
    __m128i b  = _mm256_extracti128_si256(s, 1);
    a = _mm_add_epi32(a, b);
    a = _mm_hadd_epi32(a, a);
    a = _mm_hadd_epi32(a, a);
    return _mm_cvtsi128_si32(a);
}

static inline float quantize_fp32_to_u8(uint8_t* dst, const float* src, int n) {
    __m512 vmax = _mm512_setzero_ps();
    for (int d = 0; d < n; d += 16)
        vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(src + d)));
    float maxv = _mm512_reduce_max_ps(vmax);
    if (maxv < 1e-6f) maxv = 1e-6f;
    float scale   = maxv / 127.0f;
    float inv_s   = 127.0f / maxv;
    __m512 vinv   = _mm512_set1_ps(inv_s);
    __m512i v128  = _mm512_set1_epi32(128);
    for (int d = 0; d < n; d += 16) {
        __m512 v    = _mm512_mul_ps(_mm512_loadu_ps(src + d), vinv);
        __m512i vi  = _mm512_cvtps_epi32(v);
        vi = _mm512_add_epi32(vi, v128);

        __m256i lo  = _mm512_castsi512_si256(vi);
        __m256i hi  = _mm512_extracti64x4_epi64(vi, 1);
        __m128i p0  = _mm_packs_epi32(_mm256_castsi256_si128(lo), _mm256_extracti128_si256(lo,1));
        __m128i p1  = _mm_packs_epi32(_mm256_castsi256_si128(hi), _mm256_extracti128_si256(hi,1));
        __m128i p8  = _mm_packus_epi16(p0, p1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + d), p8);
    }
    return scale;
}

static inline float quantize_bf16_to_s8(int8_t* dst, int32_t* sum_out,
                                         const uint16_t* src, int n) {

    float tmp[512];
    for (int d = 0; d < n; d += 16) {
        __m256i v16  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + d));
        __m512i v32i = _mm512_slli_epi32(_mm512_cvtepu16_epi32(v16), 16);
        _mm512_storeu_ps(tmp + d, _mm512_castsi512_ps(v32i));
    }

    __m512 vmax = _mm512_setzero_ps();
    for (int d = 0; d < n; d += 16)
        vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(tmp + d)));
    float maxv = _mm512_reduce_max_ps(vmax);
    if (maxv < 1e-6f) maxv = 1e-6f;
    float scale = maxv / 127.0f;
    float inv_s = 127.0f / maxv;
    __m512 vinv = _mm512_set1_ps(inv_s);
    __m512i vacc_sum = _mm512_setzero_si512();
    __m512i ones = _mm512_set1_epi32(1);
    for (int d = 0; d < n; d += 16) {
        __m512 v   = _mm512_mul_ps(_mm512_loadu_ps(tmp + d), vinv);
        __m512i vi = _mm512_cvtps_epi32(v);
        __m256i lo = _mm512_castsi512_si256(vi);
        __m256i hi = _mm512_extracti64x4_epi64(vi, 1);
        __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(lo), _mm256_extracti128_si256(lo,1));
        __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(hi), _mm256_extracti128_si256(hi,1));
        __m128i p8 = _mm_packs_epi16(p0, p1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + d), p8);

        __m512i ext = _mm512_cvtepi8_epi32(p8);
        vacc_sum = _mm512_add_epi32(vacc_sum, ext);
    }
    *sum_out = hsum_epi32_avx512(vacc_sum);
    return scale;
}

void fused_csa_attn_bsq_int8(
        torch::Tensor q,
        torch::Tensor kv,
        float scale,
        torch::Tensor o_out,
        torch::Tensor lse_out,
        int n_threads = 0) {

    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "q/kv must be contiguous");
    TORCH_CHECK(o_out.is_contiguous() && lse_out.is_contiguous(), "o/lse must be contiguous");

    const int bs         = q.size(0);
    const int n_heads    = q.size(1);
    const int seqlen     = q.size(2);
    const int head_dim   = q.size(3);
    const int n_compress = kv.size(1);
    TORCH_CHECK(head_dim % 64 == 0 && head_dim <= 512, "head_dim must be multiple of 64 and <=512");

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    const float*    q_ptr   = q.data_ptr<float>();
    const uint16_t* kv_ptr  = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float*          o_ptr   = o_out.data_ptr<float>();
    float*          lse_ptr = lse_out.data_ptr<float>();

    const int total_tasks = bs * seqlen;
    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        std::vector<float> max_v(n_heads);
        std::vector<float> sum_v(n_heads);

        alignas(64) int8_t  k_s8[512];

        alignas(64) uint8_t q_u8[16 * 512];
        float q_scales[16];

        for (int task = t_start; task < t_end; task++) {
            int sq = task % seqlen;
            int b  = task / seqlen;

            const uint16_t* kv_b = kv_ptr + b * n_compress * head_dim;

            constexpr int HEAD_BLOCK = 16;

            for (int h0 = 0; h0 < n_heads; h0 += HEAD_BLOCK) {
                int h1 = std::min(h0 + HEAD_BLOCK, n_heads);

                for (int h = h0; h < h1; h++) {
                    const float* q_h = q_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    q_scales[h - h0] = quantize_fp32_to_u8(q_u8 + (h - h0) * head_dim, q_h, head_dim);
                    max_v[h] = -std::numeric_limits<float>::infinity();
                    sum_v[h] = 0.0f;
                    float* o_h = o_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    for (int d = 0; d < head_dim; d++) o_h[d] = 0.0f;
                }

                for (int i = 0; i < n_compress; i++) {
                    const uint16_t* ki = kv_b + i * head_dim;
                    if (i + PREFETCH_DIST < n_compress)
                        prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST) * head_dim, head_dim);

                    int32_t sum_k;
                    float k_scale = quantize_bf16_to_s8(k_s8, &sum_k, ki, head_dim);

                    for (int h = h0; h < h1; h++) {
                        const uint8_t* qi = q_u8 + (h - h0) * head_dim;
                        float*         o_h = o_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;

                        __m512i vacc = _mm512_setzero_si512();
                        for (int d = 0; d < head_dim; d += 64) {
                            __m512i vq = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qi + d));
                            __m512i vk = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(k_s8 + d));
                            vacc = _mm512_dpbusd_epi32(vacc, vq, vk);
                        }

                        int32_t dot_i32 = hsum_epi32_avx512(vacc) - 128 * sum_k;
                        float score = (float)dot_i32 * q_scales[h - h0] * k_scale * scale;

                        float nm = score > max_v[h] ? score : max_v[h];
                        float eo = std::exp(max_v[h] - nm);
                        float en = std::exp(score    - nm);
                        sum_v[h] = eo * sum_v[h] + en;
                        max_v[h] = nm;

                        __m512 veo  = _mm512_set1_ps(eo);
                        __m512 ven  = _mm512_set1_ps(en * k_scale);
                        for (int d = 0; d < head_dim; d += 16) {
                            __m128i k8  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k_s8 + d));
                            __m256i k16 = _mm256_cvtepi8_epi16(k8);
                            __m512i k32 = _mm512_cvtepi16_epi32(k16);
                            __m512  kf  = _mm512_cvtepi32_ps(k32);
                            __m512  ov  = _mm512_loadu_ps(o_h + d);
                            _mm512_storeu_ps(o_h + d, _mm512_fmadd_ps(veo, ov, _mm512_mul_ps(ven, kf)));
                        }
                    }
                }

                for (int h = h0; h < h1; h++) {
                    float* o_h   = o_ptr   + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    float* lse_h = lse_ptr + (b * n_heads + h) * seqlen + sq;
                    normalize_fp32(o_h, sum_v[h], head_dim);
                    *lse_h = max_v[h] + std::log(sum_v[h]);
                }
            }
        }
    });
}

void fused_csa_attn_fast_bf16(
        torch::Tensor q,
        torch::Tensor kv,
        float scale,
        torch::Tensor o_out,
        torch::Tensor lse_out,
        int n_threads = 0) {

    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16);
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous());
    TORCH_CHECK(o_out.is_contiguous() && lse_out.is_contiguous());

    const int bs         = q.size(0);
    const int n_heads    = q.size(1);
    const int seqlen     = q.size(2);
    const int head_dim   = q.size(3);
    const int n_compress = kv.size(1);
    TORCH_CHECK(head_dim % 64 == 0 && head_dim <= 512);

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    const uint16_t* q_ptr   = reinterpret_cast<const uint16_t*>(q.data_ptr<at::BFloat16>());
    const uint16_t* kv_ptr  = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float*          o_ptr   = o_out.data_ptr<float>();
    float*          lse_ptr = lse_out.data_ptr<float>();

    constexpr int SQ_BLOCK  = 2;

    constexpr int HB = 4;
    const int sq_tiles = (seqlen + SQ_BLOCK - 1) / SQ_BLOCK;
    const int total_tasks = bs * sq_tiles;
    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        float q_blk  [SQ_BLOCK * HB * 512];
        float o_local[SQ_BLOCK * HB * 512];
        float max_v  [SQ_BLOCK * HB];
        float sum_v  [SQ_BLOCK * HB];

        for (int task = t_start; task < t_end; task++) {
            int sq0 = (task % sq_tiles) * SQ_BLOCK;
            int b   = task / sq_tiles;
            int sq1 = std::min(sq0 + SQ_BLOCK, seqlen);
            int nsq = sq1 - sq0;

            const uint16_t* kv_b = kv_ptr + b * n_compress * head_dim;

            for (int h0 = 0; h0 < n_heads; h0 += HB) {
                int h1 = std::min(h0 + HB, n_heads);
                int hb = h1 - h0;

                for (int s = 0; s < nsq; s++) {
                    int sq = sq0 + s;
                    for (int h = h0; h < h1; h++) {
                        int hi = h - h0;
                        const uint16_t* q_bf16 = q_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                        bf16_row_to_fp32(q_blk + (s * HB + hi) * head_dim, q_bf16, head_dim);
                        float* ol = o_local + (s * HB + hi) * head_dim;
                        for (int d = 0; d < head_dim; d++) ol[d] = 0.f;
                        max_v[s * HB + hi] = -std::numeric_limits<float>::infinity();
                        sum_v[s * HB + hi] = 0.f;
                    }
                }

                float eo_buf[SQ_BLOCK * HB];
                float en_buf[SQ_BLOCK * HB];
                for (int i = 0; i < n_compress; i++) {
                    const uint16_t* ki = kv_b + i * head_dim;
                    if (i + PREFETCH_DIST < n_compress)
                        prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST) * head_dim, head_dim);

                    __m512 acc[SQ_BLOCK][HB][4];
                    for (int s = 0; s < nsq; s++)
                        for (int h = 0; h < hb; h++)
                            acc[s][h][0] = acc[s][h][1] = acc[s][h][2] = acc[s][h][3] =
                                _mm512_setzero_ps();

                    for (int d = 0; d < head_dim; d += 64) {
                        __m512 vk0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ki + d))), 16));
                        __m512 vk1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ki + d + 16))), 16));
                        __m512 vk2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ki + d + 32))), 16));
                        __m512 vk3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ki + d + 48))), 16));
                        for (int s = 0; s < nsq; s++) {
                            for (int h = 0; h < hb; h++) {
                                const float* qh = q_blk + (s * HB + h) * head_dim + d;
                                acc[s][h][0] = _mm512_fmadd_ps(_mm512_loadu_ps(qh),      vk0, acc[s][h][0]);
                                acc[s][h][1] = _mm512_fmadd_ps(_mm512_loadu_ps(qh + 16), vk1, acc[s][h][1]);
                                acc[s][h][2] = _mm512_fmadd_ps(_mm512_loadu_ps(qh + 32), vk2, acc[s][h][2]);
                                acc[s][h][3] = _mm512_fmadd_ps(_mm512_loadu_ps(qh + 48), vk3, acc[s][h][3]);
                            }
                        }
                    }

                    for (int s = 0; s < nsq; s++) {
                        for (int h = 0; h < hb; h++) {
                            float score = hsum_avx512(_mm512_add_ps(_mm512_add_ps(acc[s][h][0], acc[s][h][1]),
                                                                     _mm512_add_ps(acc[s][h][2], acc[s][h][3]))) * scale;
                            int idx = s * HB + h;
                            float nm = score > max_v[idx] ? score : max_v[idx];
                            eo_buf[idx] = fast_expf(max_v[idx] - nm);
                            en_buf[idx] = fast_expf(score - nm);
                            sum_v[idx]  = eo_buf[idx] * sum_v[idx] + en_buf[idx];
                            max_v[idx]  = nm;
                        }
                    }

                    for (int s = 0; s < nsq; s++) {
                        for (int h = 0; h < hb; h++) {
                            int idx = s * HB + h;
                            float* ol = o_local + idx * head_dim;
                            __m512 veo = _mm512_set1_ps(eo_buf[idx]);
                            __m512 ven = _mm512_set1_ps(en_buf[idx]);
                            for (int d = 0; d < head_dim; d += 64) {
                                __m512 vk0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ki + d))), 16));
                                __m512 vk1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ki + d + 16))), 16));
                                __m512 vk2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ki + d + 32))), 16));
                                __m512 vk3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ki + d + 48))), 16));
                                _mm512_storeu_ps(ol + d,
                                    _mm512_fmadd_ps(veo, _mm512_loadu_ps(ol + d), _mm512_mul_ps(ven, vk0)));
                                _mm512_storeu_ps(ol + d + 16,
                                    _mm512_fmadd_ps(veo, _mm512_loadu_ps(ol + d + 16), _mm512_mul_ps(ven, vk1)));
                                _mm512_storeu_ps(ol + d + 32,
                                    _mm512_fmadd_ps(veo, _mm512_loadu_ps(ol + d + 32), _mm512_mul_ps(ven, vk2)));
                                _mm512_storeu_ps(ol + d + 48,
                                    _mm512_fmadd_ps(veo, _mm512_loadu_ps(ol + d + 48), _mm512_mul_ps(ven, vk3)));
                            }
                        }
                    }
                }

                for (int s = 0; s < nsq; s++) {
                    int sq = sq0 + s;
                    for (int h = h0; h < h1; h++) {
                        int hi = h - h0;
                        float* ol    = o_local + (s * HB + hi) * head_dim;
                        float* o_h   = o_ptr   + ((b * n_heads + h) * seqlen + sq) * head_dim;
                        float* lse_h = lse_ptr + (b * n_heads + h) * seqlen + sq;
                        normalize_fp32(ol, sum_v[s * HB + hi], head_dim);
                        std::memcpy(o_h, ol, head_dim * sizeof(float));
                        *lse_h = max_v[s * HB + hi] + std::log(sum_v[s * HB + hi]);
                    }
                }
            }
        }
    });
}

void fused_csa_attn_2pass_bf16(
        torch::Tensor q,
        torch::Tensor kv,
        float scale,
        torch::Tensor o_out,
        torch::Tensor lse_out,
        int n_threads = 0) {

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bf16");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous());
    TORCH_CHECK(o_out.is_contiguous() && lse_out.is_contiguous());

    const int bs         = q.size(0);
    const int n_heads    = q.size(1);
    const int seqlen     = q.size(2);
    const int head_dim   = q.size(3);
    const int n_compress = kv.size(1);
    TORCH_CHECK(head_dim % 16 == 0 && head_dim <= 512);

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    const uint16_t* q_ptr   = reinterpret_cast<const uint16_t*>(q.data_ptr<at::BFloat16>());
    const uint16_t* kv_ptr  = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float*          o_ptr   = o_out.data_ptr<float>();
    float*          lse_ptr = lse_out.data_ptr<float>();

    const int total_tasks = bs * seqlen;
    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        std::vector<float> scores_exp(n_compress);
        float k_fp32[512];
        constexpr int HEAD_BLOCK = 8;
        float q_blk  [HEAD_BLOCK * 512];
        float o_local[HEAD_BLOCK * 512];
        float max_v  [HEAD_BLOCK];
        float sum_v  [HEAD_BLOCK];

        for (int task = t_start; task < t_end; task++) {
            int sq = task % seqlen;
            int b  = task / seqlen;
            const uint16_t* kv_b = kv_ptr + b * n_compress * head_dim;

            for (int h0 = 0; h0 < n_heads; h0 += HEAD_BLOCK) {
                int h1 = std::min(h0 + HEAD_BLOCK, n_heads);
                int hb = h1 - h0;

                for (int h = h0; h < h1; h++) {
                    const uint16_t* q_h_bf16 = q_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    bf16_row_to_fp32(q_blk + (h - h0) * head_dim, q_h_bf16, head_dim);
                    max_v[h - h0] = -std::numeric_limits<float>::infinity();
                }

                for (int i = 0; i < n_compress; i++) {
                    const uint16_t* ki = kv_b + i * head_dim;
                    if (i + PREFETCH_DIST < n_compress)
                        prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST) * head_dim, head_dim);
                    bf16_row_to_fp32(k_fp32, ki, head_dim);
                    for (int h = 0; h < hb; h++) {
                        __m512 acc = _mm512_setzero_ps();
                        const float* qh = q_blk + h * head_dim;
                        for (int d = 0; d < head_dim; d += 16)
                            acc = _mm512_fmadd_ps(_mm512_loadu_ps(qh + d),
                                                  _mm512_loadu_ps(k_fp32 + d), acc);
                        float s = hsum_avx512(acc) * scale;
                        if (s > max_v[h]) max_v[h] = s;
                    }
                }

                for (int h = 0; h < hb; h++) {
                    const float* qh = q_blk + h * head_dim;
                    float gmax = max_v[h];
                    float gsum = 0.0f;

                    for (int i = 0; i < n_compress; i++) {
                        const uint16_t* ki = kv_b + i * head_dim;
                        if (i + PREFETCH_DIST < n_compress)
                            prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST) * head_dim, head_dim);
                        bf16_row_to_fp32(k_fp32, ki, head_dim);
                        __m512 acc = _mm512_setzero_ps();
                        for (int d = 0; d < head_dim; d += 16)
                            acc = _mm512_fmadd_ps(_mm512_loadu_ps(qh + d),
                                                  _mm512_loadu_ps(k_fp32 + d), acc);
                        float s = hsum_avx512(acc) * scale;
                        float e = std::exp(s - gmax);
                        scores_exp[i] = e;
                        gsum += e;
                    }

                    float* ol = o_local + h * head_dim;
                    for (int d = 0; d < head_dim; d++) ol[d] = 0.0f;
                    for (int i = 0; i < n_compress; i++) {
                        const uint16_t* ki = kv_b + i * head_dim;
                        if (i + PREFETCH_DIST < n_compress)
                            prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST) * head_dim, head_dim);
                        bf16_row_to_fp32(k_fp32, ki, head_dim);
                        __m512 ve = _mm512_set1_ps(scores_exp[i]);
                        for (int d = 0; d < head_dim; d += 16)
                            _mm512_storeu_ps(ol + d,
                                _mm512_fmadd_ps(ve, _mm512_loadu_ps(k_fp32 + d),
                                                    _mm512_loadu_ps(ol + d)));
                    }

                    float* o_h   = o_ptr   + ((b * n_heads + h0 + h) * seqlen + sq) * head_dim;
                    float* lse_h = lse_ptr + (b * n_heads + h0 + h) * seqlen + sq;
                    normalize_fp32(ol, gsum, head_dim);
                    std::memcpy(o_h, ol, head_dim * sizeof(float));
                    *lse_h = gmax + std::log(gsum);
                }
            }
        }
    });
}

std::tuple<torch::Tensor, torch::Tensor>
q_quantize_fp32_to_u8(torch::Tensor q, int n_threads = 0) {
    TORCH_CHECK(q.is_contiguous() && q.scalar_type() == at::kFloat);
    const int bs = q.size(0), nh = q.size(1), sq = q.size(2), hd = q.size(3);
    TORCH_CHECK(hd % 16 == 0 && hd <= 512);

    if (n_threads <= 0) n_threads = std::thread::hardware_concurrency();

    auto q_u8    = torch::empty({bs, nh, sq, hd}, torch::dtype(torch::kUInt8));
    auto q_scales= torch::empty({bs, nh, sq},     torch::dtype(torch::kFloat32));

    const float*  qf_ptr = q.data_ptr<float>();
    uint8_t*      qu_ptr = q_u8.data_ptr<uint8_t>();
    float*        qs_ptr = q_scales.data_ptr<float>();

    const int total = bs * nh * sq;
    ThreadPool& pool = get_pool(n_threads);
    pool.run([&](int tid) {
        int chunk   = (total + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total);
        for (int t = t_start; t < t_end; t++) {
            qs_ptr[t] = quantize_fp32_to_u8(qu_ptr + t * hd, qf_ptr + t * hd, hd);
        }
    });
    return std::make_tuple(q_u8, q_scales);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
kv_quantize_bf16_to_int8(torch::Tensor kv, int n_threads = 0) {
    TORCH_CHECK(kv.is_contiguous() && kv.scalar_type() == at::kBFloat16);
    const int bs = kv.size(0), nc = kv.size(1), hd = kv.size(2);
    TORCH_CHECK(hd % 16 == 0 && hd <= 512);

    if (n_threads <= 0) n_threads = std::thread::hardware_concurrency();

    auto kv_int8   = torch::empty({bs, nc, hd}, torch::dtype(torch::kInt8));
    auto kv_scales = torch::empty({bs, nc},     torch::dtype(torch::kFloat32));
    auto kv_sum_k  = torch::empty({bs, nc},     torch::dtype(torch::kInt32));

    const uint16_t* kv_ptr    = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    int8_t*         kvi_ptr   = kv_int8.data_ptr<int8_t>();
    float*          kvs_ptr   = kv_scales.data_ptr<float>();
    int32_t*        kvsum_ptr = kv_sum_k.data_ptr<int32_t>();

    const int total = bs * nc;
    ThreadPool& pool = get_pool(n_threads);
    pool.run([&](int tid) {
        int chunk   = (total + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total);
        for (int t = t_start; t < t_end; t++) {
            const uint16_t* src = kv_ptr  + t * hd;
            int8_t*         dst = kvi_ptr + t * hd;
            kvs_ptr[t]   = quantize_bf16_to_s8(dst, kvsum_ptr + t, src, hd);
        }
    });
    return std::make_tuple(kv_int8, kv_scales, kv_sum_k);
}

void fused_csa_attn_bsq_int8_preq(
        torch::Tensor q,
        torch::Tensor kv_i8,
        torch::Tensor kv_scales,
        torch::Tensor kv_sums,
        float scale,
        torch::Tensor o_out,
        torch::Tensor lse_out,
        int n_threads = 0) {

    TORCH_CHECK(q.is_contiguous() && kv_i8.is_contiguous());
    TORCH_CHECK(kv_scales.is_contiguous() && kv_sums.is_contiguous());
    TORCH_CHECK(o_out.is_contiguous() && lse_out.is_contiguous());

    const int bs         = q.size(0);
    const int n_heads    = q.size(1);
    const int seqlen     = q.size(2);
    const int head_dim   = q.size(3);
    const int n_compress = kv_i8.size(1);
    TORCH_CHECK(head_dim % 64 == 0 && head_dim <= 512);

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    const float*   q_ptr    = q.data_ptr<float>();
    const int8_t*  kvi_ptr  = kv_i8.data_ptr<int8_t>();
    const float*   kvs_ptr  = kv_scales.data_ptr<float>();
    const int32_t* kvsum_ptr= kv_sums.data_ptr<int32_t>();
    float*         o_ptr    = o_out.data_ptr<float>();
    float*         lse_ptr  = lse_out.data_ptr<float>();

    const int total_tasks = bs * seqlen;
    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        std::vector<float> max_v(n_heads);
        std::vector<float> sum_v(n_heads);

        alignas(64) uint8_t q_u8[16 * 512];
        float q_scales[16];

        for (int task = t_start; task < t_end; task++) {
            int sq = task % seqlen;
            int b  = task / seqlen;

            const int8_t*  kvi_b  = kvi_ptr  + b * n_compress * head_dim;
            const float*   kvs_b  = kvs_ptr  + b * n_compress;
            const int32_t* kvsum_b= kvsum_ptr + b * n_compress;

            constexpr int HEAD_BLOCK = 16;

            for (int h0 = 0; h0 < n_heads; h0 += HEAD_BLOCK) {
                int h1 = std::min(h0 + HEAD_BLOCK, n_heads);

                for (int h = h0; h < h1; h++) {
                    const float* q_h = q_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    q_scales[h - h0] = quantize_fp32_to_u8(q_u8 + (h - h0) * head_dim, q_h, head_dim);
                    max_v[h] = -std::numeric_limits<float>::infinity();
                    sum_v[h] = 0.0f;
                    float* o_h = o_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    for (int d = 0; d < head_dim; d++) o_h[d] = 0.0f;
                }

                for (int i = 0; i < n_compress; i++) {
                    if (i + PREFETCH_DIST < n_compress) {
                        const int8_t* pf = kvi_b + (i + PREFETCH_DIST) * head_dim;
                        for (int d = 0; d < head_dim; d += 64)
                            _mm_prefetch(reinterpret_cast<const char*>(pf) + d, _MM_HINT_T1);
                    }

                    const int8_t* ki  = kvi_b  + i * head_dim;
                    float         ksc = kvs_b[i];
                    int32_t       ksum= kvsum_b[i];

                    for (int h = h0; h < h1; h++) {
                        const uint8_t* qi  = q_u8 + (h - h0) * head_dim;
                        float*         o_h = o_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;

                        __m512i vacc = _mm512_setzero_si512();
                        for (int d = 0; d < head_dim; d += 64) {
                            __m512i vq = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qi + d));
                            __m512i vk = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ki + d));
                            vacc = _mm512_dpbusd_epi32(vacc, vq, vk);
                        }
                        int32_t dot_i32 = hsum_epi32_avx512(vacc) - 128 * ksum;
                        float score = (float)dot_i32 * q_scales[h - h0] * ksc * scale;

                        float nm = score > max_v[h] ? score : max_v[h];
                        float eo = std::exp(max_v[h] - nm);
                        float en = std::exp(score    - nm);
                        sum_v[h] = eo * sum_v[h] + en;
                        max_v[h] = nm;

                        __m512 veo = _mm512_set1_ps(eo);
                        __m512 ven = _mm512_set1_ps(en * ksc);
                        for (int d = 0; d < head_dim; d += 16) {
                            __m128i k8  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ki + d));
                            __m512  kf  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(k8)));
                            __m512  ov  = _mm512_loadu_ps(o_h + d);
                            _mm512_storeu_ps(o_h + d, _mm512_fmadd_ps(veo, ov, _mm512_mul_ps(ven, kf)));
                        }
                    }
                }

                for (int h = h0; h < h1; h++) {
                    float* o_h   = o_ptr   + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    float* lse_h = lse_ptr + (b * n_heads + h) * seqlen + sq;
                    normalize_fp32(o_h, sum_v[h], head_dim);
                    *lse_h = max_v[h] + std::log(sum_v[h]);
                }
            }
        }
    });
}

void fused_csa_attn_bsq_int8_full(
        torch::Tensor q_u8,
        torch::Tensor q_scales,
        torch::Tensor kv_i8,
        torch::Tensor kv_scales,
        torch::Tensor kv_sums,
        torch::Tensor kv_bf16,
        float scale,
        torch::Tensor o_out,
        torch::Tensor lse_out,
        int n_threads = 0) {

    TORCH_CHECK(q_u8.is_contiguous() && kv_i8.is_contiguous() && kv_bf16.is_contiguous());
    TORCH_CHECK(q_scales.is_contiguous() && kv_scales.is_contiguous() && kv_sums.is_contiguous());
    TORCH_CHECK(o_out.is_contiguous() && lse_out.is_contiguous());

    const int bs         = q_u8.size(0);
    const int n_heads    = q_u8.size(1);
    const int seqlen     = q_u8.size(2);
    const int head_dim   = q_u8.size(3);
    const int n_compress = kv_i8.size(1);
    TORCH_CHECK(head_dim % 64 == 0 && head_dim <= 512);

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    const uint8_t*  qu_ptr   = q_u8.data_ptr<uint8_t>();
    const float*    qs_ptr   = q_scales.data_ptr<float>();
    const int8_t*   kvi_ptr  = kv_i8.data_ptr<int8_t>();
    const float*    kvs_ptr  = kv_scales.data_ptr<float>();
    const int32_t*  kvsum_ptr= kv_sums.data_ptr<int32_t>();
    const uint16_t* kvbf_ptr = reinterpret_cast<const uint16_t*>(kv_bf16.data_ptr<at::BFloat16>());
    float*         o_ptr    = o_out.data_ptr<float>();
    float*         lse_ptr  = lse_out.data_ptr<float>();

    const int total_tasks = bs * seqlen;
    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        std::vector<float> max_v(n_heads);
        std::vector<float> sum_v(n_heads);

        for (int task = t_start; task < t_end; task++) {
            int sq = task % seqlen;
            int b  = task / seqlen;

            const int8_t*   kvi_b  = kvi_ptr  + b * n_compress * head_dim;
            const float*    kvs_b  = kvs_ptr  + b * n_compress;
            const int32_t*  kvsum_b= kvsum_ptr + b * n_compress;
            const uint16_t* kvbf_b = kvbf_ptr  + b * n_compress * head_dim;

            constexpr int HEAD_BLOCK = 16;

            for (int h0 = 0; h0 < n_heads; h0 += HEAD_BLOCK) {
                int h1 = std::min(h0 + HEAD_BLOCK, n_heads);

                for (int h = h0; h < h1; h++) {
                    max_v[h] = -std::numeric_limits<float>::infinity();
                    sum_v[h] = 0.0f;
                    float* o_h = o_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    for (int d = 0; d < head_dim; d++) o_h[d] = 0.0f;
                }

                for (int i = 0; i < n_compress; i++) {
                    if (i + PREFETCH_DIST < n_compress) {
                        const int8_t* pf = kvi_b + (i + PREFETCH_DIST) * head_dim;
                        for (int d = 0; d < head_dim; d += 64)
                            _mm_prefetch(reinterpret_cast<const char*>(pf) + d, _MM_HINT_T1);
                    }

                    const int8_t*   ki   = kvi_b   + i * head_dim;
                    const uint16_t* kbf  = kvbf_b  + i * head_dim;
                    float           ksc  = kvs_b[i];
                    int32_t         ksum = kvsum_b[i];

                    for (int h = h0; h < h1; h++) {
                        const uint8_t* qi  = qu_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                        float          qsc = qs_ptr [ (b * n_heads + h) * seqlen + sq];
                        float*         o_h = o_ptr  + ((b * n_heads + h) * seqlen + sq) * head_dim;

                        __m512i vacc = _mm512_setzero_si512();
                        for (int d = 0; d < head_dim; d += 64) {
                            __m512i vq = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qi + d));
                            __m512i vk = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ki + d));
                            vacc = _mm512_dpbusd_epi32(vacc, vq, vk);
                        }
                        int32_t dot_i32 = hsum_epi32_avx512(vacc) - 128 * ksum;
                        float score = (float)dot_i32 * qsc * ksc * scale;

                        float nm = score > max_v[h] ? score : max_v[h];
                        float eo = std::exp(max_v[h] - nm);
                        float en = std::exp(score    - nm);
                        sum_v[h] = eo * sum_v[h] + en;
                        max_v[h] = nm;

                        __m512 veo = _mm512_set1_ps(eo);
                        __m512 ven = _mm512_set1_ps(en);
                        for (int d = 0; d < head_dim; d += 16) {
                            __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kbf + d));
                            __m512  kf  = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v16), 16));
                            __m512  ov  = _mm512_loadu_ps(o_h + d);
                            _mm512_storeu_ps(o_h + d, _mm512_fmadd_ps(veo, ov, _mm512_mul_ps(ven, kf)));
                        }
                    }
                }

                for (int h = h0; h < h1; h++) {
                    float* o_h   = o_ptr   + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    float* lse_h = lse_ptr + (b * n_heads + h) * seqlen + sq;
                    normalize_fp32(o_h, sum_v[h], head_dim);
                    *lse_h = max_v[h] + std::log(sum_v[h]);
                }
            }
        }
    });
}

void fused_csa_attn_inplace(
        torch::Tensor q,
        torch::Tensor kv,
        float scale,
        torch::Tensor o_out,
        torch::Tensor lse_out,
        int n_threads) {

    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "q/kv must be contiguous");
    TORCH_CHECK(o_out.is_contiguous() && lse_out.is_contiguous(), "o_out/lse_out must be contiguous");

    const int bs = q.size(0), n_heads = q.size(1), seqlen = q.size(2);
    const int head_dim  = q.size(3);
    const int n_compress = kv.size(1);
    TORCH_CHECK(head_dim % 16 == 0);

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = std::thread::hardware_concurrency();
#endif
    }

    const float*    q_ptr   = q.data_ptr<float>();
    const uint16_t* kv_ptr  = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float*          o_ptr   = o_out.data_ptr<float>();
    float*          lse_ptr = lse_out.data_ptr<float>();

    const int total_tasks = bs * seqlen;
    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total_tasks + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total_tasks);

        std::vector<float> max_v(n_heads, -std::numeric_limits<float>::infinity());
        std::vector<float> sum_v(n_heads, 0.0f);

        for (int task = t_start; task < t_end; task++) {
            int sq = task % seqlen;
            int b  = task / seqlen;

            const uint16_t* kv_b = kv_ptr + b * n_compress * head_dim;

            const int q_bsq_stride = seqlen * head_dim;

            for (int h = 0; h < n_heads; h++) {
                max_v[h] = -std::numeric_limits<float>::infinity();
                sum_v[h] = 0.0f;
                float* o_h = o_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                for (int d = 0; d < head_dim; d++) o_h[d] = 0.0f;
            }

            for (int i = 0; i < n_compress; i++) {
                const uint16_t* ki = kv_b + i * head_dim;

                if (i + PREFETCH_DIST < n_compress)
                    prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST) * head_dim, head_dim);

                float k_fp32[512];
                bf16_row_to_fp32(k_fp32, ki, head_dim);

                for (int h = 0; h < n_heads; h++) {
                    const float* q_h = q_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                    float* o_h       = o_ptr  + ((b * n_heads + h) * seqlen + sq) * head_dim;

                    __m512 acc = _mm512_setzero_ps();
                    for (int d = 0; d < head_dim; d += 16) {
                        __m512 qv = _mm512_loadu_ps(q_h    + d);
                        __m512 kv2 = _mm512_loadu_ps(k_fp32 + d);
                        acc = _mm512_fmadd_ps(qv, kv2, acc);
                    }
                    float score = hsum_avx512(acc) * scale;

                    float nm = score > max_v[h] ? score : max_v[h];
                    float eo = std::exp(max_v[h] - nm);
                    float en = std::exp(score    - nm);
                    sum_v[h] = eo * sum_v[h] + en;
                    max_v[h] = nm;

                    __m512 veo = _mm512_set1_ps(eo);
                    __m512 ven = _mm512_set1_ps(en);
                    for (int d = 0; d < head_dim; d += 16) {
                        __m512 ov = _mm512_loadu_ps(o_h + d);
                        __m512 kv2 = _mm512_loadu_ps(k_fp32 + d);
                        _mm512_storeu_ps(o_h + d, _mm512_fmadd_ps(veo, ov, _mm512_mul_ps(ven, kv2)));
                    }
                }
            }

            for (int h = 0; h < n_heads; h++) {
                float* o_h   = o_ptr   + ((b * n_heads + h) * seqlen + sq) * head_dim;
                float* lse_h = lse_ptr + (b * n_heads + h) * seqlen + sq;
                normalize_fp32(o_h, sum_v[h], head_dim);
                *lse_h = max_v[h] + std::log(sum_v[h]);
            }
        }
    });
}

std::tuple<torch::Tensor, torch::Tensor> fused_csa_attn_range(
        torch::Tensor q,
        torch::Tensor kv,
        float scale,
        int kv_s,
        int kv_e) {

    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "must be contiguous");
    const int bs       = q.size(0);
    const int n_heads  = q.size(1);
    const int seqlen   = q.size(2);
    const int head_dim = q.size(3);
    const int n_compress = kv.size(1);
    TORCH_CHECK(kv_s >= 0 && kv_e <= n_compress && kv_s < kv_e, "invalid kv range");

    auto o_out  = torch::empty({bs, n_heads, seqlen, head_dim}, q.options());
    auto lse_out = torch::empty({bs, n_heads, seqlen}, q.options());

    const float*    q_ptr  = q.data_ptr<float>();
    const uint16_t* kv_ptr = reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>());
    float* o_ptr   = o_out.data_ptr<float>();
    float* lse_ptr = lse_out.data_ptr<float>();

    for (int b = 0; b < bs; b++) {
        const uint16_t* kv_b = kv_ptr + b * n_compress * head_dim;
        for (int h = 0; h < n_heads; h++) {
            for (int sq = 0; sq < seqlen; sq++) {
                const float* q_row = q_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                float*       o_row = o_ptr + ((b * n_heads + h) * seqlen + sq) * head_dim;
                float* lse_row     = lse_ptr + (b * n_heads + h) * seqlen + sq;

                float max_val = -std::numeric_limits<float>::infinity();
                float sum_val = 0.0f;
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

                int i = kv_s;
                for (; i + 1 < kv_e; i += 2) {
                    const uint16_t* k0 = kv_b + i * head_dim;
                    const uint16_t* k1 = kv_b + (i + 1) * head_dim;
                    if (i + PREFETCH_DIST + 1 < kv_e) {
                        prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST)     * head_dim, head_dim);
                        prefetch_kv_bf16(kv_b + (i + PREFETCH_DIST + 1) * head_dim, head_dim);
                    }
                    float s0 = dot_fp32_bf16(q_row, k0, head_dim) * scale;
                    float nm0 = s0 > max_val ? s0 : max_val;
                    float eo0 = std::exp(max_val - nm0), en0 = std::exp(s0 - nm0);
                    sum_val = eo0 * sum_val + en0; max_val = nm0;
                    update_bf16(o_row, k0, eo0, en0, head_dim);

                    float s1 = dot_fp32_bf16(q_row, k1, head_dim) * scale;
                    float nm1 = s1 > max_val ? s1 : max_val;
                    float eo1 = std::exp(max_val - nm1), en1 = std::exp(s1 - nm1);
                    sum_val = eo1 * sum_val + en1; max_val = nm1;
                    update_bf16(o_row, k1, eo1, en1, head_dim);
                }
                for (; i < kv_e; i++) {
                    const uint16_t* ki = kv_b + i * head_dim;
                    float score = dot_fp32_bf16(q_row, ki, head_dim) * scale;
                    float nm = score > max_val ? score : max_val;
                    float eo = std::exp(max_val - nm), en = std::exp(score - nm);
                    sum_val = eo * sum_val + en; max_val = nm;
                    update_bf16(o_row, ki, eo, en, head_dim);
                }
                normalize_fp32(o_row, sum_val, head_dim);
                *lse_row = max_val + std::log(sum_val);
            }
        }
    }
    return std::make_tuple(o_out, lse_out);
}

torch::Tensor indexer_topk_cpu(
        torch::Tensor q_idx,
        torch::Tensor kv_idx,
        torch::Tensor weights,
        int k,
        int n_threads = 0) {

    TORCH_CHECK(q_idx.dim() == 3 && kv_idx.dim() == 3, "q_idx/kv_idx must be 3D");
    TORCH_CHECK(q_idx.scalar_type() == at::kFloat && kv_idx.scalar_type() == at::kFloat, "must be fp32");
    TORCH_CHECK(q_idx.is_contiguous() && kv_idx.is_contiguous() && weights.is_contiguous(), "must be contiguous");

    const int bs      = q_idx.size(0);
    const int n_heads = q_idx.size(1);
    const int hd      = q_idx.size(2);
    const int nc      = kv_idx.size(1);
    k = std::min(k, nc);

    TORCH_CHECK(hd % 16 == 0, "hd must be multiple of 16");

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = (int)std::thread::hardware_concurrency();
#endif
    }

    auto out = torch::empty({bs, k}, torch::TensorOptions().dtype(torch::kInt32));

    const float* q_ptr = q_idx.data_ptr<float>();
    const float* kv_ptr = kv_idx.data_ptr<float>();
    const float* w_ptr  = weights.data_ptr<float>();
    int*         o_ptr  = out.data_ptr<int>();

    ThreadPool& pool = get_pool(n_threads);
    pool.run([&](int tid) {
        int chunk   = (bs + n_threads - 1) / n_threads;
        int b_start = tid * chunk;
        int b_end   = std::min(b_start + chunk, bs);

        std::vector<float> scores(nc);
        std::vector<int>   idx(nc);

        for (int b = b_start; b < b_end; b++) {
            const float* q_b  = q_ptr  + b * n_heads * hd;
            const float* kv_b = kv_ptr + b * nc * hd;
            const float* w_b  = w_ptr  + b * n_heads;

            for (int ni = 0; ni < nc; ni++) {
                const float* kv_ni = kv_b + ni * hd;
                float s = 0.0f;
                for (int h = 0; h < n_heads; h++) {
                    const float* q_h = q_b + h * hd;

                    __m512 acc = _mm512_setzero_ps();
                    for (int d = 0; d < hd; d += 16) {
                        __m512 qv = _mm512_loadu_ps(q_h  + d);
                        __m512 kv = _mm512_loadu_ps(kv_ni + d);
                        acc = _mm512_fmadd_ps(qv, kv, acc);
                    }
                    float dot = hsum_avx512(acc);

                    float rdot = dot > 0.0f ? dot : 0.0f;
                    s += rdot * w_b[h];
                }
                scores[ni] = s;
                idx[ni] = ni;
            }

            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b2){ return scores[a] > scores[b2]; });

            int* out_b = o_ptr + b * k;
            for (int i = 0; i < k; i++) out_b[i] = idx[i];
        }
    });

    return out;
}

torch::Tensor indexer_topk_cpu_par(
        torch::Tensor q_idx,
        torch::Tensor kv_idx,
        torch::Tensor weights,
        int k,
        int n_threads = 0) {

    TORCH_CHECK(q_idx.dim() == 3 && kv_idx.dim() == 3, "must be 3D");
    TORCH_CHECK(q_idx.is_contiguous() && kv_idx.is_contiguous() && weights.is_contiguous(),
                "must be contiguous");

    const int bs      = q_idx.size(0);
    const int n_heads = q_idx.size(1);
    const int hd      = q_idx.size(2);
    const int nc      = kv_idx.size(1);
    k = std::min(k, nc);

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = (int)std::thread::hardware_concurrency();
#endif
    }

    auto scores_t = torch::zeros({bs, nc}, torch::kFloat32);
    auto out      = torch::empty({bs, k},  torch::kInt32);

    const float* q_ptr  = q_idx.data_ptr<float>();
    const float* kv_ptr = kv_idx.data_ptr<float>();
    const float* w_ptr  = weights.data_ptr<float>();
    float*       s_ptr  = scores_t.data_ptr<float>();
    int*         o_ptr  = out.data_ptr<int>();

    const int total = bs * nc;

    ThreadPool& pool = get_pool(n_threads);

    pool.run([&](int tid) {
        int chunk   = (total + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total);

        for (int task = t_start; task < t_end; task++) {
            int b  = task / nc;
            int ni = task % nc;

            const float* q_b    = q_ptr  + b  * n_heads * hd;
            const float* kv_ni  = kv_ptr + b  * nc      * hd + ni * hd;
            const float* w_b    = w_ptr  + b  * n_heads;
            float&       s_out  = s_ptr[b * nc + ni];

            float s = 0.0f;
            for (int h = 0; h < n_heads; h++) {
                const float* q_h = q_b + h * hd;
                __m512 acc = _mm512_setzero_ps();
                for (int d = 0; d < hd; d += 16) {
                    __m512 qv = _mm512_loadu_ps(q_h   + d);
                    __m512 kv = _mm512_loadu_ps(kv_ni + d);
                    acc = _mm512_fmadd_ps(qv, kv, acc);
                }
                float dot = hsum_avx512(acc);
                s += (dot > 0.0f ? dot : 0.0f) * w_b[h];
            }
            s_out = s;
        }
    });

    std::vector<int> idx(nc);
    for (int b = 0; b < bs; b++) {
        float* sb = s_ptr + b * nc;
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b2){ return sb[a] > sb[b2]; });
        int* ob = o_ptr + b * k;
        for (int i = 0; i < k; i++) ob[i] = idx[i];
    }

    return out;
}

void indexer_attn_cpu(
        torch::Tensor q,
        torch::Tensor kv_compress,
        torch::Tensor topk_idxs,
        float scale,
        torch::Tensor o_out,
        torch::Tensor lse_out,
        int n_threads = 0) {

    TORCH_CHECK(q.is_contiguous() && kv_compress.is_contiguous(), "must be contiguous");
    const int bs = q.size(0), n_heads = q.size(1), hd = q.size(2);
    const int k  = topk_idxs.size(1);

    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = (int)std::thread::hardware_concurrency();
#endif
    }

    const float*    q_ptr   = q.data_ptr<float>();
    const uint16_t* kv_ptr  = reinterpret_cast<const uint16_t*>(kv_compress.data_ptr<at::BFloat16>());
    const int*      idx_ptr = topk_idxs.data_ptr<int>();
    float*          o_ptr   = o_out.data_ptr<float>();
    float*          lse_ptr = lse_out.data_ptr<float>();
    const int nc_hist = kv_compress.size(1);

    ThreadPool& pool = get_pool(n_threads);
    pool.run([&](int tid) {
        int total = bs * n_heads;
        int chunk   = (total + n_threads - 1) / n_threads;
        int t_start = tid * chunk;
        int t_end   = std::min(t_start + chunk, total);

        for (int t = t_start; t < t_end; t++) {
            int b = t / n_heads;
            int h = t % n_heads;

            const float*    q_row  = q_ptr   + (b * n_heads + h) * hd;
            const int*      idx_b  = idx_ptr + b * k;
            float*          o_row  = o_ptr   + (b * n_heads + h) * hd;
            float*          lse_r  = lse_ptr + (b * n_heads + h);
            const uint16_t* kv_b   = kv_ptr  + b * nc_hist * hd;

            float max_val = -std::numeric_limits<float>::infinity();
            float sum_val = 0.0f;
            for (int d = 0; d < hd; d++) o_row[d] = 0.0f;

            for (int i = 0; i < k; i++) {
                int ni = idx_b[i];
                const uint16_t* kv_ni = kv_b + ni * hd;
                if (i + PREFETCH_DIST < k)
                    prefetch_kv_bf16(kv_b + idx_b[i + PREFETCH_DIST] * hd, hd);
                float s = dot_fp32_bf16(q_row, kv_ni, hd) * scale;
                float nm = s > max_val ? s : max_val;
                float eo = std::exp(max_val - nm), en = std::exp(s - nm);
                sum_val = eo * sum_val + en; max_val = nm;
                update_bf16(o_row, kv_ni, eo, en, hd);
            }
            normalize_fp32(o_row, sum_val, hd);
            *lse_r = max_val + std::log(sum_val);
        }
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_csa_attn", &fused_csa_attn,
          "DS4 CSA CPU attention: Q@KV + online softmax (AVX512, fp32 Q, bf16 KV)",
          py::arg("q"), py::arg("kv"), py::arg("scale"),
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_range", &fused_csa_attn_range,
          "Single-threaded CSA attention over kv[kv_s:kv_e]. No OMP — caller parallelizes via Python ThreadPoolExecutor.",
          py::arg("q"), py::arg("kv"), py::arg("scale"), py::arg("kv_s"), py::arg("kv_e"),
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_pool", &fused_csa_attn_pool,
          "CSA attention with persistent C++ thread pool (no OMP fork-join overhead)",
          py::arg("q"), py::arg("kv"), py::arg("scale"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_pool_bf16", &fused_csa_attn_pool_bf16,
          "CSA attention with bf16 Q input (persistent thread pool); D2H bandwidth halved vs fp32",
          py::arg("q"), py::arg("kv"), py::arg("scale"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_bsq", &fused_csa_attn_bsq,
          "CSA attention: task=(b,sq), KV loaded once per token shared across all heads (64x less KV reads)",
          py::arg("q"), py::arg("kv"), py::arg("scale"),
          py::arg("o_out"), py::arg("lse_out"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_bsq_bf16", &fused_csa_attn_bsq_bf16,
          "CSA attention: bf16 Q input, no external bf16->fp32 conversion needed (saves D2H bandwidth)",
          py::arg("q"), py::arg("kv"), py::arg("scale"),
          py::arg("o_out"), py::arg("lse_out"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_2pass_bf16", &fused_csa_attn_2pass_bf16,
          "CSA attention 2-pass: pass1 find max, pass2 GEMV (eliminates online-softmax serial chain)",
          py::arg("q"), py::arg("kv"), py::arg("scale"),
          py::arg("o_out"), py::arg("lse_out"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_fast_bf16", &fused_csa_attn_fast_bf16,
          "CSA attention fast: 4-acc dot + direct bf16 KV read + SQ_BLOCK=2",
          py::arg("q"), py::arg("kv"), py::arg("scale"),
          py::arg("o_out"), py::arg("lse_out"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_bsq_int8", &fused_csa_attn_bsq_int8,
          "CSA attention: task=(b,sq), Q/K in-kernel int8 quantize, dot via AVX512 VNNI dpbusd",
          py::arg("q"), py::arg("kv"), py::arg("scale"),
          py::arg("o_out"), py::arg("lse_out"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("kv_quantize_bf16_to_int8", &kv_quantize_bf16_to_int8,
          "Offline quantize KV bf16[bs,nc,hd] → (int8, scales[bs,nc], sum_k[bs,nc])",
          py::arg("kv"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_bsq_int8_preq", &fused_csa_attn_bsq_int8_preq,
          "CSA attention: preq-quantized KV (int8+scales+sums), Q quantized in-kernel, VNNI dot",
          py::arg("q"), py::arg("kv_i8"), py::arg("kv_scales"), py::arg("kv_sums"),
          py::arg("scale"), py::arg("o_out"), py::arg("lse_out"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("q_quantize_fp32_to_u8", &q_quantize_fp32_to_u8,
          "Offline quantize Q fp32[bs,nh,sq,hd] → (uint8, scales[bs,nh,sq])",
          py::arg("q"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_bsq_int8_full", &fused_csa_attn_bsq_int8_full,
          "CSA attention: pre-quantized Q (uint8) + KV (int8 dot + bf16 update), pure VNNI kernel",
          py::arg("q_u8"), py::arg("q_scales"), py::arg("kv_i8"), py::arg("kv_scales"), py::arg("kv_sums"),
          py::arg("kv_bf16"), py::arg("scale"), py::arg("o_out"), py::arg("lse_out"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_csa_attn_inplace", &fused_csa_attn_inplace,
          "CSA attention in-place: caller pre-allocates o_out/lse_out to avoid page fault overhead",
          py::arg("q"), py::arg("kv"), py::arg("scale"),
          py::arg("o_out"), py::arg("lse_out"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("indexer_topk_cpu", &indexer_topk_cpu,
          "Indexer scoring on CPU: weighted relu(q·kv) aggregated over heads, returns topk indices",
          py::arg("q_idx"), py::arg("kv_idx"), py::arg("weights"), py::arg("k"),
          py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("indexer_attn_cpu", &indexer_attn_cpu,
          "Gather topk KV from compress buf + online softmax attention, inplace output",
          py::arg("q"), py::arg("kv_compress"), py::arg("topk_idxs"), py::arg("scale"),
          py::arg("o_out"), py::arg("lse_out"), py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("indexer_topk_cpu_par", &indexer_topk_cpu_par,
          "Indexer scoring: (b×nc) two-level parallel — all 48 threads active even at small bs",
          py::arg("q_idx"), py::arg("kv_idx"), py::arg("weights"), py::arg("k"),
          py::arg("n_threads") = 0,
          py::call_guard<py::gil_scoped_release>());
    m.def("set_num_threads", [](int n) {
#ifdef _OPENMP
        omp_set_num_threads(n);
#endif
    });
    m.def("get_num_threads", []() -> int {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    });
}

