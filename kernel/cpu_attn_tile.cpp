
#include <torch/extension.h>
#include <immintrin.h>
#include <cmath>
#include <limits>
#include <tuple>
#include <cstdint>
#include <vector>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

static inline float dot_fp32(const float* __restrict__ a,
                              const float* __restrict__ b,
                              int head_dim) {
    __m512 acc = _mm512_setzero_ps();
    for (int d = 0; d < head_dim; d += 16) {
        __m512 av = _mm512_loadu_ps(a + d);
        __m512 bv = _mm512_loadu_ps(b + d);
        acc = _mm512_fmadd_ps(av, bv, acc);
    }
    return hsum_avx512(acc);
}

static inline float dot_bf16(const float* __restrict__ q_fp32,
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

static inline void dot8_bf16_q(const uint16_t* __restrict__ q_bf16,
                               const uint16_t* __restrict__ k0,
                               const uint16_t* __restrict__ k1,
                               const uint16_t* __restrict__ k2,
                               const uint16_t* __restrict__ k3,
                               const uint16_t* __restrict__ k4,
                               const uint16_t* __restrict__ k5,
                               const uint16_t* __restrict__ k6,
                               const uint16_t* __restrict__ k7,
                               float scale, float* s, int head_dim) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();

    for (int d = 0; d < head_dim; d += 16) {
        __m256i q16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_bf16 + d));
        __m512i q32i = _mm512_cvtepu16_epi32(q16);
        q32i = _mm512_slli_epi32(q32i, 16);
        __m512 qv = _mm512_castsi512_ps(q32i);
        auto widen = [d](const uint16_t* kp) {
            __m256i k16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kp + d));
            __m512i k32i = _mm512_cvtepu16_epi32(k16);
            k32i = _mm512_slli_epi32(k32i, 16);
            return _mm512_castsi512_ps(k32i);
        };
        acc0 = _mm512_fmadd_ps(qv, widen(k0), acc0);
        acc1 = _mm512_fmadd_ps(qv, widen(k1), acc1);
        acc2 = _mm512_fmadd_ps(qv, widen(k2), acc2);
        acc3 = _mm512_fmadd_ps(qv, widen(k3), acc3);
        acc4 = _mm512_fmadd_ps(qv, widen(k4), acc4);
        acc5 = _mm512_fmadd_ps(qv, widen(k5), acc5);
        acc6 = _mm512_fmadd_ps(qv, widen(k6), acc6);
        acc7 = _mm512_fmadd_ps(qv, widen(k7), acc7);
    }
    s[0] = hsum_avx512(acc0) * scale;
    s[1] = hsum_avx512(acc1) * scale;
    s[2] = hsum_avx512(acc2) * scale;
    s[3] = hsum_avx512(acc3) * scale;
    s[4] = hsum_avx512(acc4) * scale;
    s[5] = hsum_avx512(acc5) * scale;
    s[6] = hsum_avx512(acc6) * scale;
    s[7] = hsum_avx512(acc7) * scale;
}

static inline float dot_bf16_q(const uint16_t* __restrict__ q_bf16,
                                const uint16_t* __restrict__ k_bf16,
                                int head_dim) {
    __m512 acc = _mm512_setzero_ps();
    for (int d = 0; d < head_dim; d += 16) {
        __m256i q16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_bf16 + d));
        __m512i q32i = _mm512_cvtepu16_epi32(q16);
        q32i = _mm512_slli_epi32(q32i, 16);
        __m512 qv = _mm512_castsi512_ps(q32i);
        __m256i k16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_bf16 + d));
        __m512i k32i = _mm512_cvtepu16_epi32(k16);
        k32i = _mm512_slli_epi32(k32i, 16);
        __m512 kv = _mm512_castsi512_ps(k32i);
        acc = _mm512_fmadd_ps(qv, kv, acc);
    }
    return hsum_avx512(acc);
}

static inline void update_fp32(float* __restrict__ out_g,
                                const float* __restrict__ v_ptr,
                                float exp_old, float exp_new,
                                int head_dim) {
    __m512 eo = _mm512_set1_ps(exp_old);
    __m512 en = _mm512_set1_ps(exp_new);
    for (int d = 0; d < head_dim; d += 16) {
        __m512 o  = _mm512_loadu_ps(out_g + d);
        __m512 vv = _mm512_loadu_ps(v_ptr + d);
        _mm512_storeu_ps(out_g + d, _mm512_fmadd_ps(eo, o, _mm512_mul_ps(en, vv)));
    }
}

static inline void update8_bf16_reg(float* __restrict__ og,
                                    const uint16_t* v0, const uint16_t* v1,
                                    const uint16_t* v2, const uint16_t* v3,
                                    const uint16_t* v4, const uint16_t* v5,
                                    const uint16_t* v6, const uint16_t* v7,
                                    float eo, const float* e) {
    __m512 ov[8];
    __m512 eo_v = _mm512_set1_ps(eo);
    for (int j = 0; j < 8; j++) ov[j] = _mm512_mul_ps(_mm512_loadu_ps(og + 16*j), eo_v);
    const uint16_t* vv[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    for (int r = 0; r < 8; r++) {
        __m512 en = _mm512_set1_ps(e[r]);
        const uint16_t* vp = vv[r];
        for (int j = 0; j < 8; j++) {
            __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vp + 16*j));
            __m512i v32i = _mm512_cvtepu16_epi32(v16);
            v32i = _mm512_slli_epi32(v32i, 16);
            ov[j] = _mm512_fmadd_ps(en, _mm512_castsi512_ps(v32i), ov[j]);
        }
    }
    for (int j = 0; j < 8; j++) _mm512_storeu_ps(og + 16*j, ov[j]);
}

static inline void update_add_bf16(float* __restrict__ out_g,
                                     const uint16_t* __restrict__ v_bf16,
                                     float exp_new,
                                     int head_dim) {
    __m512 en = _mm512_set1_ps(exp_new);
    for (int d = 0; d < head_dim; d += 16) {
        __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v_bf16 + d));
        __m512i v32i = _mm512_cvtepu16_epi32(v16);
        v32i = _mm512_slli_epi32(v32i, 16);
        __m512 vv = _mm512_castsi512_ps(v32i);
        __m512 o = _mm512_loadu_ps(out_g + d);
        _mm512_storeu_ps(out_g + d, _mm512_fmadd_ps(en, vv, o));
    }
}

static inline void update_bf16(float* __restrict__ out_g,
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
        __m512 o  = _mm512_loadu_ps(out_g + d);
        _mm512_storeu_ps(out_g + d, _mm512_fmadd_ps(eo, o, _mm512_mul_ps(en, vv)));
    }
}

static inline void normalize_fp32(float* __restrict__ out_g,
                                   float sum_g, int head_dim) {
    __m512 inv = _mm512_set1_ps(1.0f / sum_g);
    for (int d = 0; d < head_dim; d += 16) {
        __m512 o = _mm512_loadu_ps(out_g + d);
        _mm512_storeu_ps(out_g + d, _mm512_mul_ps(o, inv));
    }
}

static constexpr int PREFETCH_DIST = 4;

static inline void prefetch_kv_bf16(const uint16_t* k, const uint16_t* v, int head_dim) {
    for (int d = 0; d < head_dim * 2; d += 64)
        _mm_prefetch(reinterpret_cast<const char*>(k) + d, _MM_HINT_T1);
    for (int d = 0; d < head_dim * 2; d += 64)
        _mm_prefetch(reinterpret_cast<const char*>(v) + d, _MM_HINT_T1);
}

template<bool KV_BF16, bool Q_BF16, bool Normalize = true>
static void run_kernel(
        const void*     q_ptr_raw,
        const void*     k_ptr_raw,
        const void*     v_ptr_raw,
        float*          out_ptr,
        float*          max_ptr,
        float*          sum_ptr,
        int B_kv, int groups, int head_dim, int v_head_dim, int cpu_seq,
        int64_t k_stride_b, int64_t k_stride_i,
        int64_t v_stride_b, int64_t v_stride_i,
        float scale) {

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    int target_tiles = 1;
    if (B_kv < n_threads) {
        target_tiles = (n_threads + B_kv - 1) / B_kv;
        target_tiles = std::max(target_tiles, 2);
    }

    int seq_tile_size = (cpu_seq + target_tiles - 1) / target_tiles;
    seq_tile_size = ((seq_tile_size + 15) / 16) * 16;
    seq_tile_size = std::max(seq_tile_size, 256);
    seq_tile_size = std::min(seq_tile_size, cpu_seq);

    const int n_tiles = (cpu_seq + seq_tile_size - 1) / seq_tile_size;
    const int total_tasks = B_kv * n_tiles;

    const int q_stride_b   = groups * head_dim;
    const int out_stride_b = groups * v_head_dim;

    if (n_tiles == 1) {

        #pragma omp parallel for schedule(static)
        for (int b = 0; b < B_kv; b++) {

            const float*    q_b_fp32 = Q_BF16 ? nullptr :
                reinterpret_cast<const float*>(q_ptr_raw) + b * q_stride_b;
            const uint16_t* q_b_bf16 = Q_BF16 ?
                reinterpret_cast<const uint16_t*>(q_ptr_raw) + b * q_stride_b : nullptr;

            float*       out_b = out_ptr + b * out_stride_b;
            float*       max_b = max_ptr + b * groups;
            float*       sum_b = sum_ptr + b * groups;

            for (int g = 0; g < groups; g++) {
                max_b[g] = -std::numeric_limits<float>::infinity();
                sum_b[g] = 0.0f;
                float* og = out_b + g * v_head_dim;
                for (int d = 0; d < v_head_dim; d++) og[d] = 0.0f;
            }

            int i = 0;
            if constexpr (KV_BF16) {
                const uint16_t* kp = reinterpret_cast<const uint16_t*>(k_ptr_raw);
                const uint16_t* vp = reinterpret_cast<const uint16_t*>(v_ptr_raw);
                for (; i + 7 < cpu_seq; i += 8) {
                    const uint16_t* k0 = kp + b * k_stride_b + (i + 0) * k_stride_i;
                    const uint16_t* k1 = kp + b * k_stride_b + (i + 1) * k_stride_i;
                    const uint16_t* k2 = kp + b * k_stride_b + (i + 2) * k_stride_i;
                    const uint16_t* k3 = kp + b * k_stride_b + (i + 3) * k_stride_i;
                    const uint16_t* k4 = kp + b * k_stride_b + (i + 4) * k_stride_i;
                    const uint16_t* k5 = kp + b * k_stride_b + (i + 5) * k_stride_i;
                    const uint16_t* k6 = kp + b * k_stride_b + (i + 6) * k_stride_i;
                    const uint16_t* k7 = kp + b * k_stride_b + (i + 7) * k_stride_i;
                    const uint16_t* v0 = vp + b * v_stride_b + (i + 0) * v_stride_i;
                    const uint16_t* v1 = vp + b * v_stride_b + (i + 1) * v_stride_i;
                    const uint16_t* v2 = vp + b * v_stride_b + (i + 2) * v_stride_i;
                    const uint16_t* v3 = vp + b * v_stride_b + (i + 3) * v_stride_i;
                    const uint16_t* v4 = vp + b * v_stride_b + (i + 4) * v_stride_i;
                    const uint16_t* v5 = vp + b * v_stride_b + (i + 5) * v_stride_i;
                    const uint16_t* v6 = vp + b * v_stride_b + (i + 6) * v_stride_i;
                    const uint16_t* v7 = vp + b * v_stride_b + (i + 7) * v_stride_i;
                    if (i + PREFETCH_DIST + 7 < cpu_seq) {
                        for (int r = 0; r < 8; r++) {
                            prefetch_kv_bf16(kp + b * k_stride_b + (i + PREFETCH_DIST + r) * k_stride_i,
                                             vp + b * v_stride_b + (i + PREFETCH_DIST + r) * v_stride_i, head_dim);
                        }
                    }
                    for (int g = 0; g < groups; g++) {
                        const uint16_t* qg = q_b_bf16 + g * head_dim;
                        float* og = out_b + g * v_head_dim;
                        float s[8];
                        dot8_bf16_q(qg, k0, k1, k2, k3, k4, k5, k6, k7, scale, s, head_dim);

                        float om = max_b[g];
                        float nm = om;
                        for (int r = 0; r < 8; r++) nm = (s[r] > nm) ? s[r] : nm;
                        float eo = std::exp(om - nm);
                        float e[8], se = 0.0f;
                        for (int r = 0; r < 8; r++) { e[r] = std::exp(s[r] - nm); se += e[r]; }
                        sum_b[g] = eo * sum_b[g] + se;
                        max_b[g] = nm;

                        if (v_head_dim == 128) {
                            update8_bf16_reg(og, v0, v1, v2, v3, v4, v5, v6, v7, eo, e);
                        } else {
                            for (int j = 0; j < v_head_dim; j++) og[j] *= eo;
                            const uint16_t* vv[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
                            for (int r = 0; r < 8; r++)
                                update_add_bf16(og, vv[r], e[r], v_head_dim);
                        }
                    }
                }
                for (; i + 1 < cpu_seq; i += 2) {
                    const uint16_t* k0 = kp + b * k_stride_b + i       * k_stride_i;
                    const uint16_t* k1 = kp + b * k_stride_b + (i + 1) * k_stride_i;
                    const uint16_t* v0 = vp + b * v_stride_b + i       * v_stride_i;
                    const uint16_t* v1 = vp + b * v_stride_b + (i + 1) * v_stride_i;
                    if (i + PREFETCH_DIST + 1 < cpu_seq) {
                        prefetch_kv_bf16(kp + b * k_stride_b + (i + PREFETCH_DIST)     * k_stride_i,
                                         vp + b * v_stride_b + (i + PREFETCH_DIST)     * v_stride_i, head_dim);
                        prefetch_kv_bf16(kp + b * k_stride_b + (i + PREFETCH_DIST + 1) * k_stride_i,
                                         vp + b * v_stride_b + (i + PREFETCH_DIST + 1) * v_stride_i, head_dim);
                    }
                    for (int g = 0; g < groups; g++) {
                        const uint16_t* qg = q_b_bf16 + g * head_dim;
                        float* og = out_b + g * v_head_dim;
                        float s0 = dot_bf16_q(qg, k0, head_dim) * scale;
                        float s1 = dot_bf16_q(qg, k1, head_dim) * scale;

                        float om = max_b[g];
                        float nm = (s0 > om) ? s0 : om;
                        float eo = std::exp(om - nm), en = std::exp(s0 - nm);
                        sum_b[g] = eo * sum_b[g] + en;
                        max_b[g] = nm;
                        update_bf16(og, v0, eo, en, v_head_dim);

                        om = max_b[g];
                        nm = (s1 > om) ? s1 : om;
                        eo = std::exp(om - nm); en = std::exp(s1 - nm);
                        sum_b[g] = eo * sum_b[g] + en;
                        max_b[g] = nm;
                        update_bf16(og, v1, eo, en, v_head_dim);
                    }
                }

                for (; i < cpu_seq; i++) {
                    const uint16_t* ki = kp + b * k_stride_b + i * k_stride_i;
                    const uint16_t* vi = vp + b * v_stride_b + i * v_stride_i;
                    for (int g = 0; g < groups; g++) {
                        const uint16_t* qg = q_b_bf16 + g * head_dim;
                        float* og = out_b + g * v_head_dim;
                        float score = dot_bf16_q(qg, ki, head_dim) * scale;
                        float om = max_b[g], nm = (score > om) ? score : om;
                        float eo = std::exp(om - nm), en = std::exp(score - nm);
                        sum_b[g] = eo * sum_b[g] + en;
                        max_b[g] = nm;
                        update_bf16(og, vi, eo, en, v_head_dim);
                    }
                }
            } else {

                const float*    kp_f = Q_BF16 ? nullptr : reinterpret_cast<const float*>(k_ptr_raw);
                const uint16_t* kp_b = Q_BF16 ? reinterpret_cast<const uint16_t*>(k_ptr_raw) : nullptr;
                const float*    vp_f = Q_BF16 ? nullptr : reinterpret_cast<const float*>(v_ptr_raw);
                const uint16_t* vp_b = Q_BF16 ? reinterpret_cast<const uint16_t*>(v_ptr_raw) : nullptr;
                for (; i < cpu_seq; i++) {
                    const float*    k_i_fp32 = kp_f ? kp_f + b * k_stride_b + i * k_stride_i : nullptr;
                    const uint16_t* k_i_bf16 = kp_b ? kp_b + b * k_stride_b + i * k_stride_i : nullptr;
                    const float*    v_i_fp32 = vp_f ? vp_f + b * v_stride_b + i * v_stride_i : nullptr;
                    const uint16_t* v_i_bf16 = vp_b ? vp_b + b * v_stride_b + i * v_stride_i : nullptr;
                    if constexpr (KV_BF16) {
                        if (i + PREFETCH_DIST < cpu_seq)
                            prefetch_kv_bf16(kp_b + b * k_stride_b + (i + PREFETCH_DIST) * k_stride_i,
                                             vp_b + b * v_stride_b + (i + PREFETCH_DIST) * v_stride_i, head_dim);
                    }
                    for (int g = 0; g < groups; g++) {
                        const float*    q_g_fp32 = Q_BF16 ? nullptr : (q_b_fp32 + g * head_dim);
                        const uint16_t* q_g_bf16 = Q_BF16 ? (q_b_bf16 + g * head_dim) : nullptr;
                        float* og = out_b + g * v_head_dim;
                        float score;
                        if constexpr (KV_BF16)
                            score = dot_bf16(q_g_fp32, k_i_bf16, head_dim) * scale;
                        else
                            score = dot_fp32(q_g_fp32, k_i_fp32, head_dim) * scale;
                        float om = max_b[g], nm = (score > om) ? score : om;
                        float eo = std::exp(om - nm), en = std::exp(score - nm);
                        sum_b[g] = eo * sum_b[g] + en;
                        max_b[g] = nm;
                        if constexpr (KV_BF16)
                            update_bf16(og, v_i_bf16, eo, en, v_head_dim);
                        else
                            update_fp32(og, v_i_fp32, eo, en, v_head_dim);
                    }
                }
            }
            for (int g = 0; g < groups; g++)
                if constexpr (Normalize)
                    normalize_fp32(out_b + g * v_head_dim, sum_b[g], v_head_dim);
        }
        return;
    }

    std::vector<float> tile_out(B_kv * n_tiles * groups * v_head_dim, 0.0f);
    std::vector<float> tile_max(B_kv * n_tiles * groups,
                                -std::numeric_limits<float>::infinity());
    std::vector<float> tile_sum(B_kv * n_tiles * groups, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int task = 0; task < total_tasks; task++) {
        const int b    = task / n_tiles;
        const int tile = task % n_tiles;
        const int seq_start = tile * seq_tile_size;
        const int seq_end   = std::min(seq_start + seq_tile_size, cpu_seq);
        const int tile_len  = seq_end - seq_start;

        const float*    q_b_fp32 = Q_BF16 ? nullptr :
            reinterpret_cast<const float*>(q_ptr_raw) + b * q_stride_b;
        const uint16_t* q_b_bf16 = Q_BF16 ?
            reinterpret_cast<const uint16_t*>(q_ptr_raw) + b * q_stride_b : nullptr;

        float* t_out = tile_out.data() + (b * n_tiles + tile) * groups * v_head_dim;
        float* t_max = tile_max.data() + (b * n_tiles + tile) * groups;
        float* t_sum = tile_sum.data() + (b * n_tiles + tile) * groups;

        for (int g = 0; g < groups; g++) {
            t_max[g] = -std::numeric_limits<float>::infinity();
            t_sum[g] = 0.0f;
            float* og = t_out + g * v_head_dim;
            for (int d = 0; d < v_head_dim; d++) og[d] = 0.0f;
        }

        int i = seq_start;
        if constexpr (KV_BF16) {
            const uint16_t* kp = reinterpret_cast<const uint16_t*>(k_ptr_raw);
            const uint16_t* vp = reinterpret_cast<const uint16_t*>(v_ptr_raw);
            for (; i + 7 < seq_end; i += 8) {
                const uint16_t* k0 = kp + b * k_stride_b + (i + 0) * k_stride_i;
                const uint16_t* k1 = kp + b * k_stride_b + (i + 1) * k_stride_i;
                const uint16_t* k2 = kp + b * k_stride_b + (i + 2) * k_stride_i;
                const uint16_t* k3 = kp + b * k_stride_b + (i + 3) * k_stride_i;
                const uint16_t* k4 = kp + b * k_stride_b + (i + 4) * k_stride_i;
                const uint16_t* k5 = kp + b * k_stride_b + (i + 5) * k_stride_i;
                const uint16_t* k6 = kp + b * k_stride_b + (i + 6) * k_stride_i;
                const uint16_t* k7 = kp + b * k_stride_b + (i + 7) * k_stride_i;
                const uint16_t* v0 = vp + b * v_stride_b + (i + 0) * v_stride_i;
                const uint16_t* v1 = vp + b * v_stride_b + (i + 1) * v_stride_i;
                const uint16_t* v2 = vp + b * v_stride_b + (i + 2) * v_stride_i;
                const uint16_t* v3 = vp + b * v_stride_b + (i + 3) * v_stride_i;
                const uint16_t* v4 = vp + b * v_stride_b + (i + 4) * v_stride_i;
                const uint16_t* v5 = vp + b * v_stride_b + (i + 5) * v_stride_i;
                const uint16_t* v6 = vp + b * v_stride_b + (i + 6) * v_stride_i;
                const uint16_t* v7 = vp + b * v_stride_b + (i + 7) * v_stride_i;
                if (i + PREFETCH_DIST + 7 < seq_end) {
                    for (int r = 0; r < 8; r++) {
                        prefetch_kv_bf16(kp + b * k_stride_b + (i + PREFETCH_DIST + r) * k_stride_i,
                                         vp + b * v_stride_b + (i + PREFETCH_DIST + r) * v_stride_i, head_dim);
                    }
                }
                for (int g = 0; g < groups; g++) {
                    const uint16_t* qg = q_b_bf16 + g * head_dim;
                    float*          og = t_out + g * v_head_dim;
                    float s[8];
                    dot8_bf16_q(qg, k0, k1, k2, k3, k4, k5, k6, k7, scale, s, head_dim);
                    float om = t_max[g];
                    float nm = om;
                    for (int r = 0; r < 8; r++) nm = (s[r] > nm) ? s[r] : nm;
                    float eo = std::exp(om - nm);
                    float e[8], se = 0.0f;
                    for (int r = 0; r < 8; r++) { e[r] = std::exp(s[r] - nm); se += e[r]; }
                    t_sum[g] = eo * t_sum[g] + se;
                    t_max[g] = nm;
                    if (v_head_dim == 128) {
                        update8_bf16_reg(og, v0, v1, v2, v3, v4, v5, v6, v7, eo, e);
                    } else {
                        for (int j = 0; j < v_head_dim; j++) og[j] *= eo;
                        const uint16_t* vv[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
                        for (int r = 0; r < 8; r++)
                            update_add_bf16(og, vv[r], e[r], v_head_dim);
                    }
                }
            }
            for (; i + 1 < seq_end; i += 2) {
                const uint16_t* k0 = kp + b * k_stride_b + i       * k_stride_i;
                const uint16_t* k1 = kp + b * k_stride_b + (i + 1) * k_stride_i;
                const uint16_t* v0 = vp + b * v_stride_b + i       * v_stride_i;
                const uint16_t* v1 = vp + b * v_stride_b + (i + 1) * v_stride_i;
                if (i + PREFETCH_DIST + 1 < seq_end) {
                    prefetch_kv_bf16(kp + b * k_stride_b + (i + PREFETCH_DIST)     * k_stride_i,
                                     vp + b * v_stride_b + (i + PREFETCH_DIST)     * v_stride_i, head_dim);
                    prefetch_kv_bf16(kp + b * k_stride_b + (i + PREFETCH_DIST + 1) * k_stride_i,
                                     vp + b * v_stride_b + (i + PREFETCH_DIST + 1) * v_stride_i, head_dim);
                }
                for (int g = 0; g < groups; g++) {
                    const uint16_t* qg = q_b_bf16 + g * head_dim;
                    float*          og = t_out + g * v_head_dim;
                    float s0 = dot_bf16_q(qg, k0, head_dim) * scale;
                    float s1 = dot_bf16_q(qg, k1, head_dim) * scale;
                    float om = t_max[g];
                    float nm = (s0 > om) ? s0 : om;
                    float eo = std::exp(om - nm), en = std::exp(s0 - nm);
                    t_sum[g] = eo * t_sum[g] + en;
                    t_max[g] = nm;
                    update_bf16(og, v0, eo, en, v_head_dim);
                    om = t_max[g];
                    nm = (s1 > om) ? s1 : om;
                    eo = std::exp(om - nm); en = std::exp(s1 - nm);
                    t_sum[g] = eo * t_sum[g] + en;
                    t_max[g] = nm;
                    update_bf16(og, v1, eo, en, v_head_dim);
                }
            }
            for (; i < seq_end; i++) {
                const uint16_t* ki = kp + b * k_stride_b + i * k_stride_i;
                const uint16_t* vi = vp + b * v_stride_b + i * v_stride_i;
                for (int g = 0; g < groups; g++) {
                    const uint16_t* qg = q_b_bf16 + g * head_dim;
                    float*          og = t_out + g * v_head_dim;
                    float score = dot_bf16_q(qg, ki, head_dim) * scale;
                    float om = t_max[g], nm = (score > om) ? score : om;
                    float eo = std::exp(om - nm), en = std::exp(score - nm);
                    t_sum[g] = eo * t_sum[g] + en;
                    t_max[g] = nm;
                    update_bf16(og, vi, eo, en, v_head_dim);
                }
            }
        } else {
            for (; i < seq_end; i++) {
                const float*    k_i_fp32 = reinterpret_cast<const float*>(k_ptr_raw) + b * k_stride_b + i * k_stride_i;
                const float*    v_i_fp32 = reinterpret_cast<const float*>(v_ptr_raw) + b * v_stride_b + i * v_stride_i;
                for (int g = 0; g < groups; g++) {
                    const float*    q_g_fp32 = Q_BF16 ? nullptr : (q_b_fp32 + g * head_dim);
                    float*          og       = t_out + g * v_head_dim;
                    float score = dot_fp32(q_g_fp32, k_i_fp32, head_dim) * scale;
                    float om = t_max[g], nm = (score > om) ? score : om;
                    float eo = std::exp(om - nm), en = std::exp(score - nm);
                    t_sum[g] = eo * t_sum[g] + en;
                    t_max[g] = nm;
                    update_fp32(og, v_i_fp32, eo, en, v_head_dim);
                }
            }
        }

    }

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B_kv; b++) {
        float* out_b = out_ptr + b * out_stride_b;
        float* max_b = max_ptr + b * groups;
        float* sum_b = sum_ptr + b * groups;

        for (int g = 0; g < groups; g++) {
            float global_max = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < n_tiles; t++)
                global_max = std::max(global_max,
                                      tile_max[(b * n_tiles + t) * groups + g]);

            float global_sum = 0.0f;
            float* og = out_b + g * v_head_dim;
            for (int d = 0; d < v_head_dim; d++) og[d] = 0.0f;

            for (int t = 0; t < n_tiles; t++) {
                float t_m = tile_max[(b * n_tiles + t) * groups + g];
                float t_s = tile_sum[(b * n_tiles + t) * groups + g];
                float* t_o = tile_out.data() + (b * n_tiles + t) * groups * v_head_dim
                             + g * v_head_dim;
                float w = std::exp(t_m - global_max);
                global_sum += w * t_s;
                __m512 wv = _mm512_set1_ps(w);
                for (int d = 0; d < v_head_dim; d += 16) {
                    __m512 o  = _mm512_loadu_ps(og + d);
                    __m512 to = _mm512_loadu_ps(t_o + d);
                    _mm512_storeu_ps(og + d, _mm512_fmadd_ps(wv, to, o));
                }
            }

            if constexpr (Normalize)
                normalize_fp32(og, global_sum, v_head_dim);
            max_b[g] = global_max;
            sum_b[g] = global_sum;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_cpu_attn(const torch::Tensor& q,
               const torch::Tensor& k,
               const torch::Tensor& v,
               float scale) {

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kFloat32 || q.dtype() == torch::kBFloat16,
                "q must be float32 or bfloat16");
    TORCH_CHECK(k.dtype() == q.dtype(), "k dtype must match q");
    TORCH_CHECK(v.dtype() == q.dtype(), "v dtype must match q");
    TORCH_CHECK(q.size(2) % 16 == 0, "head_dim must be multiple of 16 for AVX512");

    const bool is_bf16 = (q.dtype() == torch::kBFloat16);

    const int B_kv     = q.size(0);
    const int groups   = q.size(1);
    const int head_dim = q.size(2);
    const int cpu_seq  = k.size(1);

    auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto out_fp32 = torch::zeros({B_kv, groups, head_dim}, fp32_opts);
    auto lse_max  = torch::full({B_kv, groups, 1},
                                -std::numeric_limits<float>::infinity(), fp32_opts);
    auto lse_sum  = torch::zeros({B_kv, groups, 1}, fp32_opts);

    float* out_ptr = out_fp32.data_ptr<float>();
    float* max_ptr = lse_max.data_ptr<float>();
    float* sum_ptr = lse_sum.data_ptr<float>();

    const int64_t k_stride_b = k.stride(0);
    const int64_t k_stride_i = k.stride(1);
    const int64_t v_stride_b = v.stride(0);
    const int64_t v_stride_i = v.stride(1);

    if (is_bf16) {

        run_kernel<true, true>(
            reinterpret_cast<const void*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(v.data_ptr<at::BFloat16>()),
            out_ptr, max_ptr, sum_ptr,
            B_kv, groups, head_dim, head_dim, cpu_seq,
            k_stride_b, k_stride_i, v_stride_b, v_stride_i,
            scale);
    } else {
        run_kernel<false, false>(
            reinterpret_cast<const void*>(q.data_ptr<float>()),
            reinterpret_cast<const void*>(k.data_ptr<float>()),
            reinterpret_cast<const void*>(v.data_ptr<float>()),
            out_ptr, max_ptr, sum_ptr,
            B_kv, groups, head_dim, head_dim, cpu_seq,
            k_stride_b, k_stride_i, v_stride_b, v_stride_i,
            scale);
    }

    lse_max = lse_max.view({B_kv, groups, 1});
    lse_sum = lse_sum.view({B_kv, groups, 1});

    if (is_bf16)
        return {out_fp32.to(torch::kBFloat16),
                lse_max.to(torch::kBFloat16),
                lse_sum.to(torch::kBFloat16)};
    return {out_fp32, lse_max, lse_sum};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_cpu_attn_mla(const torch::Tensor& q,
                   const torch::Tensor& k,
                   const torch::Tensor& v,
                   float scale) {

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "fused_cpu_attn_mla requires bf16 inputs");
    TORCH_CHECK(k.dtype() == torch::kBFloat16);
    TORCH_CHECK(v.dtype() == torch::kBFloat16);

    const int B_kv       = q.size(0);
    const int groups     = q.size(1);
    const int k_head_dim = q.size(2);
    const int v_head_dim = v.size(2);
    const int cpu_seq    = k.size(1);

    auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto out_fp32 = torch::zeros({B_kv, groups, v_head_dim}, fp32_opts);
    auto lse_max  = torch::full({B_kv, groups, 1},
                                -std::numeric_limits<float>::infinity(), fp32_opts);
    auto lse_sum  = torch::zeros({B_kv, groups, 1}, fp32_opts);

    run_kernel< true, true, false>(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        out_fp32.data_ptr<float>(), lse_max.data_ptr<float>(), lse_sum.data_ptr<float>(),
        B_kv, groups, k_head_dim, v_head_dim, cpu_seq,
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        scale);

    return {out_fp32,
            lse_max.view({B_kv, groups, 1}),
            lse_sum.view({B_kv, groups, 1})};
}

static void run_kernel_scalar_fp32(
        const float* q_ptr,
        const float* k_ptr,
        const float* v_ptr,
        float*       out_ptr,
        float*       max_ptr,
        float*       sum_ptr,
        int B_kv, int groups, int head_dim, int cpu_seq,
        int64_t k_stride_b, int64_t k_stride_i,
        int64_t v_stride_b, int64_t v_stride_i,
        float scale) {

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    int target_tiles = (4 * n_threads + B_kv - 1) / B_kv;
    target_tiles = std::max(target_tiles, 1);
    int seq_tile_size = (cpu_seq + target_tiles - 1) / target_tiles;
    seq_tile_size = ((seq_tile_size + 15) / 16) * 16;
    seq_tile_size = std::max(seq_tile_size, 256);
    seq_tile_size = std::min(seq_tile_size, cpu_seq);

    const int n_tiles      = (cpu_seq + seq_tile_size - 1) / seq_tile_size;
    const int total_tasks  = B_kv * n_tiles;
    const int q_stride_b   = groups * head_dim;
    const int out_stride_b = groups * head_dim;

    if (n_tiles == 1) {
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < B_kv; b++) {
            const float* q_b = q_ptr + b * q_stride_b;
            float*       ob  = out_ptr + b * out_stride_b;
            float*       mb  = max_ptr + b * groups;
            float*       sb  = sum_ptr + b * groups;
            for (int g = 0; g < groups; g++) {
                mb[g] = -std::numeric_limits<float>::infinity();
                sb[g] = 0.0f;
                float* og = ob + g * head_dim;
                for (int d = 0; d < head_dim; d++) og[d] = 0.0f;
            }
            for (int i = 0; i < cpu_seq; i++) {
                const float* ki = k_ptr + b * k_stride_b + i * k_stride_i;
                const float* vi = v_ptr + b * v_stride_b + i * v_stride_i;
                for (int g = 0; g < groups; g++) {
                    const float* qg = q_b + g * head_dim;
                    float*       og = ob  + g * head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) score += qg[d] * ki[d];
                    score *= scale;
                    float old_max = mb[g];
                    float new_max = (score > old_max) ? score : old_max;
                    float exp_old = std::exp(old_max - new_max);
                    float exp_new = std::exp(score   - new_max);
                    sb[g] = exp_old * sb[g] + exp_new;
                    mb[g] = new_max;
                    for (int d = 0; d < head_dim; d++)
                        og[d] = exp_old * og[d] + exp_new * vi[d];
                }
            }
            for (int g = 0; g < groups; g++) {
                float* og = ob + g * head_dim;
                float inv = 1.0f / sb[g];
                for (int d = 0; d < head_dim; d++) og[d] *= inv;
            }
        }
        return;
    }

    std::vector<float> tile_out(B_kv * n_tiles * groups * head_dim, 0.0f);
    std::vector<float> tile_max(B_kv * n_tiles * groups,
                                -std::numeric_limits<float>::infinity());
    std::vector<float> tile_sum(B_kv * n_tiles * groups, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int task = 0; task < total_tasks; task++) {
        const int b    = task / n_tiles;
        const int tile = task % n_tiles;
        const int seq_start = tile * seq_tile_size;
        const int seq_end   = std::min(seq_start + seq_tile_size, cpu_seq);

        const float* q_b = q_ptr + b * q_stride_b;
        float* t_out = tile_out.data() + (b * n_tiles + tile) * groups * head_dim;
        float* t_max = tile_max.data() + (b * n_tiles + tile) * groups;
        float* t_sum = tile_sum.data() + (b * n_tiles + tile) * groups;

        for (int g = 0; g < groups; g++) {
            t_max[g] = -std::numeric_limits<float>::infinity();
            t_sum[g] = 0.0f;
            float* og = t_out + g * head_dim;
            for (int d = 0; d < head_dim; d++) og[d] = 0.0f;
        }
        for (int i = seq_start; i < seq_end; i++) {
            const float* ki = k_ptr + b * k_stride_b + i * k_stride_i;
            const float* vi = v_ptr + b * v_stride_b + i * v_stride_i;
            for (int g = 0; g < groups; g++) {
                const float* qg = q_b + g * head_dim;
                float*       og = t_out + g * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) score += qg[d] * ki[d];
                score *= scale;
                float old_max = t_max[g];
                float new_max = (score > old_max) ? score : old_max;
                float exp_old = std::exp(old_max - new_max);
                float exp_new = std::exp(score   - new_max);
                t_sum[g] = exp_old * t_sum[g] + exp_new;
                t_max[g] = new_max;
                for (int d = 0; d < head_dim; d++)
                    og[d] = exp_old * og[d] + exp_new * vi[d];
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B_kv; b++) {
        float* ob = out_ptr + b * out_stride_b;
        float* mb = max_ptr + b * groups;
        float* sb = sum_ptr + b * groups;
        for (int g = 0; g < groups; g++) {
            float global_max = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < n_tiles; t++)
                global_max = std::max(global_max,
                                      tile_max[(b * n_tiles + t) * groups + g]);
            float global_sum = 0.0f;
            float* og = ob + g * head_dim;
            for (int d = 0; d < head_dim; d++) og[d] = 0.0f;
            for (int t = 0; t < n_tiles; t++) {
                float tm = tile_max[(b * n_tiles + t) * groups + g];
                float ts = tile_sum[(b * n_tiles + t) * groups + g];
                float* to = tile_out.data() + (b * n_tiles + t) * groups * head_dim
                            + g * head_dim;
                float w = std::exp(tm - global_max);
                global_sum += w * ts;
                for (int d = 0; d < head_dim; d++) og[d] += w * to[d];
            }
            float inv = 1.0f / global_sum;
            for (int d = 0; d < head_dim; d++) og[d] *= inv;
            mb[g] = global_max;
            sb[g] = global_sum;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_cpu_attn_fp32(const torch::Tensor& q,
                    const torch::Tensor& k,
                    const torch::Tensor& v,
                    float scale) {
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "fused_cpu_attn_fp32 requires float32 inputs");
    TORCH_CHECK(k.dtype() == torch::kFloat32, "fused_cpu_attn_fp32 requires float32 inputs");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "fused_cpu_attn_fp32 requires float32 inputs");

    const int B_kv     = q.size(0);
    const int groups   = q.size(1);
    const int head_dim = q.size(2);
    const int cpu_seq  = k.size(1);

    auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto out_fp32 = torch::zeros({B_kv, groups, head_dim}, fp32_opts);
    auto lse_max  = torch::full({B_kv, groups, 1},
                                -std::numeric_limits<float>::infinity(), fp32_opts);
    auto lse_sum  = torch::zeros({B_kv, groups, 1}, fp32_opts);

    run_kernel_scalar_fp32(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out_fp32.data_ptr<float>(), lse_max.data_ptr<float>(), lse_sum.data_ptr<float>(),
        B_kv, groups, head_dim, cpu_seq,
        k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        scale);

    return {out_fp32,
            lse_max.view({B_kv, groups, 1}),
            lse_sum.view({B_kv, groups, 1})};
}

static inline float hsum512(__m512 v) { return _mm512_reduce_add_ps(v); }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_cpu_attn_mla_compressed(const torch::Tensor& q_proj,
                              const torch::Tensor& q_pe,
                              const torch::Tensor& c_kv,
                              const torch::Tensor& k_pe,
                              float scale) {
    TORCH_CHECK(q_proj.is_contiguous() && c_kv.is_contiguous());
    const int B  = c_kv.size(0);
    const int nH = q_proj.size(0) / B;
    const int S  = c_kv.size(1);
    const int CLR = 512, QRD = 64;
    TORCH_CHECK(q_proj.size(1) == CLR && q_pe.size(1) == QRD && c_kv.size(2) == CLR && k_pe.size(2) == QRD);

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto out = torch::zeros({B * nH, CLR}, f32);
    auto lse_max = torch::full({B * nH}, -std::numeric_limits<float>::infinity(), f32);
    auto lse_sum = torch::zeros({B * nH}, f32);

    const uint16_t* qp0 = reinterpret_cast<const uint16_t*>(q_proj.data_ptr());
    const uint16_t* qpe0 = reinterpret_cast<const uint16_t*>(q_pe.data_ptr());
    const uint16_t* c0 = reinterpret_cast<const uint16_t*>(c_kv.data_ptr());
    const uint16_t* p0 = reinterpret_cast<const uint16_t*>(k_pe.data_ptr());

    auto widen = [](const uint16_t* p) -> __m512 {
        __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        __m512i v32i = _mm512_cvtepu16_epi32(v16);
        v32i = _mm512_slli_epi32(v32i, 16);
        return _mm512_castsi512_ps(v32i);
    };

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    const char* _tile_env = getenv("MLA_TILE");
    const int tile_len = _tile_env ? atoi(_tile_env)
                                   : std::max(1, (S + std::max(1, n_threads / B) - 1) / std::max(1, n_threads / B));
    const int n_tiles_act = (S + tile_len - 1) / tile_len;

    auto t_out = torch::zeros({B, n_tiles_act, nH, CLR}, f32);
    auto t_max = torch::full({B, n_tiles_act, nH}, -std::numeric_limits<float>::infinity(), f32);
    auto t_sum = torch::zeros({B, n_tiles_act, nH}, f32);

    #pragma omp parallel for schedule(static)
    for (int task = 0; task < B * n_tiles_act; task++) {
        const int b = task / n_tiles_act, tile = task % n_tiles_act;
        const int s0 = tile * tile_len, s1 = std::min(s0 + tile_len, S);
        const uint16_t* cb = c0 + (size_t)b * S * CLR;
        const uint16_t* pb = p0 + (size_t)b * S * QRD;
        float* t_out_b = t_out.data_ptr<float>() + ((size_t)(b * n_tiles_act + tile) * nH) * CLR;
        float* t_max_b = t_max.data_ptr<float>() + (b * n_tiles_act + tile) * nH;
        float* t_sum_b = t_sum.data_ptr<float>() + (b * n_tiles_act + tile) * nH;

        int i = s0;
        for (; i + 7 < s1; i += 8) {
            const uint16_t* c_base = cb + (size_t)i * CLR;
            const uint16_t* p_base = pb + (size_t)i * QRD;
            for (int h = 0; h < nH; h++) {
                const uint16_t* qph = qp0 + (size_t)(b * nH + h) * CLR;
                const uint16_t* qpeh = qpe0 + (size_t)(b * nH + h) * QRD;
                float* oa = t_out_b + h * CLR;

                __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
                __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
                __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
                __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();
                for (int d = 0; d < 32; d++) {
                    __m512 qv = widen(qph + 16 * d);
                    const uint16_t* cd = c_base + 16 * d;
                    acc0 = _mm512_fmadd_ps(qv, widen(cd), acc0);
                    acc1 = _mm512_fmadd_ps(qv, widen(cd + CLR), acc1);
                    acc2 = _mm512_fmadd_ps(qv, widen(cd + 2*CLR), acc2);
                    acc3 = _mm512_fmadd_ps(qv, widen(cd + 3*CLR), acc3);
                    acc4 = _mm512_fmadd_ps(qv, widen(cd + 4*CLR), acc4);
                    acc5 = _mm512_fmadd_ps(qv, widen(cd + 5*CLR), acc5);
                    acc6 = _mm512_fmadd_ps(qv, widen(cd + 6*CLR), acc6);
                    acc7 = _mm512_fmadd_ps(qv, widen(cd + 7*CLR), acc7);
                }
                for (int d = 0; d < 4; d++) {
                    __m512 qv = widen(qpeh + 16 * d);
                    const uint16_t* pd_ = p_base + 16 * d;
                    acc0 = _mm512_fmadd_ps(qv, widen(pd_), acc0);
                    acc1 = _mm512_fmadd_ps(qv, widen(pd_ + QRD), acc1);
                    acc2 = _mm512_fmadd_ps(qv, widen(pd_ + 2*QRD), acc2);
                    acc3 = _mm512_fmadd_ps(qv, widen(pd_ + 3*QRD), acc3);
                    acc4 = _mm512_fmadd_ps(qv, widen(pd_ + 4*QRD), acc4);
                    acc5 = _mm512_fmadd_ps(qv, widen(pd_ + 5*QRD), acc5);
                    acc6 = _mm512_fmadd_ps(qv, widen(pd_ + 6*QRD), acc6);
                    acc7 = _mm512_fmadd_ps(qv, widen(pd_ + 7*QRD), acc7);
                }
                float s[8];
                s[0] = hsum512(acc0) * scale; s[1] = hsum512(acc1) * scale;
                s[2] = hsum512(acc2) * scale; s[3] = hsum512(acc3) * scale;
                s[4] = hsum512(acc4) * scale; s[5] = hsum512(acc5) * scale;
                s[6] = hsum512(acc6) * scale; s[7] = hsum512(acc7) * scale;

                float om = t_max_b[h];
                float nm = om;
                for (int r = 0; r < 8; r++) nm = (s[r] > nm) ? s[r] : nm;
                float eo = std::exp(om - nm);
                float e[8], se = 0.0f;
                for (int r = 0; r < 8; r++) { e[r] = std::exp(s[r] - nm); se += e[r]; }
                t_sum_b[h] = eo * t_sum_b[h] + se;
                t_max_b[h] = nm;

                for (int d = 0; d < 32; d++) {
                    __m512 o = _mm512_loadu_ps(oa + 16 * d);
                    o = _mm512_mul_ps(o, _mm512_set1_ps(eo));
                    const uint16_t* cd = c_base + 16 * d;
                    o = _mm512_fmadd_ps(_mm512_set1_ps(e[0]), widen(cd), o);
                    o = _mm512_fmadd_ps(_mm512_set1_ps(e[1]), widen(cd + CLR), o);
                    o = _mm512_fmadd_ps(_mm512_set1_ps(e[2]), widen(cd + 2*CLR), o);
                    o = _mm512_fmadd_ps(_mm512_set1_ps(e[3]), widen(cd + 3*CLR), o);
                    o = _mm512_fmadd_ps(_mm512_set1_ps(e[4]), widen(cd + 4*CLR), o);
                    o = _mm512_fmadd_ps(_mm512_set1_ps(e[5]), widen(cd + 5*CLR), o);
                    o = _mm512_fmadd_ps(_mm512_set1_ps(e[6]), widen(cd + 6*CLR), o);
                    o = _mm512_fmadd_ps(_mm512_set1_ps(e[7]), widen(cd + 7*CLR), o);
                    _mm512_storeu_ps(oa + 16 * d, o);
                }
            }
        }

        for (; i < s1; i++) {
            const uint16_t* cd = cb + (size_t)i * CLR;
            const uint16_t* pd_ = pb + (size_t)i * QRD;
            for (int h = 0; h < nH; h++) {
                const uint16_t* qph = qp0 + (size_t)(b * nH + h) * CLR;
                const uint16_t* qpeh = qpe0 + (size_t)(b * nH + h) * QRD;
                float* oa = t_out_b + h * CLR;
                __m512 acc = _mm512_setzero_ps();
                for (int d = 0; d < 32; d++)
                    acc = _mm512_fmadd_ps(widen(qph + 16 * d), widen(cd + 16 * d), acc);
                for (int d = 0; d < 4; d++)
                    acc = _mm512_fmadd_ps(widen(qpeh + 16 * d), widen(pd_ + 16 * d), acc);
                float score = hsum512(acc) * scale;
                float om = t_max_b[h];
                float nm = (score > om) ? score : om;
                float eo = std::exp(om - nm), en = std::exp(score - nm);
                t_sum_b[h] = eo * t_sum_b[h] + en;
                t_max_b[h] = nm;
                for (int d = 0; d < 32; d++) {
                    __m512 o = _mm512_loadu_ps(oa + 16 * d);
                    o = _mm512_fmadd_ps(_mm512_set1_ps(en), widen(cd + 16 * d),
                                        _mm512_mul_ps(o, _mm512_set1_ps(eo)));
                    _mm512_storeu_ps(oa + 16 * d, o);
                }
            }
        }
    }

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < nH; h++) {
            float gmax = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < n_tiles_act; t++)
                gmax = std::max(gmax, t_max.data_ptr<float>()[(b * n_tiles_act + t) * nH + h]);
            float gsum = 0.0f;
            float* og = out.data_ptr<float>() + (b * nH + h) * CLR;
            for (int d = 0; d < CLR; d++) og[d] = 0.0f;
            for (int t = 0; t < n_tiles_act; t++) {
                float tm = t_max.data_ptr<float>()[(b * n_tiles_act + t) * nH + h];
                float ts = t_sum.data_ptr<float>()[(b * n_tiles_act + t) * nH + h];
                const float* to = t_out.data_ptr<float>() + ((size_t)(b * n_tiles_act + t) * nH + h) * CLR;
                float w = std::exp(tm - gmax);
                gsum += w * ts;
                for (int d = 0; d < CLR; d++) og[d] += w * to[d];
            }
            lse_max.data_ptr<float>()[b * nH + h] = gmax;
            lse_sum.data_ptr<float>()[b * nH + h] = gsum;
        }
    }
    return {out, lse_max, lse_sum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_cpu_attn", &fused_cpu_attn,
          "Fused CPU attention: Q@K + online softmax + e@V (AVX512, fp32/bf16, 2D parallel)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_cpu_attn_fp32", &fused_cpu_attn_fp32,
          "Fused CPU attention: scalar fp32 GEMV, no AVX512 intrinsics (ablation baseline)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_cpu_attn_mla", &fused_cpu_attn_mla,
          "Fused CPU attention for MLA: k_head_dim != v_head_dim (AVX512 bf16)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
          py::call_guard<py::gil_scoped_release>());
    m.def("fused_cpu_attn_mla_compressed", &fused_cpu_attn_mla_compressed,
          "Fused CPU decoupled MLA attention (compressed KV): score=q_proj.c_kv+q_pe.k_pe, "
          "o_acc=sum(softmax*c_kv)",
          py::arg("q_proj"), py::arg("q_pe"), py::arg("c_kv"), py::arg("k_pe"), py::arg("scale"),
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

