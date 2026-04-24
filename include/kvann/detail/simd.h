// SIMD-accelerated primitives used on the search hot path.
//
// Backends are selected at compile time:
//   - x86_64 with AVX2+FMA  -> AVX2 path
//   - aarch64 / NEON        -> NEON path
//   - everything else        -> portable scalar
//
// All functions are header-only and inlinable so the compiler can fuse them
// into the HNSW inner loops.
#pragma once

#include <kvann/detail/arch.h>

#include <cmath>
#include <cstddef>

#if defined(KVANN_HAVE_AVX2)
#include <immintrin.h>
#endif

#if defined(KVANN_HAVE_NEON)
#include <arm_neon.h>
#endif

namespace kvann::simd {

// ---------- Dot product (a . b) ----------

KVANN_FORCE_INLINE float dot_f32_scalar(const float* KVANN_RESTRICT a,
                                        const float* KVANN_RESTRICT b,
                                        std::size_t n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

#if defined(KVANN_HAVE_AVX2)
KVANN_FORCE_INLINE float dot_f32_avx2(const float* KVANN_RESTRICT a,
                                      const float* KVANN_RESTRICT b,
                                      std::size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
    #if defined(KVANN_HAVE_FMA)
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    #else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
    #endif
    }
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
    #if defined(KVANN_HAVE_FMA)
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    #else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va, vb));
    #endif
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float s = _mm_cvtss_f32(sum128);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
#endif

#if defined(KVANN_HAVE_NEON)
KVANN_FORCE_INLINE float dot_f32_neon(const float* KVANN_RESTRICT a,
                                      const float* KVANN_RESTRICT b,
                                      std::size_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, va0, vb0);
        acc1 = vfmaq_f32(acc1, va1, vb1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc0 = vfmaq_f32(acc0, va, vb);
    }
    float32x4_t acc = vaddq_f32(acc0, acc1);
    float s = vaddvq_f32(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
#endif

KVANN_FORCE_INLINE float dot_f32(const float* a, const float* b, std::size_t n) {
#if defined(KVANN_HAVE_AVX2)
    return dot_f32_avx2(a, b, n);
#elif defined(KVANN_HAVE_NEON)
    return dot_f32_neon(a, b, n);
#else
    return dot_f32_scalar(a, b, n);
#endif
}

// ---------- Squared L2 distance ----------

KVANN_FORCE_INLINE float l2sq_f32(const float* a, const float* b, std::size_t n) {
    // For unit vectors (kvann normalizes inputs), L2^2 = 2 - 2*dot.
    // Generic implementation for completeness.
    float s = 0;
    for (std::size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// ---------- In-place L2 normalization ----------

KVANN_FORCE_INLINE void normalize_f32(float* v, std::size_t n) {
    float nrm2 = dot_f32(v, v, n);
    if (nrm2 <= 0.0f) return;
    float inv = 1.0f / std::sqrt(nrm2);
#if defined(KVANN_HAVE_AVX2)
    __m256 s = _mm256_set1_ps(inv);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        _mm256_storeu_ps(v + i, _mm256_mul_ps(x, s));
    }
    for (; i < n; ++i) v[i] *= inv;
#elif defined(KVANN_HAVE_NEON)
    float32x4_t s = vdupq_n_f32(inv);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(v + i);
        vst1q_f32(v + i, vmulq_f32(x, s));
    }
    for (; i < n; ++i) v[i] *= inv;
#else
    for (std::size_t i = 0; i < n; ++i) v[i] *= inv;
#endif
}

// ---------- Backend name (for diagnostics) ----------

KVANN_FORCE_INLINE const char* backend_name() {
#if defined(KVANN_HAVE_AVX2)
    return "avx2";
#elif defined(KVANN_HAVE_NEON)
    return "neon";
#else
    return "scalar";
#endif
}

} // namespace kvann::simd
