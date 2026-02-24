/**
 * kvann - 向量量化支持
 * V2 版本：FP16, INT8 per-vector scale
 */

#pragma once

#include "core.h"

namespace kvann {

// ============================================================================
// 量化类型定义
// ============================================================================

enum class QuantizeType {
    FP32,       // 原始FP32
    FP16,       // 半精度浮点
    INT8,       // 8位整型 + per-vector scale
};

// ============================================================================
// FP16 支持
// ============================================================================

#ifdef ENABLE_AVX2
#include <immintrin.h>

/**
 * AVX2 FP32 -> FP16 转换
 */
inline void float32_to_float16_avx2(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 f32 = _mm256_loadu_ps(src + i);
        __m128i f16 = _mm256_cvtps_ph(f32, 0);  // 0 = round to nearest even
        _mm_storeu_si128((__m128i*)(dst + i), f16);
    }
    // 剩余元素（使用标量版本）
    for (; i < n; ++i) {
        uint32_t f = *reinterpret_cast<const uint32_t*>(src + i);
        uint32_t sign = (f >> 31) & 0x1;
        uint32_t exp = (f >> 23) & 0xFF;
        uint32_t mant = f & 0x7FFFFF;
        if (exp == 0) {
            dst[i] = (sign << 15);
        } else if (exp == 0xFF) {
            dst[i] = (sign << 15) | 0x7C00 | (mant >> 13);
        } else {
            int new_exp = (int)exp - 127 + 15;
            if (new_exp >= 31) dst[i] = (sign << 15) | 0x7C00;
            else if (new_exp <= 0) dst[i] = (sign << 15);
            else dst[i] = (sign << 15) | (new_exp << 10) | (mant >> 13);
        }
    }
}

/**
 * AVX2 FP16 -> FP32 转换
 */
inline void float16_to_float32_avx2(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m128i f16 = _mm_loadu_si128((__m128i*)(src + i));
        __m256 f32 = _mm256_cvtph_ps(f16);
        _mm256_storeu_ps(dst + i, f32);
    }
    // 剩余元素（使用标量版本）
    for (; i < n; ++i) {
        uint16_t h = src[i];
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f;
        if (exp == 0) {
            f = (sign << 31);
        } else if (exp == 0x1F) {
            f = (sign << 31) | 0x7F800000 | (mant << 13);
        } else {
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        *reinterpret_cast<uint32_t*>(dst + i) = f;
    }
}

#else  // 非AVX2版本

inline void float32_to_float16_scalar(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        // 简单实现：截取尾数（实际应该用F16C指令）
        uint32_t f = *reinterpret_cast<const uint32_t*>(src + i);
        uint32_t sign = (f >> 31) & 0x1;
        uint32_t exp = (f >> 23) & 0xFF;
        uint32_t mant = f & 0x7FFFFF;
        
        if (exp == 0) {
            dst[i] = (sign << 15);  // 零或次正规
        } else if (exp == 0xFF) {
            dst[i] = (sign << 15) | 0x7C00 | (mant >> 13);  // Inf或NaN
        } else {
            int new_exp = (int)exp - 127 + 15;
            if (new_exp >= 31) {
                dst[i] = (sign << 15) | 0x7C00;  // 溢出到Inf
            } else if (new_exp <= 0) {
                dst[i] = (sign << 15);  // 次正规或下溢
            } else {
                dst[i] = (sign << 15) | (new_exp << 10) | (mant >> 13);
            }
        }
    }
}

inline void float16_to_float32_scalar(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint16_t h = src[i];
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = (sign << 31);  // 零
            } else {
                // 次正规数
                int shift = __builtin_clz(mant) - 21;
                f = (sign << 31) | ((112 - shift) << 23) | (mant << shift);
            }
        } else if (exp == 0x1F) {
            f = (sign << 31) | 0x7F800000 | (mant << 13);  // Inf或NaN
        } else {
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        *reinterpret_cast<uint32_t*>(dst + i) = f;
    }
}

#define float32_to_float16 float32_to_float16_scalar
#define float16_to_float32 float16_to_float32_scalar

#endif  // ENABLE_AVX2

// ============================================================================
// INT8 量化 (per-vector scale)
// ============================================================================

struct Int8Vector {
    std::vector<int8_t> data;
    float scale;  // scale: FP32 = INT8 * scale
    
    void quantize(const float* src, size_t n) {
        // 找到最大值以计算scale
        float max_val = 0;
        for (size_t i = 0; i < n; ++i) {
            max_val = std::max(max_val, std::abs(src[i]));
        }
        
        if (max_val > 0) {
            scale = max_val / 127.0f;
            data.resize(n);
            for (size_t i = 0; i < n; ++i) {
                data[i] = static_cast<int8_t>(std::round(src[i] / scale));
            }
        } else {
            scale = 1.0f;
            data.resize(n, 0);
        }
    }
    
    void dequantize(float* dst, size_t n) const {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = data[i] * scale;
        }
    }
};

// ============================================================================
// AVX2 优化点积计算
// ============================================================================

#ifdef ENABLE_AVX2

/**
 * AVX2 FP32 点积
 */
inline float dot_product_avx2(const float* a, const float* b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // 水平求和
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    // 剩余元素
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

/**
 * AVX2 INT8 点积（使用FP16C指令将INT8扩展到FP32）
 * 注意：这是一个简化实现，实际需要unpack INT8到INT32再转换
 */
inline float dot_product_int8_avx2(const int8_t* a, const int8_t* b, size_t n) {
    int32_t sum = 0;
    size_t i = 0;
    
    // 使用AVX2进行INT8点积
    __m256i sum_vec = _mm256_setzero_si256();
    
    for (; i + 31 < n; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        
        // 分解为低16字节和高16字节
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0));
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        
        // 16位乘法并累加
        sum_vec = _mm256_add_epi32(sum_vec, _mm256_madd_epi16(va_lo, vb_lo));
        sum_vec = _mm256_add_epi32(sum_vec, _mm256_madd_epi16(va_hi, vb_hi));
    }
    
    // 水平求和
    int32_t temp[8];
    _mm256_storeu_si256((__m256i*)temp, sum_vec);
    for (int j = 0; j < 8; ++j) {
        sum += temp[j];
    }
    
    // 剩余元素
    for (; i < n; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    
    return static_cast<float>(sum);
}

#endif  // ENABLE_AVX2

// INT8 x FP32 点积（用于量化向量与FP32查询）
inline float dot_product_int8_float_scalar(const int8_t* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<float>(a[i]) * b[i];
    }
    return sum;
}

inline float dot_product_scalar(const float* a, const float* b, size_t n) {
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float dot_product_int8_scalar(const int8_t* a, const int8_t* b, size_t n) {
    int32_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return static_cast<float>(sum);
}

// ============================================================================
// 统一的相似度计算接口
// ============================================================================

class QuantizedVectorStorage {
public:
    QuantizedVectorStorage(size_t dim, QuantizeType type = QuantizeType::FP32)
        : dim_(dim), type_(type) {}
    
    /**
     * 存储向量（自动量化）
     */
    void store(Slot slot, const float* vec) {
        if (slot >= data_fp32_.size()) {
            resize(slot + 1);
        }
        
        switch (type_) {
            case QuantizeType::FP32:
                std::memcpy(data_fp32_[slot].data(), vec, dim_ * sizeof(float));
                break;
                
            case QuantizeType::FP16:
                data_fp16_[slot].resize(dim_);
                #ifdef ENABLE_AVX2
                float32_to_float16_avx2(vec, data_fp16_[slot].data(), dim_);
                #else
                float32_to_float16_scalar(vec, data_fp16_[slot].data(), dim_);
                #endif
                break;
                
            case QuantizeType::INT8:
                data_int8_[slot].quantize(vec, dim_);
                break;
        }
    }
    
    /**
     * 获取向量（反量化到FP32）
     */
    void retrieve(Slot slot, float* out) const {
        switch (type_) {
            case QuantizeType::FP32:
                std::memcpy(out, data_fp32_[slot].data(), dim_ * sizeof(float));
                break;
                
            case QuantizeType::FP16:
                #ifdef ENABLE_AVX2
                float16_to_float32_avx2(data_fp16_[slot].data(), out, dim_);
                #else
                float16_to_float32_scalar(data_fp16_[slot].data(), out, dim_);
                #endif
                break;
                
            case QuantizeType::INT8:
                data_int8_[slot].dequantize(out, dim_);
                break;
        }
    }
    
    /**
     * 计算点积（支持量化格式）
     */
    float dot(Slot slot, const float* query) const {
        switch (type_) {
            case QuantizeType::FP32: {
                #ifdef ENABLE_AVX2
                return dot_product_avx2(data_fp32_[slot].data(), query, dim_);
                #else
                return dot_product_scalar(data_fp32_[slot].data(), query, dim_);
                #endif
            }
            
            case QuantizeType::FP16: {
                // 反量化后计算（或优化为直接FP16计算）
                std::vector<float> temp(dim_);
                retrieve(slot, temp.data());
                #ifdef ENABLE_AVX2
                return dot_product_avx2(temp.data(), query, dim_);
                #else
                return dot_product_scalar(temp.data(), query, dim_);
                #endif
            }
            
            case QuantizeType::INT8: {
                const auto& v = data_int8_[slot];
                float int8_dot = dot_product_int8_float_scalar(v.data.data(), query, dim_);
                return int8_dot * v.scale;
            }
        }
        return 0;
    }
    
    QuantizeType type() const { return type_; }
    size_t dim() const { return dim_; }

private:
    void resize(size_t new_size) {
        switch (type_) {
            case QuantizeType::FP32:
                data_fp32_.resize(new_size);
                for (auto& v : data_fp32_) {
                    v.resize(dim_);
                }
                break;
            case QuantizeType::FP16:
                data_fp16_.resize(new_size);
                break;
            case QuantizeType::INT8:
                data_int8_.resize(new_size);
                break;
        }
    }

private:
    size_t dim_;
    QuantizeType type_;
    
    std::vector<std::vector<float>> data_fp32_;
    std::vector<std::vector<uint16_t>> data_fp16_;
    std::vector<Int8Vector> data_int8_;
};

} // namespace kvann
