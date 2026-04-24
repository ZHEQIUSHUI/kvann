// kvann 公共基础类型与相似度工具
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace kvann {

using Key  = std::uint64_t;             // 外部唯一 ID
using Slot = std::uint32_t;             // 内部稠密编号

// 保留值
inline constexpr Slot kInvalidSlot = std::numeric_limits<Slot>::max();
inline constexpr Key  kInvalidKey  = std::numeric_limits<Key>::max();

// ---------------------------------------------------------------------------
// 相似度工具
// ---------------------------------------------------------------------------

// 余弦相似度（输入应为已归一化的向量；内部走 SIMD dot）
float cosine_similarity(const float* a, const float* b, std::size_t dim);

// L2 in-place 归一化
void normalize_vector(float* vec, std::size_t dim);

// 检查是否近似归一化
bool is_normalized(const float* vec, std::size_t dim, float eps = 1e-5f);

// 当前 SIMD 后端 "avx2" / "neon" / "scalar"
const char* simd_backend();

} // namespace kvann
