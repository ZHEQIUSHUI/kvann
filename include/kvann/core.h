/**
 * kvann - 动态向量检索引擎核心定义
 * V1 版本：最小可用版本 (MVP)
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace kvann {

// ============================================================================
// 类型定义
// ============================================================================

using Key = uint64_t;           // 外部唯一ID（用户可见）
using Slot = uint32_t;          // 内部连续编号（索引/存储用）
using Vector = std::vector<float>;

// 无效的slot标记
static constexpr Slot INVALID_SLOT = static_cast<Slot>(-1);

/**
 * 搜索结果
 */
struct SearchResult {
    Key key;
    float score;                // 相似度分数（越高越相似）
    
    SearchResult() : key(0), score(0) {}
    SearchResult(Key k, float s) : key(k), score(s) {}
    
    // 用于priority_queue（分数高的在前）
    bool operator<(const SearchResult& other) const {
        return score < other.score;
    }
};

/**
 * 索引统计信息
 */
struct IndexStats {
    size_t total_vectors;       // 总向量数（含tombstone）
    size_t live_vectors;        // 有效向量数
    size_t tombstone_count;     // tombstone数量
    size_t base_count;          // base层向量数
    size_t delta_count;         // delta层向量数
    float tombstone_ratio;      // tombstone比例
    float delta_ratio;          // delta比例
    size_t dim;                 // 向量维度
    
    IndexStats() : total_vectors(0), live_vectors(0), tombstone_count(0),
                   base_count(0), delta_count(0), tombstone_ratio(0),
                   delta_ratio(0), dim(0) {}
};

// ============================================================================
// 相似度计算
// ============================================================================

/**
 * 余弦相似度计算（normalize + dot）
 * 输入向量必须是归一化的
 */
float cosine_similarity(const float* a, const float* b, size_t dim);

/**
 * 向量归一化
 */
void normalize_vector(float* vec, size_t dim);

/**
 * 检查向量是否已归一化
 */
bool is_normalized(const float* vec, size_t dim, float eps = 1e-5f);

} // namespace kvann
