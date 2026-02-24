/**
 * kvann - V3 版本索引
 * 多层索引架构：
 *   Query
 *    ├── Delta (CPU, HNSW)      <- 热数据，可写
 *    ├── Base  (CPU, HNSW/PQ)   <- 温数据，只读
 *    └── Cold  (GPU, IVF-PQ)    <- 冷数据，GPU 加速
 * 
 * 特性：
 * - 自动分层管理
 * - GPU 加速（Brute-force/IVF）
 * - Product Quantization (PQ)
 * - 分布式支持接口
 */

#pragma once

#include "index_v2.h"
#include "gpu.h"
#include "pq.h"
#include <memory>
#include <queue>

namespace kvann {

// ============================================================================
// 分层策略
// ============================================================================

struct TieringPolicy {
    size_t hot_threshold = 1000;        // Delta 层最大数量
    size_t warm_threshold = 10000;      // Base 层最大数量，超过则进入 Cold
    bool enable_gpu = false;            // 是否启用 GPU
    bool enable_pq = false;             // 是否启用 PQ
    int pq_M = 8;                       // PQ 子空间数
    int pq_ksub = 256;                  // PQ 每子空间聚类数
    int nprobe = 10;                    // IVF nprobe
};

// ============================================================================
// V3 索引
// ============================================================================

class IndexV3 : public IndexV2 {
public:
    IndexV3(size_t dim,
            size_t max_elements = 1000000,
            QuantizeType quant_type = QuantizeType::FP32,
            const AutoRebuildPolicy& rebuild_policy = AutoRebuildPolicy(),
            const TieringPolicy& tiering_policy = TieringPolicy())
        : IndexV2(dim, max_elements, quant_type, rebuild_policy),
          tiering_policy_(tiering_policy) {
        
        // 初始化各层
        if (tiering_policy_.enable_gpu && gpu_available()) {
            cold_index_ = std::make_unique<GPUIVFIndex>(dim, 100, tiering_policy_.nprobe);
            gpu_storage_ = std::make_unique<GPVectorStorage>(dim, max_elements);
        }
        
        if (tiering_policy_.enable_pq) {
            pq_index_ = std::make_unique<PQIndex>(
                tiering_policy_.pq_M, 
                tiering_policy_.pq_ksub, 
                dim
            );
        }
    }
    
    ~IndexV3() = default;

    /**
     * 分层重建
     * 将数据按访问频率分层：
     * - Delta: 最近访问的
     * - Base: 中等访问频率
     * - Cold: 很少访问的（移到 GPU）
     */
    void rebuild_with_tiering() {
        std::cout << "[rebuild_v3] Starting tiered rebuild..." << std::endl;
        
        // 1. 收集所有有效 key
        auto live_keys = key_manager_.get_all_live();
        
        // 2. 按访问频率排序（简化：假设 ID 越大越热）
        std::vector<std::pair<Key, Slot>> hot, warm, cold;
        
        size_t total = live_keys.size();
        for (const auto& [key, slot] : live_keys) {
            if (hot.size() < tiering_policy_.hot_threshold) {
                hot.push_back({key, slot});
            } else if (warm.size() < tiering_policy_.warm_threshold) {
                warm.push_back({key, slot});
            } else {
                cold.push_back({key, slot});
            }
        }
        
        // 3. 重建 Base（温数据）
        {
            HNSWIndex new_base(dim_, max_elements_);
            new_base.set_vector_source(&storage_);
            
            for (const auto& [key, slot] : warm) {
                new_base.add(slot);
            }
            
            std::unique_lock<std::shared_mutex> lock(write_mutex_);
            base_index_ = std::move(new_base);
            std::cout << "[rebuild_v3] Base layer: " << warm.size() << " vectors" << std::endl;
        }
        
        // 4. 重建 Delta（热数据）
        {
            delta_keys_.clear();
            delta_slots_.clear();
            
            for (const auto& [key, slot] : hot) {
                delta_keys_.insert(key);
                delta_slots_[key] = slot;
            }
            
            std::cout << "[rebuild_v3] Delta layer: " << hot.size() << " vectors" << std::endl;
        }
        
        // 5. 重建 Cold（冷数据到 GPU）
        if (cold_index_ && !cold.empty()) {
            cold_index_->clear();
            
            // 训练 IVF（如果需要）
            if (!cold_index_->size()) {
                std::vector<float> train_vectors;
                train_vectors.reserve(std::min(cold.size(), (size_t)1000) * dim_);
                for (size_t i = 0; i < std::min(cold.size(), (size_t)1000); ++i) {
                    const float* vec = storage_.get_vector(cold[i].second);
                    train_vectors.insert(train_vectors.end(), vec, vec + dim_);
                }
                cold_index_->train(train_vectors, train_vectors.size() / dim_);
            }
            
            for (const auto& [key, slot] : cold) {
                const float* vec = storage_.get_vector(slot);
                cold_index_->add(slot, vec);
                
                if (gpu_storage_) {
                    gpu_storage_->upload(slot, vec);
                }
            }
            
            std::cout << "[rebuild_v3] Cold layer: " << cold.size() << " vectors (GPU)" << std::endl;
        }
        
        // 6. 训练 PQ（如果启用）
        if (pq_index_ && !pq_index_->is_trained() && !live_keys.empty()) {
            std::vector<float> train_vectors;
            train_vectors.reserve(std::min(live_keys.size(), (size_t)10000) * dim_);
            for (size_t i = 0; i < std::min(live_keys.size(), (size_t)10000); ++i) {
                const float* vec = storage_.get_vector(live_keys[i].second);
                train_vectors.insert(train_vectors.end(), vec, vec + dim_);
            }
            pq_index_->train(train_vectors, train_vectors.size() / dim_);
            
            // 将所有 Base 层数据加入 PQ
            for (const auto& [key, slot] : warm) {
                pq_index_->add(slot, storage_.get_vector(slot));
            }
        }
        
        std::cout << "[rebuild_v3] Done" << std::endl;
    }
    
    /**
     * 多层搜索
     * 1. Delta (HNSW) - 精确
     * 2. Base (HNSW) - 精确
     * 3. Cold (GPU IVF) - 近似（需要 rerank）
     * 4. 统一 rerank
     */
    std::vector<SearchResult> search(const float* query, int topk) override {
        // 归一化查询向量
        Vector normalized_query(query, query + dim_);
        normalize_vector(normalized_query.data(), dim_);
        
        std::shared_lock<std::shared_mutex> lock(write_mutex_);
        
        // 收集所有候选
        std::vector<std::pair<Slot, float>> candidates;
        
        // 1. Delta 层（热数据）
        for (Key key : delta_keys_) {
            auto it = delta_slots_.find(key);
            if (it == delta_slots_.end()) continue;
            
            Slot slot = it->second;
            const float* vec = storage_.get_vector(slot);
            float sim = cosine_similarity(normalized_query.data(), vec, dim_);
            candidates.emplace_back(slot, 1.0f - sim);
        }
        
        // 2. Base 层（温数据）
        if (!base_index_.empty()) {
            auto base_candidates = base_index_.search(
                normalized_query.data(),
                topk * 2,
                64
            );
            candidates.insert(candidates.end(), base_candidates.begin(), base_candidates.end());
        }
        
        // 3. Cold 层（GPU 冷数据）
        if (cold_index_ && cold_index_->size() > 0) {
            auto cold_candidates = cold_index_->search(
                normalized_query.data(),
                topk  // 召回更多用于 rerank
            );
            
            // Cold 层结果标记为需要 rerank（因为是近似结果）
            for (Slot slot : cold_candidates) {
                // 简单过滤：检查是否已在候选中
                bool exists = false;
                for (const auto& [s, _] : candidates) {
                    if (s == slot) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    // 标记为需要重新计算（添加一个哨兵值）
                    candidates.emplace_back(slot, -1.0f);
                }
            }
        }
        
        // 4. 统一 rerank
        return rerank_with_cold(normalized_query.data(), candidates, topk);
    }
    
    /**
     * 使用 PQ 加速搜索（仅用于召回，仍需 rerank）
     */
    std::vector<SearchResult> search_with_pq(const float* query, int topk) {
        if (!pq_index_ || !pq_index_->is_trained()) {
            return search(query, topk);
        }
        
        // 归一化
        Vector normalized_query(query, query + dim_);
        normalize_vector(normalized_query.data(), dim_);
        
        // 1. 使用 PQ 召回候选
        auto pq_candidates = pq_index_->search(normalized_query.data(), topk * 4);
        
        // 2. 精确 rerank
        std::vector<std::pair<Slot, float>> candidates;
        for (Slot slot : pq_candidates) {
            const float* vec = storage_.get_vector(slot);
            float sim = cosine_similarity(normalized_query.data(), vec, dim_);
            candidates.emplace_back(slot, 1.0f - sim);  // 转换为距离
        }
        
        // 3. 同时搜索 Delta 层（保证新数据可见）
        {
            std::shared_lock<std::shared_mutex> lock(write_mutex_);
            for (Key key : delta_keys_) {
                auto it = delta_slots_.find(key);
                if (it == delta_slots_.end()) continue;
                
                Slot slot = it->second;
                const float* vec = storage_.get_vector(slot);
                float sim = cosine_similarity(normalized_query.data(), vec, dim_);
                candidates.emplace_back(slot, 1.0f - sim);
            }
        }
        
        return rerank_with_cold(normalized_query.data(), candidates, topk);
    }
    
    /**
     * 获取各层统计
     */
    struct TierStats {
        size_t delta_count;
        size_t base_count;
        size_t cold_count;
        size_t pq_count;
    };
    
    TierStats tier_stats() const {
        TierStats stats;
        stats.delta_count = delta_keys_.size();
        stats.base_count = base_index_.size();
        stats.cold_count = cold_index_ ? cold_index_->size() : 0;
        stats.pq_count = pq_index_ ? pq_index_->size() : 0;
        return stats;
    }
    
    /**
     * 是否启用 GPU
     */
    bool gpu_enabled() const {
        return tiering_policy_.enable_gpu && gpu_available();
    }
    
    /**
     * 是否启用 PQ
     */
    bool pq_enabled() const {
        return tiering_policy_.enable_pq;
    }

protected:
    /**
     * Rerank（支持 Cold 层）
     */
    std::vector<SearchResult> rerank_with_cold(
            const float* query,
            const std::vector<std::pair<Slot, float>>& candidates,
            int topk) {
        
        // 去重并重新计算距离（Cold 层需要精确计算）
        std::unordered_set<Slot> seen;
        std::vector<std::pair<Slot, float>> unique_candidates;
        
        for (const auto& [slot, cached_dist] : candidates) {
            if (seen.insert(slot).second) {
                float sim;
                if (cached_dist < 0) {
                    // Cold 层结果，需要重新计算
                    const float* vec = storage_.get_vector(slot);
                    sim = cosine_similarity(query, vec, dim_);
                } else {
                    // 其他层，使用精确计算
                    const float* vec = storage_.get_vector(slot);
                    sim = cosine_similarity(query, vec, dim_);
                }
                unique_candidates.emplace_back(slot, sim);
            }
        }
        
        // 按相似度排序（降序）
        std::partial_sort(unique_candidates.begin(),
                          unique_candidates.begin() + std::min((size_t)topk, unique_candidates.size()),
                          unique_candidates.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // 转换为结果
        std::vector<SearchResult> result;
        result.reserve(std::min((size_t)topk, unique_candidates.size()));
        
        auto all_live = key_manager_.get_all_live();
        std::unordered_map<Slot, Key> slot_to_key;
        for (const auto& [key, slot] : all_live) {
            slot_to_key[slot] = key;
        }
        
        for (size_t i = 0; i < unique_candidates.size() && i < (size_t)topk; ++i) {
            Slot slot = unique_candidates[i].first;
            float score = unique_candidates[i].second;
            
            auto it = slot_to_key.find(slot);
            if (it != slot_to_key.end()) {
                Key key = it->second;
                auto meta = key_manager_.get_meta(key);
                if (meta) {
                    result.emplace_back(key, score, meta->user_data);
                } else {
                    result.emplace_back(key, score);
                }
            }
        }
        
        return result;
    }

protected:
    TieringPolicy tiering_policy_;
    
    // Cold 层（GPU）
    std::unique_ptr<GPUIVFIndex> cold_index_;
    std::unique_ptr<GPVectorStorage> gpu_storage_;
    
    // PQ 索引
    std::unique_ptr<PQIndex> pq_index_;
    
    using IndexV2::dim_;
    using IndexV2::max_elements_;
    using IndexV2::storage_;
    using IndexV2::key_manager_;
    using IndexV2::base_index_;
    using IndexV2::delta_keys_;
    using IndexV2::delta_slots_;
    using IndexV2::write_mutex_;
};

} // namespace kvann
