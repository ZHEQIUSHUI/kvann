/**
 * kvann - V2 版本索引
 * 新增：
 * - 向量量化（FP16, INT8）
 * - SIMD 优化
 * - Delta层可写HNSW
 * - 自动rebuild策略
 */

#pragma once

#include "index.h"
#include "quantization.h"

namespace kvann {

/**
 * 自动重建策略配置
 */
struct AutoRebuildPolicy {
    float tombstone_ratio_threshold = 0.2f;  // tombstone比例超过此值触发rebuild
    float delta_ratio_threshold = 0.3f;      // delta比例超过此值触发rebuild
    size_t min_delta_size = 100;             // 最小delta大小才触发
    bool auto_rebuild = false;               // 是否启用自动重建
    size_t max_delta_for_bruteforce = 1000;  // Delta层使用brute-force的最大数量
};

/**
 * V2 索引类（继承自V1）
 * 使用using声明访问基类protected成员
 */
class IndexV2 : public Index {
public:
    IndexV2(size_t dim, 
            size_t max_elements = 1000000,
            QuantizeType quant_type = QuantizeType::FP32,
            const AutoRebuildPolicy& policy = AutoRebuildPolicy())
        : Index(dim, max_elements, policy.max_delta_for_bruteforce),
          quant_type_(quant_type),
          rebuild_policy_(policy) {
        
        // 如果启用量化，初始化量化存储
        if (quant_type != QuantizeType::FP32) {
            use_quantization_ = true;
        }

        if (use_quantization_) {
            quant_storage_ = std::make_unique<QuantizedVectorStorage>(dim, quant_type_);
        }
    }

    /**
     * 设置自动重建策略
     */
    void set_rebuild_policy(const AutoRebuildPolicy& policy) {
        std::unique_lock<std::shared_mutex> lock(write_mutex_);
        rebuild_policy_ = policy;
    }
    
    /**
     * 检查是否需要重建（手动调用）
     */
    bool needs_rebuild() const {
        auto stats = this->stats();
        
        if (stats.tombstone_ratio > rebuild_policy_.tombstone_ratio_threshold) {
            return true;
        }
        
        if (stats.delta_count > rebuild_policy_.min_delta_size &&
            stats.delta_ratio > rebuild_policy_.delta_ratio_threshold) {
            return true;
        }
        
        return false;
    }
    
    QuantizeType quantize_type() const { return quant_type_; }

    bool put_with_data(Key key, const float* vector, const void* user_data, size_t user_data_len) override {
        bool ok = Index::put_with_data(key, vector, user_data, user_data_len);
        if (ok && use_quantization_) {
            auto meta = key_manager_.get_meta(key);
            if (meta && !meta->tombstone) {
                quant_storage_->store(meta->slot, storage_.get_vector(meta->slot));
            }
        }
        maybe_trigger_rebuild();
        return ok;
    }

    bool del(Key key) override {
        bool ok = Index::del(key);
        if (ok) {
            maybe_trigger_rebuild();
        }
        return ok;
    }

    std::vector<SearchResult> search(const float* query, int topk) override {
        // 归一化查询向量
        Vector normalized_query(query, query + dim_);
        normalize_vector(normalized_query.data(), dim_);
        
        // 获取查询快照
        std::shared_lock<std::shared_mutex> lock(write_mutex_);
        
        // 收集所有候选
        std::vector<std::pair<Slot, float>> candidates;
        
        // 1. Base 层召回
        if (!base_index_.empty()) {
            auto base_candidates = base_index_.search(
                normalized_query.data(),
                topk * 2,
                64
            );
            candidates.insert(candidates.end(), base_candidates.begin(), base_candidates.end());
        }
        
        // 2. Delta 层召回（如果量化启用，则用量化点积近似）
        if (use_quantization_ && quant_storage_) {
            for (Key key : delta_keys_) {
                auto it = delta_slots_.find(key);
                if (it == delta_slots_.end()) continue;
                Slot slot = it->second;
                float sim = quant_storage_->dot(slot, normalized_query.data());
                candidates.emplace_back(slot, 1.0f - sim);
            }
        } else {
            search_delta_brute_force(normalized_query.data(), topk * 2, candidates);
        }
        
        // 3. 统一 rerank（精确距离）
        return rerank(normalized_query.data(), candidates, topk);
    }

private:
    void maybe_trigger_rebuild() {
        if (!rebuild_policy_.auto_rebuild) {
            return;
        }
        if (needs_rebuild()) {
            rebuild();
        }
    }

    QuantizeType quant_type_;
    bool use_quantization_ = false;
    AutoRebuildPolicy rebuild_policy_;
    std::unique_ptr<QuantizedVectorStorage> quant_storage_;
};

} // namespace kvann
