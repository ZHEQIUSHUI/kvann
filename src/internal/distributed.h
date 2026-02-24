/**
 * kvann - 分布式支持接口
 * V3 版本
 * 
 * 设计：
 * - 按 Key hash 分片（Sharding）
 * - 每 shard 独立维护 base/delta
 * - 查询时 fan-out 到所有 shards，合并结果
 * 
 * 注意：这是单机模拟版本，实际分布式需要网络通信
 */

#pragma once

#include "index_v3.h"
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include <thread>

namespace kvann {

// ============================================================================
// 分片策略
// ============================================================================

using ShardId = uint32_t;

/**
 * 分片函数
 */
using ShardFunc = std::function<ShardId(Key, ShardId)>;

/**
 * 默认分片函数：key % nshards
 */
inline ShardId default_shard_func(Key key, ShardId nshards) {
    return static_cast<ShardId>(key % nshards);
}

// ============================================================================
// 分片索引
// ============================================================================

/**
 * 分布式索引
 * 管理多个本地 shard，提供统一接口
 */
class DistributedIndex {
public:
    DistributedIndex(size_t dim,
                     size_t nshards = 4,
                     ShardFunc shard_func = default_shard_func)
        : dim_(dim), nshards_(nshards), shard_func_(shard_func) {
        
        shards_.reserve(nshards);
        for (ShardId i = 0; i < nshards; ++i) {
            shards_.push_back(std::make_unique<IndexV3>(dim));
        }
    }
    
    ~DistributedIndex() = default;

    // ============================================================================
    // 基本操作
    // ============================================================================
    
    /**
     * 插入/更新向量
     */
    bool put(Key key, const float* vector) {
        return get_shard(key)->put(key, vector);
    }
    
    /**
     * 插入/更新（带 user_data）
     */
    bool put_with_data(Key key, const float* vector, const void* user_data, size_t user_data_len) {
        return get_shard(key)->put_with_data(key, vector, user_data, user_data_len);
    }
    
    /**
     * 删除
     */
    bool del(Key key) {
        return get_shard(key)->del(key);
    }
    
    /**
     * 检查存在
     */
    bool exists(Key key) const {
        return get_shard(key)->exists(key);
    }
    
    /**
     * 获取 user_data
     */
    std::vector<uint8_t> get_user_data(Key key) const {
        return get_shard(key)->get_user_data(key);
    }

    // ============================================================================
    // 搜索（Fan-out / Merge）
    // ============================================================================
    
    /**
     * 搜索所有分片，合并结果
     * 策略：从每个 shard 取 topk，再全局 rerank
     */
    std::vector<SearchResult> search(const float* query, int topk) {
        // 归一化查询向量
        std::vector<float> normalized_query(query, query + dim_);
        normalize_vector(normalized_query.data(), dim_);
        
        // Fan-out: 并行搜索所有 shards
        std::vector<std::future<std::vector<SearchResult>>> futures;
        futures.reserve(nshards_);
        
        for (ShardId i = 0; i < nshards_; ++i) {
            futures.push_back(std::async(std::launch::async, [this, i, &normalized_query, topk]() {
                return shards_[i]->search(normalized_query.data(), topk);
            }));
        }
        
        // 收集结果
        std::vector<SearchResult> all_results;
        for (auto& f : futures) {
            auto shard_results = f.get();
            all_results.insert(all_results.end(), shard_results.begin(), shard_results.end());
        }
        
        // Merge: 全局排序取 topk
        std::partial_sort(all_results.begin(),
                          all_results.begin() + std::min((size_t)topk, all_results.size()),
                          all_results.end(),
                          [](const auto& a, const auto& b) { return a.score > b.score; });
        
        if (all_results.size() > (size_t)topk) {
            all_results.resize(topk);
        }
        
        return all_results;
    }
    
    /**
     * 使用 PQ 搜索
     */
    std::vector<SearchResult> search_with_pq(const float* query, int topk) {
        std::vector<float> normalized_query(query, query + dim_);
        normalize_vector(normalized_query.data(), dim_);
        
        std::vector<std::future<std::vector<SearchResult>>> futures;
        futures.reserve(nshards_);
        
        for (ShardId i = 0; i < nshards_; ++i) {
            futures.push_back(std::async(std::launch::async, [this, i, &normalized_query, topk]() {
                return shards_[i]->search_with_pq(normalized_query.data(), topk);
            }));
        }
        
        std::vector<SearchResult> all_results;
        for (auto& f : futures) {
            auto shard_results = f.get();
            all_results.insert(all_results.end(), shard_results.begin(), shard_results.end());
        }
        
        std::partial_sort(all_results.begin(),
                          all_results.begin() + std::min((size_t)topk, all_results.size()),
                          all_results.end(),
                          [](const auto& a, const auto& b) { return a.score > b.score; });
        
        if (all_results.size() > (size_t)topk) {
            all_results.resize(topk);
        }
        
        return all_results;
    }

    // ============================================================================
    // 维护操作
    // ============================================================================
    
    /**
     * 重建所有 shards
     */
    void rebuild_all() {
        std::vector<std::future<void>> futures;
        futures.reserve(nshards_);
        
        for (ShardId i = 0; i < nshards_; ++i) {
            futures.push_back(std::async(std::launch::async, [this, i]() {
                shards_[i]->rebuild();
            }));
        }
        
        for (auto& f : futures) {
            f.get();
        }
    }
    
    /**
     * 分层重建所有 shards
     */
    void rebuild_all_with_tiering() {
        std::vector<std::future<void>> futures;
        futures.reserve(nshards_);
        
        for (ShardId i = 0; i < nshards_; ++i) {
            futures.push_back(std::async(std::launch::async, [this, i]() {
                shards_[i]->rebuild_with_tiering();
            }));
        }
        
        for (auto& f : futures) {
            f.get();
        }
    }
    
    /**
     * 等待所有 shards 的重建完成
     */
    void wait_all_rebuilds() {
        for (auto& shard : shards_) {
            shard->wait_rebuild();
        }
    }

    // ============================================================================
    // 统计信息
    // ============================================================================
    
    struct DistStats {
        size_t total_vectors = 0;
        size_t total_live = 0;
        size_t total_tombstones = 0;
        std::vector<IndexStats> shard_stats;
    };
    
    DistStats stats() const {
        DistStats ds;
        ds.shard_stats.reserve(nshards_);
        
        for (const auto& shard : shards_) {
            auto s = shard->stats();
            ds.shard_stats.push_back(s);
            ds.total_vectors += s.total_vectors;
            ds.total_live += s.live_vectors;
            ds.total_tombstones += s.tombstone_count;
        }
        
        return ds;
    }
    
    size_t nshards() const { return nshards_; }
    size_t dim() const { return dim_; }

protected:
    IndexV3* get_shard(Key key) {
        ShardId sid = shard_func_(key, nshards_);
        return shards_[sid].get();
    }
    
    const IndexV3* get_shard(Key key) const {
        ShardId sid = shard_func_(key, nshards_);
        return shards_[sid].get();
    }

protected:
    size_t dim_;
    size_t nshards_;
    ShardFunc shard_func_;
    std::vector<std::unique_ptr<IndexV3>> shards_;
};

} // namespace kvann
