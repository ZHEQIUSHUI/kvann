/**
 * kvann - 主索引类
 * V1 版本：Base/Delta 双层架构
 */

#pragma once

#include "core.h"
#include "hnsw.h"
#include <thread>
#include <future>

namespace kvann {

/**
 * Base/Delta 双层索引
 * 
 * 架构：
 * - Base: 已build的只读HNSW索引
 * - Delta: 新写入的可变层，使用brute-force
 * 
 * 查询流程：
 * 1. 从Base层召回候选
 * 2. 从Delta层召回候选（brute-force）
 * 3. 合并候选
 * 4. 统一rerank（使用精确距离）
 */
class Index {
public:
    /**
     * 构造函数
     * @param dim: 向量维度
     * @param max_elements: 最大元素数
     * @param delta_threshold: Delta层使用brute-force的最大数量，超过则触发rebuild建议
     */
    Index(size_t dim, size_t max_elements = 1000000, size_t delta_threshold = 1000)
        : dim_(dim), 
          max_elements_(max_elements),
          delta_threshold_(delta_threshold),
          storage_(dim, max_elements),
          base_index_(dim, max_elements),
          rebuild_running_(false) {
        
        base_index_.set_vector_source(&storage_);
    }

    virtual ~Index() {
        if (rebuild_thread_.joinable()) {
            rebuild_thread_.join();
        }
    }
    
    // ============================================================================
    // 核心 API
    // ============================================================================
    
    /**
     * 插入/更新向量
     */
    bool put(Key key, const float* vector) {
        return put_with_data(key, vector, nullptr, 0);
    }
    
    /**
     * 插入/更新向量（带user_data）
     * @param key: 唯一键
     * @param vector: 向量数据
     * @param user_data: 用户自定义数据（可为nullptr）
     * @param user_data_len: user_data长度（字节）
     */
    virtual bool put_with_data(Key key, const float* vector, const void* user_data, size_t user_data_len) {
        // 归一化输入向量（原地修改）
        Vector normalized_vec(vector, vector + dim_);
        normalize_vector(normalized_vec.data(), dim_);
        
        std::unique_lock<std::shared_mutex> lock(write_mutex_);
        
        // 检查是否已存在
        auto existing = key_manager_.get_meta(key);
        
        if (existing && !existing->tombstone) {
            // 更新：使用原slot
            Slot slot = existing->slot;
            storage_.set_vector(slot, normalized_vec.data());
            
            VectorMeta meta = *existing;
            meta.version++;
            // 更新user_data
            if (user_data && user_data_len > 0) {
                const uint8_t* data_ptr = static_cast<const uint8_t*>(user_data);
                meta.user_data.assign(data_ptr, data_ptr + user_data_len);
            } else {
                meta.user_data.clear();
            }
            key_manager_.update_meta(key, meta);
            
            // 添加到delta（用于查询时可见）
            delta_keys_.insert(key);
            delta_slots_[key] = slot;
            
            return true;
        }
        
        // 新增
        Slot slot;
        if (existing) {
            // 复用被删除的slot
            slot = existing->slot;
        } else {
            slot = storage_.allocate_slot();
        }
        
        storage_.set_vector(slot, normalized_vec.data());
        
        VectorMeta meta(slot);
        // 存储user_data
        if (user_data && user_data_len > 0) {
            const uint8_t* data_ptr = static_cast<const uint8_t*>(user_data);
            meta.user_data.assign(data_ptr, data_ptr + user_data_len);
        }
        key_manager_.put(key, meta);
        
        // 添加到delta
        delta_keys_.insert(key);
        delta_slots_[key] = slot;
        
        return true;
    }
    
    /**
     * 删除向量（tombstone机制）
     */
    virtual bool del(Key key) {
        std::unique_lock<std::shared_mutex> lock(write_mutex_);
        
        if (!key_manager_.exists(key)) {
            return false;
        }
        
        key_manager_.del(key);
        
        // 从delta中移除（如果存在）
        delta_keys_.erase(key);
        delta_slots_.erase(key);
        
        return true;
    }
    
    /**
     * 检查key是否存在（且未删除）
     */
    bool exists(Key key) const {
        auto meta = key_manager_.get_meta(key);
        return meta.has_value() && !meta->tombstone;
    }
    
    /**
     * 获取key对应的user_data
     * @return user_data 的指针，如果不存在或没有user_data则返回nullptr
     */
    std::vector<uint8_t> get_user_data(Key key) const {
        auto meta = key_manager_.get_meta(key);
        if (meta && !meta->tombstone) {
            return meta->user_data;
        }
        return {};
    }
    
    /**
     * 搜索（多线程安全）
     */
    virtual std::vector<SearchResult> search(const float* query, int topk) {
        // 归一化查询向量
        Vector normalized_query(query, query + dim_);
        normalize_vector(normalized_query.data(), dim_);
        
        // 获取查询快照
        std::shared_lock<std::shared_mutex> lock(write_mutex_);
        
        // 收集所有候选
        std::vector<std::pair<Slot, float>> candidates;
        
        // 1. 从Base层召回（Base层只包含有效向量，无需过滤）
        if (!base_index_.empty()) {
            auto base_candidates = base_index_.search(
                normalized_query.data(), 
                topk * 2,  // 召回更多，用于rerank
                64
            );
            candidates.insert(candidates.end(), base_candidates.begin(), base_candidates.end());
        }
        
        // 2. 从Delta层召回（brute-force）
        search_delta_brute_force(normalized_query.data(), topk * 2, candidates);
        
        // 3. 统一rerank（使用精确距离）
        return rerank(normalized_query.data(), candidates, topk);
    }
    
    /**
     * 手动重建索引
     * 将Delta层合并到Base层
     */
    void rebuild() {
        std::unique_lock<std::shared_mutex> lock(write_mutex_);
        
        if (rebuild_running_.load()) {
            return;  // 已有重建在进行中
        }
        
        rebuild_running_ = true;
        lock.unlock();
        
        // 在后台执行重建
        if (rebuild_thread_.joinable()) {
            rebuild_thread_.join();
        }
        
        rebuild_thread_ = std::thread([this]() {
            do_rebuild();
        });
    }
    
    /**
     * 等待重建完成
     */
    void wait_rebuild() {
        if (rebuild_thread_.joinable()) {
            rebuild_thread_.join();
        }
    }
    
    /**
     * 获取统计信息
     */
    IndexStats stats() const {
        IndexStats s;
        
        size_t total, live, tombstones;
        key_manager_.get_stats(total, live, tombstones);
        
        s.total_vectors = total;
        s.live_vectors = live;
        s.tombstone_count = tombstones;
        s.base_count = base_index_.size();
        
        std::shared_lock<std::shared_mutex> lock(write_mutex_);
        s.delta_count = delta_keys_.size();
        
        s.tombstone_ratio = total > 0 ? (float)tombstones / total : 0;
        s.delta_ratio = live > 0 ? (float)s.delta_count / live : 0;
        s.dim = dim_;
        
        return s;
    }
    
    /**
     * 保存索引到文件
     */
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open file for writing: " + path);
        }
        
        // 写入元数据
        size_t dim = dim_;
        size_t max_elements = max_elements_;
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(&max_elements), sizeof(max_elements));
        
        // 写入Key管理器
        key_manager_.save(out);
        
        // 写入向量存储
        storage_.save(out);
        
        out.close();
    }
    
    /**
     * 从文件加载索引
     */
    static std::unique_ptr<Index> load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open file for reading: " + path);
        }
        
        // 读取元数据
        size_t dim, max_elements;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        in.read(reinterpret_cast<char*>(&max_elements), sizeof(max_elements));
        
        // 创建索引
        auto index = std::make_unique<Index>(dim, max_elements);
        
        // 加载Key管理器
        index->key_manager_.load(in);
        
        // 加载向量存储
        index->storage_.load(in);
        
        in.close();
        
        // 重建索引
        index->rebuild_base_from_kv();
        
        return index;
    }

protected:
    /**
     * Delta层brute-force搜索
     */
    void search_delta_brute_force(const float* query, int /* topk */, 
                                   std::vector<std::pair<Slot, float>>& results) const {
        // 简单实现：遍历所有delta向量
        for (Key key : delta_keys_) {
            auto it = delta_slots_.find(key);
            if (it == delta_slots_.end()) continue;
            
            Slot slot = it->second;
            const float* vec = storage_.get_vector(slot);
            float sim = cosine_similarity(query, vec, dim_);
            
            results.emplace_back(slot, 1.0f - sim);  // 转换为距离
        }
    }
    
    /**
     * Rerank阶段：使用精确距离排序
     */
    std::vector<SearchResult> rerank(const float* query,
                                      const std::vector<std::pair<Slot, float>>& candidates,
                                      int topk) {
        // 去重（按slot）
        std::unordered_set<Slot> seen;
        std::vector<std::pair<Slot, float>> unique_candidates;
        
        for (const auto& [slot, _] : candidates) {
            if (seen.insert(slot).second) {
                // 精确计算距离
                const float* vec = storage_.get_vector(slot);
                float sim = cosine_similarity(query, vec, dim_);
                unique_candidates.emplace_back(slot, sim);
            }
        }
        
        // 按相似度排序（降序）
        std::partial_sort(unique_candidates.begin(), 
                          unique_candidates.begin() + std::min((size_t)topk, unique_candidates.size()),
                          unique_candidates.end(),
                          [](const auto& a, const auto& b) {
                              return a.second > b.second;
                          });
        
        // 转换为结果
        std::vector<SearchResult> result;
        result.reserve(std::min((size_t)topk, unique_candidates.size()));
        
        // 获取所有key->slot映射的反向查找
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
                // 获取 user_data
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
    
    /**
     * 检查slot是否已被删除
     */
    bool is_slot_deleted(Slot slot) const {
        // 检查这个slot对应的key是否是tombstone
        auto all_live = key_manager_.get_all_live();
        for (const auto& [key, s] : all_live) {
            if (s == slot) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * 执行重建（内部，可被V2覆盖）
     */
    virtual void do_rebuild() {
        std::cout << "[rebuild] Starting..." << std::endl;
        
        // 1. 收集所有有效key
        auto live_keys = key_manager_.get_all_live();
        
        // 2. 创建新的base索引
        HNSWIndex new_base(dim_, max_elements_);
        new_base.set_vector_source(&storage_);
        
        // 3. 将所有有效向量添加到新索引
        for (const auto& [key, slot] : live_keys) {
            new_base.add(slot);
        }
        
        // 4. 原子切换
        std::unique_lock<std::shared_mutex> lock(write_mutex_);
        base_index_ = std::move(new_base);
        delta_keys_.clear();
        delta_slots_.clear();
        
        rebuild_running_ = false;
        std::cout << "[rebuild] Done. Base size: " << base_index_.size() << std::endl;
    }
    
    /**
     * 从KV重建base索引（用于加载）
     */
    void rebuild_base_from_kv() {
        base_index_.clear();
        
        auto live_keys = key_manager_.get_all_live();
        for (const auto& [key, slot] : live_keys) {
            base_index_.add(slot);
        }
        
        // 清空delta（因为已经合并到base）
        delta_keys_.clear();
        delta_slots_.clear();
    }

protected:
    size_t dim_;                    // 向量维度
    size_t max_elements_;           // 最大元素数
    size_t delta_threshold_;        // Delta阈值
    
    VectorStorage storage_;         // 向量存储（真相源）
    KeyManager key_manager_;        // Key管理器
    
    HNSWIndex base_index_;          // Base层（只读HNSW）
    
    // Delta层
    std::unordered_set<Key> delta_keys_;
    std::unordered_map<Key, Slot> delta_slots_;
    
    // 同步
    mutable std::shared_mutex write_mutex_;
    
    // 重建
    std::atomic<bool> rebuild_running_;
    std::thread rebuild_thread_;
};

} // namespace kvann
