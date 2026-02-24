/**
 * kvann - 动态向量检索引擎核心定义
 * V1 版本：最小可用版本 (MVP)
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <memory>
#include <optional>
#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <iostream>
#include <fstream>

namespace kvann {

// ============================================================================
// 类型定义
// ============================================================================

using Key = uint64_t;           // 外部唯一ID（用户可见）
using Slot = uint32_t;          // 内部连续编号（索引/存储用）
using Vector = std::vector<float>;

// 无效的slot标记
static constexpr Slot INVALID_SLOT = static_cast<Slot>(-1);

// ============================================================================
// 数据结构
// ============================================================================

/**
 * 向量元数据
 */
struct VectorMeta {
    Slot slot;                  // 内部slot编号
    bool tombstone;             // 逻辑删除标记
    uint64_t version;           // 版本号（用于更新）
    std::vector<uint8_t> user_data;  // 用户自定义数据
    
    VectorMeta() : slot(INVALID_SLOT), tombstone(false), version(0) {}
    VectorMeta(Slot s) : slot(s), tombstone(false), version(1) {}
    
    /**
     * 序列化 user_data 大小（用于持久化）
     */
    size_t serialized_size() const {
        return sizeof(slot) + sizeof(tombstone) + sizeof(version) + sizeof(size_t) + user_data.size();
    }
};

/**
 * 搜索结果
 */
struct SearchResult {
    Key key;
    float score;                // 相似度分数（越高越相似）
    std::vector<uint8_t> user_data;  // 用户自定义数据
    
    SearchResult() : key(0), score(0) {}
    SearchResult(Key k, float s) : key(k), score(s) {}
    SearchResult(Key k, float s, const std::vector<uint8_t>& data) 
        : key(k), score(s), user_data(data) {}
    
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
inline float cosine_similarity(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

/**
 * 向量归一化
 */
inline void normalize_vector(float* vec, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (size_t i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
    }
}

/**
 * 检查向量是否已归一化
 */
inline bool is_normalized(const float* vec, size_t dim, float eps = 1e-5f) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    return std::abs(norm - 1.0f) < eps;
}

// ============================================================================
// KV 存储层 - 真相源
// ============================================================================

/**
 * 向量存储（连续内存，对齐）
 */
class VectorStorage {
public:
    VectorStorage(size_t dim, size_t initial_capacity = 1024)
        : dim_(dim), capacity_(initial_capacity), size_(0) {
        // 64字节对齐
        data_ = alloc_aligned_floats(capacity_ * dim_);
        if (!data_) {
            throw std::bad_alloc();
        }
    }
    
    ~VectorStorage() {
        free(data_);
    }
    
    // 禁止拷贝
    VectorStorage(const VectorStorage&) = delete;
    VectorStorage& operator=(const VectorStorage&) = delete;
    
    // 允许移动
    VectorStorage(VectorStorage&& other) noexcept
        : data_(other.data_), dim_(other.dim_), capacity_(other.capacity_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    /**
     * 分配新的slot
     */
    Slot allocate_slot() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (free_slots_.empty()) {
            if (size_ >= capacity_) {
                expand();
            }
            return size_++;
        } else {
            Slot slot = free_slots_.back();
            free_slots_.pop_back();
            return slot;
        }
    }
    
    /**
     * 释放slot（仅用于重建后整理）
     */
    void release_slot(Slot slot) {
        std::unique_lock<std::mutex> lock(mutex_);
        free_slots_.push_back(slot);
    }
    
    /**
     * 设置向量数据
     */
    void set_vector(Slot slot, const float* vec) {
        std::memcpy(data_ + slot * dim_, vec, dim_ * sizeof(float));
    }
    
    /**
     * 获取向量指针
     */
    const float* get_vector(Slot slot) const {
        return data_ + slot * dim_;
    }
    
    float* get_vector(Slot slot) {
        return data_ + slot * dim_;
    }
    
    /**
     * 获取指定slot的可写缓冲区
     */
    float* get_buffer(Slot slot) {
        return data_ + slot * dim_;
    }
    
    size_t dim() const { return dim_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    
    /**
     * 序列化到文件
     */
    void save(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        out.write(reinterpret_cast<const char*>(&size_), sizeof(size_));
        out.write(reinterpret_cast<const char*>(data_), size_ * dim_ * sizeof(float));
    }
    
    /**
     * 从文件反序列化
     */
    void load(std::ifstream& in) {
        in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
        in.read(reinterpret_cast<char*>(&size_), sizeof(size_));
        
        // 重新分配内存
        if (data_) free(data_);
        capacity_ = (size_ > 0) ? size_ : 1;
        data_ = alloc_aligned_floats(capacity_ * dim_);
        if (!data_) {
            throw std::bad_alloc();
        }
        
        if (size_ > 0) {
            in.read(reinterpret_cast<char*>(data_), size_ * dim_ * sizeof(float));
        }
    }

private:
    void expand() {
        size_t new_capacity = capacity_ * 2;
        float* new_data = alloc_aligned_floats(new_capacity * dim_);
        if (!new_data) {
            throw std::bad_alloc();
        }
        std::memcpy(new_data, data_, size_ * dim_ * sizeof(float));
        free(data_);
        data_ = new_data;
        capacity_ = new_capacity;
    }

private:
    static size_t round_up_64(size_t bytes) {
        return (bytes + 63u) & ~size_t(63u);
    }

    static float* alloc_aligned_floats(size_t count) {
        size_t bytes = round_up_64(count * sizeof(float));
        void* ptr = nullptr;
        if (bytes == 0) {
            bytes = 64;
        }
        if (posix_memalign(&ptr, 64, bytes) != 0) {
            return nullptr;
        }
        return static_cast<float*>(ptr);
    }

    float* data_;                           // 连续存储的向量数据
    size_t dim_;                            // 向量维度
    size_t capacity_;                       // 容量
    size_t size_;                           // 当前大小
    std::vector<Slot> free_slots_;          // 空闲slot列表
    mutable std::mutex mutex_;              // 保护free_slots_
};

/**
 * Key到Slot的映射 + 元数据
 */
class KeyManager {
public:
    bool exists(Key key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = key_map_.find(key);
        return it != key_map_.end() && !it->second.tombstone;
    }
    
    std::optional<VectorMeta> get_meta(Key key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = key_map_.find(key);
        if (it != key_map_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    /**
     * 添加新的key
     */
    void put(Key key, const VectorMeta& meta) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        key_map_[key] = meta;
    }
    
    /**
     * 标记删除（tombstone）
     */
    bool del(Key key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = key_map_.find(key);
        if (it != key_map_.end()) {
            it->second.tombstone = true;
            return true;
        }
        return false;
    }
    
    /**
     * 更新元数据
     */
    void update_meta(Key key, const VectorMeta& meta) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        key_map_[key] = meta;
    }
    
    /**
     * 获取所有有效的key（用于重建）
     */
    std::vector<std::pair<Key, Slot>> get_all_live() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<std::pair<Key, Slot>> result;
        result.reserve(key_map_.size());
        for (const auto& [key, meta] : key_map_) {
            if (!meta.tombstone) {
                result.emplace_back(key, meta.slot);
            }
        }
        return result;
    }
    
    /**
     * 获取统计信息
     */
    void get_stats(size_t& total, size_t& live, size_t& tombstones) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        total = key_map_.size();
        live = 0;
        tombstones = 0;
        for (const auto& [key, meta] : key_map_) {
            if (meta.tombstone) {
                tombstones++;
            } else {
                live++;
            }
        }
    }
    
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        key_map_.clear();
    }
    
    /**
     * 序列化
     */
    void save(std::ofstream& out) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        size_t size = key_map_.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& [key, meta] : key_map_) {
            out.write(reinterpret_cast<const char*>(&key), sizeof(key));
            out.write(reinterpret_cast<const char*>(&meta.slot), sizeof(meta.slot));
            out.write(reinterpret_cast<const char*>(&meta.tombstone), sizeof(meta.tombstone));
            out.write(reinterpret_cast<const char*>(&meta.version), sizeof(meta.version));
            
            // 序列化 user_data
            size_t user_data_size = meta.user_data.size();
            out.write(reinterpret_cast<const char*>(&user_data_size), sizeof(user_data_size));
            if (user_data_size > 0) {
                out.write(reinterpret_cast<const char*>(meta.user_data.data()), user_data_size);
            }
        }
    }
    
    /**
     * 反序列化
     */
    void load(std::ifstream& in) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        key_map_.clear();
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        for (size_t i = 0; i < size; ++i) {
            Key key;
            VectorMeta meta;
            in.read(reinterpret_cast<char*>(&key), sizeof(key));
            in.read(reinterpret_cast<char*>(&meta.slot), sizeof(meta.slot));
            in.read(reinterpret_cast<char*>(&meta.tombstone), sizeof(meta.tombstone));
            in.read(reinterpret_cast<char*>(&meta.version), sizeof(meta.version));
            
            // 反序列化 user_data
            size_t user_data_size;
            in.read(reinterpret_cast<char*>(&user_data_size), sizeof(user_data_size));
            if (user_data_size > 0) {
                meta.user_data.resize(user_data_size);
                in.read(reinterpret_cast<char*>(meta.user_data.data()), user_data_size);
            }
            
            key_map_[key] = meta;
        }
    }

private:
    std::unordered_map<Key, VectorMeta> key_map_;
    mutable std::shared_mutex mutex_;
};

} // namespace kvann
