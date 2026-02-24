/**
 * kvann - GPU/CUDA 支持接口
 * V3 版本
 * 
 * 注意：这是接口定义，实际 CUDA 实现需要在 .cu 文件中
 * 未启用 CUDA 时提供 CPU 回退实现
 */

#pragma once

#include "core.h"
#include <vector>
#include <memory>
#include <cstring>

namespace kvann {

// ============================================================================
// GPU 可用性检查
// ============================================================================

/**
 * 检查 GPU 是否可用
 */
inline bool gpu_available() {
#ifdef KVANN_ENABLE_CUDA
    return true;  // 实际实现需要检查 CUDA 设备
#else
    return false;
#endif
}

/**
 * 获取 GPU 信息
 */
struct GPUInfo {
    int device_count = 0;
    std::string device_name;
    size_t total_memory = 0;
    size_t free_memory = 0;
};

inline GPUInfo get_gpu_info() {
    GPUInfo info;
#ifdef KVANN_ENABLE_CUDA
    // 实际 CUDA 实现
    // cudaGetDeviceCount(&info.device_count);
    // ...
#endif
    return info;
}

// ============================================================================
// GPU 向量存储
// ============================================================================

/**
 * GPU 向量存储基类
 * 管理 GPU 内存中的向量数据
 */
class GPVectorStorage {
public:
    GPVectorStorage(size_t dim, size_t max_elements)
        : dim_(dim), max_elements_(max_elements), size_(0) {
    }
    
    virtual ~GPVectorStorage() = default;
    
    /**
     * 将向量从 CPU 复制到 GPU
     */
    virtual void upload(Slot slot, const float* vec) {
        // CPU 回退：存储在内存中
        if (slot >= cpu_buffer_.size()) {
            cpu_buffer_.resize(slot + 1);
        }
        cpu_buffer_[slot].resize(dim_);
        std::memcpy(cpu_buffer_[slot].data(), vec, dim_ * sizeof(float));
        if (slot + 1 > size_) {
            size_ = slot + 1;
        }
    }
    
    /**
     * 批量上传向量
     */
    virtual void upload_batch(const std::vector<Slot>& slots, 
                              const std::vector<float>& vectors) {
        for (size_t i = 0; i < slots.size(); ++i) {
            upload(slots[i], vectors.data() + i * dim_);
        }
    }
    
    /**
     * 删除 GPU 上的向量
     */
    virtual void remove(Slot slot) {
        if (slot < cpu_buffer_.size()) {
            cpu_buffer_[slot].clear();
        }
    }
    
    size_t dim() const { return dim_; }
    size_t size() const { return size_; }

public:
    // CPU 回退存储（允许子类访问）
    std::vector<std::vector<float>> cpu_buffer_;
    
protected:
    size_t dim_;
    size_t max_elements_;
    size_t size_;
};

// ============================================================================
// GPU Brute-force 搜索
// ============================================================================

/**
 * GPU Brute-force 索引
 * 在 GPU 上执行暴力搜索
 */
class GPUBruteForceIndex {
public:
    GPUBruteForceIndex(size_t dim, size_t max_elements)
        : dim_(dim), max_elements_(max_elements) {
        storage_ = std::make_unique<GPVectorStorage>(dim, max_elements);
    }
    
    virtual ~GPUBruteForceIndex() = default;
    
    /**
     * 添加向量
     */
    virtual void add(Slot slot, const float* vec) {
        storage_->upload(slot, vec);
        slots_.push_back(slot);
    }
    
    /**
     * 批量添加
     */
    virtual void add_batch(const std::vector<Slot>& slots,
                           const std::vector<float>& vectors) {
        for (size_t i = 0; i < slots.size(); ++i) {
            add(slots[i], vectors.data() + i * dim_);
        }
    }
    
    /**
     * 搜索
     * @return 返回候选 slot 列表（CPU rerank 用）
     */
    virtual std::vector<Slot> search(const float* query, int k) {
        // CPU 回退实现
        std::vector<std::pair<Slot, float>> scores;
        scores.reserve(slots_.size());
        
        for (Slot slot : slots_) {
            if (slot >= storage_->cpu_buffer_.size() || 
                storage_->cpu_buffer_[slot].empty()) {
                continue;
            }
            
            float sim = cosine_similarity(query, storage_->cpu_buffer_[slot].data(), dim_);
            scores.emplace_back(slot, sim);
        }
        
        // 取 top k
        std::partial_sort(scores.begin(), 
                          scores.begin() + std::min((size_t)k, scores.size()),
                          scores.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::vector<Slot> result;
        result.reserve(std::min((size_t)k, scores.size()));
        for (size_t i = 0; i < scores.size() && i < (size_t)k; ++i) {
            result.push_back(scores[i].first);
        }
        
        return result;
    }
    
    /**
     * 清空索引
     */
    virtual void clear() {
        slots_.clear();
        storage_ = std::make_unique<GPVectorStorage>(dim_, max_elements_);
    }
    
    size_t size() const { return slots_.size(); }

protected:
    size_t dim_;
    size_t max_elements_;
    std::unique_ptr<GPVectorStorage> storage_;
    std::vector<Slot> slots_;
};

// ============================================================================
// GPU IVF (Inverted File) 索引
// ============================================================================

/**
 * IVF 聚类中心
 */
struct IVFCluster {
    std::vector<float> centroid;
    std::vector<Slot> slots;  // 属于该聚类的向量
};

/**
 * GPU IVF 索引
 * CPU 训练聚类中心，GPU 存储倒排列表
 */
class GPUIVFIndex {
public:
    GPUIVFIndex(size_t dim, size_t nclusters = 100, size_t nprobe = 10)
        : dim_(dim), nclusters_(nclusters), nprobe_(nprobe) {
        clusters_.resize(nclusters);
        for (auto& c : clusters_) {
            c.centroid.resize(dim);
        }
    }
    
    virtual ~GPUIVFIndex() = default;
    
    /**
     * 训练聚类中心（在 CPU 上执行 K-means）
     */
    virtual void train(const std::vector<float>& vectors, size_t n) {
        // 简化的训练：随机选择中心点
        // 实际实现应使用 K-means++
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        
        for (auto& cluster : clusters_) {
            size_t idx = dist(rng);
            std::memcpy(cluster.centroid.data(), 
                       vectors.data() + idx * dim_, 
                       dim_ * sizeof(float));
        }
    }
    
    /**
     * 添加向量到 IVF
     */
    virtual void add(Slot slot, const float* vec) {
        // 找到最近的聚类
        int best_cluster = find_nearest_cluster(vec);
        clusters_[best_cluster].slots.push_back(slot);
        
        // 存储向量
        vectors_[slot] = std::vector<float>(vec, vec + dim_);
    }
    
    /**
     * 搜索
     * @param query: 查询向量
     * @param k: 返回结果数
     * @param nprobe: 搜索的聚类数（默认使用构造时的 nprobe）
     */
    virtual std::vector<Slot> search(const float* query, int k, int nprobe = -1) {
        if (nprobe < 0) nprobe = nprobe_;
        
        // 找到最近的 nprobe 个聚类
        std::vector<std::pair<int, float>> cluster_scores;
        cluster_scores.reserve(nclusters_);
        
        for (size_t i = 0; i < clusters_.size(); ++i) {
            float sim = cosine_similarity(query, clusters_[i].centroid.data(), dim_);
            cluster_scores.emplace_back(i, sim);
        }
        
        std::partial_sort(cluster_scores.begin(),
                          cluster_scores.begin() + std::min((size_t)nprobe, cluster_scores.size()),
                          cluster_scores.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // 在这些聚类中搜索
        std::vector<std::pair<Slot, float>> candidates;
        
        for (int i = 0; i < nprobe && i < (int)cluster_scores.size(); ++i) {
            int cid = cluster_scores[i].first;
            for (Slot slot : clusters_[cid].slots) {
                auto it = vectors_.find(slot);
                if (it != vectors_.end()) {
                    float sim = cosine_similarity(query, it->second.data(), dim_);
                    candidates.emplace_back(slot, sim);
                }
            }
        }
        
        // 取 top k
        std::partial_sort(candidates.begin(),
                          candidates.begin() + std::min((size_t)k, candidates.size()),
                          candidates.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::vector<Slot> result;
        result.reserve(std::min((size_t)k, candidates.size()));
        for (size_t i = 0; i < candidates.size() && i < (size_t)k; ++i) {
            result.push_back(candidates[i].first);
        }
        
        return result;
    }
    
    virtual void clear() {
        for (auto& c : clusters_) {
            c.slots.clear();
        }
        vectors_.clear();
    }
    
    virtual size_t size() const { return vectors_.size(); }

protected:
    int find_nearest_cluster(const float* vec) {
        int best = 0;
        float best_sim = cosine_similarity(vec, clusters_[0].centroid.data(), dim_);
        
        for (size_t i = 1; i < clusters_.size(); ++i) {
            float sim = cosine_similarity(vec, clusters_[i].centroid.data(), dim_);
            if (sim > best_sim) {
                best_sim = sim;
                best = i;
            }
        }
        
        return best;
    }

protected:
    size_t dim_;
    size_t nclusters_;
    size_t nprobe_;
    
    std::vector<IVFCluster> clusters_;
    std::unordered_map<Slot, std::vector<float>> vectors_;
};

} // namespace kvann
