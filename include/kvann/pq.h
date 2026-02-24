/**
 * kvann - Product Quantization (PQ) 实现
 * V3 版本
 * 
 * PQ 流程：
 * 1. 将向量分成 M 个子空间
 * 2. 每个子空间用 K-means 训练出 k 个聚类中心（码本）
 * 3. 每个子空间用最近的聚类中心索引（通常 8 位）表示
 * 
 * 查询：
 * 1. ADC (Asymmetric Distance Computation): 查询向量不量化，与量化后的数据库向量计算距离
 * 2. 使用查找表加速距离计算
 */

#pragma once

#include "core.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace kvann {

// ============================================================================
// PQ 码本
// ============================================================================

/**
 * 子空间码本
 */
struct PQCodebook {
    int M;              // 子空间数量
    int ksub;           // 每个子空间的聚类数（通常是 256，用 8 位表示）
    int dim;            // 原始维度
    int dsub;           // 每个子空间的维度 = dim / M
    
    // centroids[m][ksub][dsub]: 第 m 个子空间的第 i 个聚类中心
    std::vector<std::vector<std::vector<float>>> centroids;
    
    PQCodebook() : M(0), ksub(0), dim(0), dsub(0) {}
    PQCodebook(int M_, int ksub_, int dim_) : M(M_), ksub(ksub_), dim(dim_) {
        dsub = dim / M;
        centroids.resize(M);
        for (int m = 0; m < M; ++m) {
            centroids[m].resize(ksub);
            for (int i = 0; i < ksub; ++i) {
                centroids[m][i].resize(dsub);
            }
        }
    }
    
    /**
     * 获取码本大小（字节）
     */
    size_t size_bytes() const {
        return M * ksub * dsub * sizeof(float);
    }
};

// ============================================================================
// PQ 量化器
// ============================================================================

class ProductQuantizer {
public:
    ProductQuantizer(int M = 8, int ksub = 256, int dim = 128)
        : M_(M), ksub_(ksub), dim_(dim), trained_(false) {
        dsub_ = dim / M;
        codebook_ = PQCodebook(M, ksub, dim);
    }
    
    /**
     * 训练码本
     * @param vectors: 训练向量集，大小为 n * dim
     * @param n: 向量数量
     */
    void train(const float* vectors, size_t n, int niter = 10) {
        if (n == 0) return;
        
        std::mt19937 rng(42);
        
        // 对每个子空间分别训练 K-means
        for (int m = 0; m < M_; ++m) {
            train_subspace(m, vectors, n, niter, rng);
        }
        
        trained_ = true;
    }
    
    /**
     * 量化单个向量
     * @return 量化后的码字（M 个字节）
     */
    std::vector<uint8_t> encode(const float* vec) const {
        std::vector<uint8_t> codes(M_);
        
        for (int m = 0; m < M_; ++m) {
            codes[m] = find_nearest_centroid(m, vec + m * dsub_);
        }
        
        return codes;
    }
    
    /**
     * 批量量化
     */
    std::vector<uint8_t> encode_batch(const float* vectors, size_t n) const {
        std::vector<uint8_t> codes(n * M_);
        
        for (size_t i = 0; i < n; ++i) {
            auto code = encode(vectors + i * dim_);
            std::memcpy(codes.data() + i * M_, code.data(), M_);
        }
        
        return codes;
    }
    
    /**
     * 解码（近似重建）
     */
    std::vector<float> decode(const uint8_t* codes) const {
        std::vector<float> vec(dim_);
        
        for (int m = 0; m < M_; ++m) {
            const auto& centroid = codebook_.centroids[m][codes[m]];
            std::memcpy(vec.data() + m * dsub_, centroid.data(), dsub_ * sizeof(float));
        }
        
        return vec;
    }
    
    /**
     * ADC (Asymmetric Distance Computation)
     * 计算查询向量与量化向量的距离
     * @param query: 查询向量（未量化）
     * @param codes: 量化后的码字
     * @return 近似距离（平方欧氏距离）
     */
    float adc_distance(const float* query, const uint8_t* codes) const {
        float dist = 0;
        
        for (int m = 0; m < M_; ++m) {
            const auto& centroid = codebook_.centroids[m][codes[m]];
            for (int d = 0; d < dsub_; ++d) {
                float diff = query[m * dsub_ + d] - centroid[d];
                dist += diff * diff;
            }
        }
        
        return dist;
    }
    
    /**
     * 预计算距离查找表
     * 用于加速批量 ADC 计算
     */
    std::vector<float> compute_distance_table(const float* query) const {
        std::vector<float> table(M_ * ksub_);
        
        for (int m = 0; m < M_; ++m) {
            for (int k = 0; k < ksub_; ++k) {
                const auto& centroid = codebook_.centroids[m][k];
                float dist = 0;
                for (int d = 0; d < dsub_; ++d) {
                    float diff = query[m * dsub_ + d] - centroid[d];
                    dist += diff * diff;
                }
                table[m * ksub_ + k] = dist;
            }
        }
        
        return table;
    }
    
    /**
     * 使用查找表计算距离（更快）
     */
    float adc_distance_with_table(const std::vector<float>& table, 
                                   const uint8_t* codes) const {
        float dist = 0;
        
        for (int m = 0; m < M_; ++m) {
            dist += table[m * ksub_ + codes[m]];
        }
        
        return dist;
    }
    
    bool is_trained() const { return trained_; }
    int M() const { return M_; }
    int ksub() const { return ksub_; }
    int dim() const { return dim_; }
    const PQCodebook& codebook() const { return codebook_; }

private:
    void train_subspace(int m, const float* vectors, size_t n, 
                        int niter, std::mt19937& rng) {
        // 提取子空间向量
        std::vector<float> sub_vectors(n * dsub_);
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(sub_vectors.data() + i * dsub_,
                       vectors + i * dim_ + m * dsub_,
                       dsub_ * sizeof(float));
        }
        
        // 随机初始化聚类中心
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        for (int k = 0; k < ksub_; ++k) {
            size_t idx = dist(rng);
            std::memcpy(codebook_.centroids[m][k].data(),
                       sub_vectors.data() + idx * dsub_,
                       dsub_ * sizeof(float));
        }
        
        // K-means 迭代
        std::vector<std::vector<float>> new_centroids(ksub_, std::vector<float>(dsub_, 0));
        std::vector<int> counts(ksub_, 0);
        
        for (int iter = 0; iter < niter; ++iter) {
            // 清空
            for (int k = 0; k < ksub_; ++k) {
                std::fill(new_centroids[k].begin(), new_centroids[k].end(), 0);
                counts[k] = 0;
            }
            
            // 分配
            for (size_t i = 0; i < n; ++i) {
                int best = find_nearest_centroid_in_subspace(
                    m, sub_vectors.data() + i * dsub_);
                
                for (int d = 0; d < dsub_; ++d) {
                    new_centroids[best][d] += sub_vectors[i * dsub_ + d];
                }
                counts[best]++;
            }
            
            // 更新
            for (int k = 0; k < ksub_; ++k) {
                if (counts[k] > 0) {
                    for (int d = 0; d < dsub_; ++d) {
                        codebook_.centroids[m][k][d] = new_centroids[k][d] / counts[k];
                    }
                }
            }
        }
    }
    
    int find_nearest_centroid(int m, const float* sub_vec) const {
        return find_nearest_centroid_in_subspace(m, sub_vec);
    }
    
    int find_nearest_centroid_in_subspace(int m, const float* sub_vec) const {
        int best = 0;
        float best_dist = subspace_distance(m, sub_vec, codebook_.centroids[m][0].data());
        
        for (int k = 1; k < ksub_; ++k) {
            float dist = subspace_distance(m, sub_vec, codebook_.centroids[m][k].data());
            if (dist < best_dist) {
                best_dist = dist;
                best = k;
            }
        }
        
        return best;
    }
    
    float subspace_distance(int m, const float* a, const float* b) const {
        float dist = 0;
        for (int d = 0; d < dsub_; ++d) {
            float diff = a[d] - b[d];
            dist += diff * diff;
        }
        return dist;
    }

private:
    int M_;             // 子空间数
    int ksub_;          // 每子空间聚类数
    int dim_;           // 原始维度
    int dsub_;          // 子空间维度
    bool trained_;
    
    PQCodebook codebook_;
};

// ============================================================================
// PQ 索引
// ============================================================================

/**
 * PQ 索引
 * 存储量化后的码字，使用 ADC 进行近似距离计算
 */
class PQIndex {
public:
    PQIndex(int M = 8, int ksub = 256, int dim = 128)
        : pq_(M, ksub, dim), dim_(dim) {}
    
    /**
     * 训练
     */
    void train(const std::vector<float>& vectors, size_t n) {
        pq_.train(vectors.data(), n);
    }
    
    /**
     * 添加向量
     */
    void add(Slot slot, const float* vec) {
        auto codes = pq_.encode(vec);
        codes_[slot] = std::move(codes);
    }
    
    /**
     * 搜索
     * @return 候选 slot 列表（需要回原始向量 rerank）
     */
    std::vector<Slot> search(const float* query, int k) {
        // 预计算距离表
        auto table = pq_.compute_distance_table(query);
        
        // 计算所有向量的近似距离
        std::vector<std::pair<Slot, float>> scores;
        scores.reserve(codes_.size());
        
        for (const auto& [slot, codes] : codes_) {
            float dist = pq_.adc_distance_with_table(table, codes.data());
            scores.emplace_back(slot, dist);
        }
        
        // 取 top k（PQ 使用距离，越小越近）
        std::partial_sort(scores.begin(),
                          scores.begin() + std::min((size_t)k * 2, scores.size()),
                          scores.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });
        
        std::vector<Slot> result;
        result.reserve(std::min((size_t)k, scores.size()));
        for (size_t i = 0; i < scores.size() && i < (size_t)k; ++i) {
            result.push_back(scores[i].first);
        }
        
        return result;
    }
    
    /**
     * 解码向量（近似）
     */
    std::vector<float> reconstruct(Slot slot) const {
        auto it = codes_.find(slot);
        if (it != codes_.end()) {
            return pq_.decode(it->second.data());
        }
        return {};
    }
    
    size_t size() const { return codes_.size(); }
    bool is_trained() const { return pq_.is_trained(); }

private:
    ProductQuantizer pq_;
    int dim_;
    std::unordered_map<Slot, std::vector<uint8_t>> codes_;
};

} // namespace kvann
