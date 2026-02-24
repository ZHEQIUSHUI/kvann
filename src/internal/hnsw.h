/**
 * kvann - HNSW 索引实现
 * V1 版本：基础HNSW实现
 */

#pragma once

#include "core.h"
#include <vector>
#include <random>
#include <queue>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <mutex>

namespace kvann {

/**
 * HNSW 节点
 */
struct HNSWNode {
    Slot slot;
    std::vector<std::vector<Slot>> neighbors;  // 每层邻居
    
    HNSWNode() : slot(INVALID_SLOT) {}
    explicit HNSWNode(Slot s, int max_level) : slot(s) {
        neighbors.resize(max_level + 1);
    }
};

/**
 * 简单HNSW索引实现
 * 
 * 参数说明：
 * - M: 每层最大邻居数（默认16）
 * - ef_construction: 构建时搜索范围（默认200）
 * - max_elements: 最大元素数
 */
class HNSWIndex {
public:
    HNSWIndex(size_t dim, size_t max_elements, int M = 16, int ef_construction = 200)
        : dim_(dim), max_elements_(max_elements), M_(M), M_max_(M),
          ef_construction_(ef_construction), enterpoint_(INVALID_SLOT),
          max_level_(-1), size_(0), vector_data_(nullptr) {
        
        // 预分配节点空间
        nodes_.reserve(max_elements);
        
        // 随机数生成器
        rng_.seed(42);
        level_gen_ = std::geometric_distribution<int>(1.0 / std::log(M));
    }
    
    ~HNSWIndex() = default;
    
    // 禁止拷贝，允许移动
    HNSWIndex(const HNSWIndex&) = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;
    
    HNSWIndex(HNSWIndex&& other) noexcept
        : dim_(other.dim_), max_elements_(other.max_elements_), M_(other.M_),
          M_max_(other.M_max_), ef_construction_(other.ef_construction_),
          enterpoint_(other.enterpoint_), max_level_(other.max_level_),
          size_(other.size_), nodes_(std::move(other.nodes_)),
          vector_data_(other.vector_data_), rng_(other.rng_),
          level_gen_(other.level_gen_) {
        other.size_ = 0;
        other.enterpoint_ = INVALID_SLOT;
        other.max_level_ = -1;
    }
    
    HNSWIndex& operator=(HNSWIndex&& other) noexcept {
        if (this != &other) {
            dim_ = other.dim_;
            max_elements_ = other.max_elements_;
            M_ = other.M_;
            M_max_ = other.M_max_;
            ef_construction_ = other.ef_construction_;
            enterpoint_ = other.enterpoint_;
            max_level_ = other.max_level_;
            size_ = other.size_;
            nodes_ = std::move(other.nodes_);
            vector_data_ = other.vector_data_;
            rng_ = other.rng_;
            level_gen_ = other.level_gen_;
            
            other.size_ = 0;
            other.enterpoint_ = INVALID_SLOT;
            other.max_level_ = -1;
        }
        return *this;
    }
    
    /**
     * 设置向量数据源（不拷贝向量，只存储引用）
     */
    void set_vector_source(const VectorStorage* storage) {
        vector_data_ = storage;
    }
    
    /**
     * 添加向量
     */
    void add(Slot slot) {
        if (size_ >= max_elements_) {
            throw std::runtime_error("HNSW index is full");
        }
        
        int level = random_level();
        
        std::unique_lock<std::shared_mutex> lock(global_mutex_);
        
        // 确保节点空间足够
        if (slot >= nodes_.size()) {
            nodes_.resize(slot + 1);
        }
        
        HNSWNode& node = nodes_[slot];
        node.slot = slot;
        node.neighbors.resize(level + 1);
        
        const float* vec = vector_data_->get_vector(slot);
        
        // 第一个节点
        if (enterpoint_ == INVALID_SLOT) {
            enterpoint_ = slot;
            max_level_ = level;
            size_++;
            return;
        }
        
        // 从顶层开始搜索
        Slot curr_ep = enterpoint_;
        for (int lc = max_level_; lc > level; --lc) {
            curr_ep = search_layer_simple(vec, curr_ep, lc);
        }
        
        // 从当前层向下处理每一层
        for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
            auto candidates = search_layer(vec, curr_ep, ef_construction_, lc);
            auto neighbors = select_neighbors(vec, candidates, M_);
            
            // 添加双向连接
            for (Slot neighbor_slot : neighbors) {
                node.neighbors[lc].push_back(neighbor_slot);
                if (neighbor_slot < nodes_.size()) {
                    nodes_[neighbor_slot].neighbors[lc].push_back(slot);
                    
                    // 收缩邻居列表
                    shrink_connections(neighbor_slot, lc);
                }
            }
            
            if (!neighbors.empty()) {
                curr_ep = neighbors[0];
            }
        }
        
        // 更新入口点和最大层
        if (level > max_level_) {
            max_level_ = level;
            enterpoint_ = slot;
        }
        
        size_++;
    }
    
    /**
     * 搜索K近邻
     * @param query: 查询向量（已归一化）
     * @param k: 返回结果数
     * @param ef: 搜索范围
     * @param filter: slot过滤函数（返回true表示保留）
     */
    std::vector<std::pair<Slot, float>> search(
            const float* query, 
            int k, 
            int ef = 64,
            std::function<bool(Slot)> filter = nullptr) const {
        
        std::shared_lock<std::shared_mutex> lock(global_mutex_);
        
        if (enterpoint_ == INVALID_SLOT || size_ == 0) {
            return {};
        }
        
        // 从顶层开始搜索
        Slot curr_ep = enterpoint_;
        for (int lc = max_level_; lc > 0; --lc) {
            curr_ep = search_layer_simple(query, curr_ep, lc);
        }
        
        // 底层搜索
        auto candidates = search_layer(query, curr_ep, ef, 0, filter);
        
        // 取top k（距离最小的k个）
        std::vector<std::pair<Slot, float>> result;
        result.reserve(std::min(k, (int)candidates.size()));
        
        auto vec = candidates.top_vector();
        int count = 0;
        for (auto it = vec.begin(); it != vec.end() && count < k; ++it, ++count) {
            result.push_back(*it);
        }
        
        return result;
    }
    
    /**
     * 批量获取候选（用于rerank）
     */
    std::vector<Slot> get_candidates(const float* query, int n, int ef = 100) const {
        auto results = search(query, n, ef);
        std::vector<Slot> candidates;
        candidates.reserve(results.size());
        for (const auto& [slot, score] : results) {
            candidates.push_back(slot);
        }
        return candidates;
    }
    
    /**
     * 清空索引
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(global_mutex_);
        nodes_.clear();
        enterpoint_ = INVALID_SLOT;
        max_level_ = -1;
        size_ = 0;
    }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

private:
    /**
     * 候选队列（最小堆，按距离排序）
     */
    struct Candidate {
        Slot slot;
        float dist;
        
        bool operator>(const Candidate& other) const {
            return dist > other.dist;
        }
        
        bool operator<(const Candidate& other) const {
            return dist < other.dist;
        }
    };
    
    /**
     * 搜索结果队列（最大堆变体）
     */
    class SearchResultQueue {
    public:
        void push(Slot slot, float dist) {
            queue_.emplace_back(slot, dist);
        }
        
        std::vector<std::pair<Slot, float>> top_vector() const {
            auto sorted = queue_;
            // 按距离升序排序（距离越小越近）
            std::sort(sorted.begin(), sorted.end(), 
                [](const auto& a, const auto& b) { return a.second < b.second; });
            return sorted;
        }
        
        size_t size() const { return queue_.size(); }
        
    private:
        std::vector<std::pair<Slot, float>> queue_;
    };
    
    /**
     * 单层简单搜索（返回最近点）
     */
    Slot search_layer_simple(const float* query, Slot enterpoint, int level) const {
        Slot curr = enterpoint;
        float curr_dist = distance(query, vector_data_->get_vector(curr));
        
        bool changed = true;
        while (changed) {
            changed = false;
            if (curr >= nodes_.size()) break;
            
            const auto& neighbors = nodes_[curr].neighbors;
            if (level >= (int)neighbors.size()) continue;
            
            for (Slot neighbor : neighbors[level]) {
                if (neighbor == INVALID_SLOT) continue;
                
                float dist = distance(query, vector_data_->get_vector(neighbor));
                if (dist < curr_dist) {
                    curr = neighbor;
                    curr_dist = dist;
                    changed = true;
                }
            }
        }
        
        return curr;
    }
    
    /**
     * 单层搜索（返回多个候选）
     */
    SearchResultQueue search_layer(
            const float* query, 
            Slot enterpoint, 
            int ef, 
            int level,
            std::function<bool(Slot)> filter = nullptr) const {
        
        SearchResultQueue result;
        
        // 候选队列（最小堆）
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
        // 已访问集合
        std::unordered_set<Slot> visited;
        // 结果队列（最大堆，保存最近的ef个）
        std::priority_queue<Candidate> best;
        
        float enter_dist = distance(query, vector_data_->get_vector(enterpoint));
        candidates.push({enterpoint, enter_dist});
        visited.insert(enterpoint);
        best.push({enterpoint, enter_dist});  // enterpoint也是候选
        
        while (!candidates.empty()) {
            auto curr = candidates.top();
            candidates.pop();
            
            // 如果当前距离已经大于结果中最差的，退出
            if (!best.empty() && curr.dist > best.top().dist) {
                break;
            }
            
            // 处理当前节点的邻居
            if (curr.slot >= nodes_.size()) continue;
            
            const auto& neighbors = nodes_[curr.slot].neighbors;
            if (level >= (int)neighbors.size()) continue;
            
            for (Slot neighbor : neighbors[level]) {
                if (neighbor == INVALID_SLOT || visited.count(neighbor)) continue;
                
                visited.insert(neighbor);
                float dist = distance(query, vector_data_->get_vector(neighbor));
                
                // 应用过滤
                if (filter && !filter(neighbor)) continue;
                
                if (best.size() < (size_t)ef || dist < best.top().dist) {
                    candidates.push({neighbor, dist});
                    best.push({neighbor, dist});
                    if (best.size() > (size_t)ef) {
                        best.pop();
                    }
                }
            }
        }
        
        // 转换为结果
        while (!best.empty()) {
            auto c = best.top();
            best.pop();
            result.push(c.slot, c.dist);
        }
        
        return result;
    }
    
    /**
     * 选择邻居（简单启发式：选最近的M个）
     */
    std::vector<Slot> select_neighbors(const float* /* query */, 
                                       const SearchResultQueue& candidates, 
                                       int M) const {
        auto vec = candidates.top_vector();
        std::vector<Slot> result;
        result.reserve(std::min((size_t)M, vec.size()));
        for (size_t i = 0; i < vec.size() && i < (size_t)M; ++i) {
            result.push_back(vec[i].first);
        }
        return result;
    }
    
    /**
     * 收缩连接（当邻居太多时）
     */
    void shrink_connections(Slot slot, int level) {
        if (slot >= nodes_.size()) return;
        
        auto& neighbors = nodes_[slot].neighbors[level];
        int M_max = (level == 0) ? M_max_ * 2 : M_max_;
        
        if ((int)neighbors.size() > M_max) {
            // 简单的收缩策略：保留前M_max个
            neighbors.resize(M_max);
        }
    }
    
    /**
     * 距离计算（余弦距离 = 1 - 余弦相似度）
     */
    float distance(const float* a, const float* b) const {
        // 使用余弦相似度，转换为距离（越小越近）
        float sim = cosine_similarity(a, b, dim_);
        return 1.0f - sim;
    }
    
    /**
     * 随机层数生成
     */
    int random_level() {
        int level = level_gen_(rng_);
        if (level < 0) level = 0;
        // 限制最大层数
        int max_possible_level = 16;
        if (level > max_possible_level) level = max_possible_level;
        return level;
    }

private:
    size_t dim_;                    // 向量维度
    size_t max_elements_;           // 最大元素数
    int M_;                         // 每层邻居数
    int M_max_;                     // 最大邻居数
    int ef_construction_;           // 构建搜索范围
    
    Slot enterpoint_;               // 入口点
    int max_level_;                 // 最大层数
    size_t size_;                   // 当前大小
    
    std::vector<HNSWNode> nodes_;   // 节点数组
    const VectorStorage* vector_data_;  // 向量数据源
    
    mutable std::shared_mutex global_mutex_;  // 全局锁（V1使用简单锁）
    
    std::mt19937 rng_;
    std::geometric_distribution<int> level_gen_;
};

} // namespace kvann
