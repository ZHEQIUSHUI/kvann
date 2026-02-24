#include <kvann/index.h>
#include <kvann/core.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kvann {

// ============================================================================
// 内部数据结构（非公开）
// ============================================================================

struct VectorMeta {
    Slot slot;
    bool tombstone;
    uint64_t version;
    std::vector<uint8_t> user_data;

    VectorMeta() : slot(INVALID_SLOT), tombstone(false), version(0) {}
    VectorMeta(Slot s) : slot(s), tombstone(false), version(1) {}
};

class VectorStorage {
public:
    VectorStorage(size_t dim, size_t initial_capacity = 1024)
        : dim_(dim), capacity_(initial_capacity), size_(0) {
        data_ = alloc_aligned_floats(capacity_ * dim_);
        if (!data_) {
            throw std::bad_alloc();
        }
    }

    ~VectorStorage() {
        free(data_);
    }

    VectorStorage(const VectorStorage&) = delete;
    VectorStorage& operator=(const VectorStorage&) = delete;

    VectorStorage(VectorStorage&& other) noexcept
        : data_(other.data_), dim_(other.dim_), capacity_(other.capacity_), size_(other.size_),
          free_slots_(std::move(other.free_slots_)) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    Slot allocate_slot() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (free_slots_.empty()) {
            if (size_ >= capacity_) {
                expand();
            }
            return size_++;
        }
        Slot slot = free_slots_.back();
        free_slots_.pop_back();
        return slot;
    }

    void release_slot(Slot slot) {
        std::unique_lock<std::mutex> lock(mutex_);
        free_slots_.push_back(slot);
    }

    void set_vector(Slot slot, const float* vec) {
        std::memcpy(data_ + slot * dim_, vec, dim_ * sizeof(float));
    }

    const float* get_vector(Slot slot) const {
        return data_ + slot * dim_;
    }

    float* get_vector(Slot slot) {
        return data_ + slot * dim_;
    }

    float* get_buffer(Slot slot) {
        return data_ + slot * dim_;
    }

    size_t dim() const { return dim_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

    void save(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        out.write(reinterpret_cast<const char*>(&size_), sizeof(size_));
        out.write(reinterpret_cast<const char*>(data_), size_ * dim_ * sizeof(float));
    }

    void load(std::ifstream& in) {
        in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
        in.read(reinterpret_cast<char*>(&size_), sizeof(size_));

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

    float* data_;
    size_t dim_;
    size_t capacity_;
    size_t size_;
    std::vector<Slot> free_slots_;
    mutable std::mutex mutex_;
};

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

    void put(Key key, const VectorMeta& meta) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        key_map_[key] = meta;
    }

    bool del(Key key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = key_map_.find(key);
        if (it != key_map_.end()) {
            it->second.tombstone = true;
            return true;
        }
        return false;
    }

    void update_meta(Key key, const VectorMeta& meta) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        key_map_[key] = meta;
    }

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

    void save(std::ofstream& out) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        size_t size = key_map_.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& [key, meta] : key_map_) {
            out.write(reinterpret_cast<const char*>(&key), sizeof(key));
            out.write(reinterpret_cast<const char*>(&meta.slot), sizeof(meta.slot));
            out.write(reinterpret_cast<const char*>(&meta.tombstone), sizeof(meta.tombstone));
            out.write(reinterpret_cast<const char*>(&meta.version), sizeof(meta.version));

            size_t user_data_size = meta.user_data.size();
            out.write(reinterpret_cast<const char*>(&user_data_size), sizeof(user_data_size));
            if (user_data_size > 0) {
                out.write(reinterpret_cast<const char*>(meta.user_data.data()), user_data_size);
            }
        }
    }

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

// ============================================================================
// HNSW 实现（内部）
// ============================================================================

struct HNSWNode {
    Slot slot;
    std::vector<std::vector<Slot>> neighbors;

    HNSWNode() : slot(INVALID_SLOT) {}
    explicit HNSWNode(Slot s, int max_level) : slot(s) {
        neighbors.resize(max_level + 1);
    }
};

class HNSWIndex {
public:
    HNSWIndex(size_t dim, size_t max_elements, int M = 16, int ef_construction = 200)
        : dim_(dim), max_elements_(max_elements), M_(M), M_max_(M),
          ef_construction_(ef_construction), enterpoint_(INVALID_SLOT),
          max_level_(-1), size_(0), vector_data_(nullptr) {

        nodes_.reserve(max_elements);
        rng_.seed(42);
        level_gen_ = std::geometric_distribution<int>(1.0 / std::log(M));
    }

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

    void set_vector_source(const VectorStorage* storage) {
        vector_data_ = storage;
    }

    void add(Slot slot) {
        if (size_ >= max_elements_) {
            throw std::runtime_error("HNSW index is full");
        }

        int level = random_level();

        std::unique_lock<std::shared_mutex> lock(global_mutex_);

        if (slot >= nodes_.size()) {
            nodes_.resize(slot + 1);
        }

        HNSWNode& node = nodes_[slot];
        node.slot = slot;
        node.neighbors.resize(level + 1);

        const float* vec = vector_data_->get_vector(slot);

        if (enterpoint_ == INVALID_SLOT) {
            enterpoint_ = slot;
            max_level_ = level;
            size_++;
            return;
        }

        Slot curr_ep = enterpoint_;
        for (int lc = max_level_; lc > level; --lc) {
            curr_ep = search_layer_simple(vec, curr_ep, lc);
        }

        for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
            auto candidates = search_layer(vec, curr_ep, ef_construction_, lc);
            auto neighbors = select_neighbors(vec, candidates, M_);

            for (Slot neighbor_slot : neighbors) {
                node.neighbors[lc].push_back(neighbor_slot);
                if (neighbor_slot < nodes_.size()) {
                    nodes_[neighbor_slot].neighbors[lc].push_back(slot);
                    shrink_connections(neighbor_slot, lc);
                }
            }

            if (!neighbors.empty()) {
                curr_ep = neighbors[0];
            }
        }

        if (level > max_level_) {
            max_level_ = level;
            enterpoint_ = slot;
        }

        size_++;
    }

    std::vector<std::pair<Slot, float>> search(
            const float* query,
            int k,
            int ef = 64,
            std::function<bool(Slot)> filter = nullptr) const {

        std::shared_lock<std::shared_mutex> lock(global_mutex_);

        if (enterpoint_ == INVALID_SLOT || size_ == 0) {
            return {};
        }

        Slot curr_ep = enterpoint_;
        for (int lc = max_level_; lc > 0; --lc) {
            curr_ep = search_layer_simple(query, curr_ep, lc);
        }

        auto candidates = search_layer(query, curr_ep, ef, 0, filter);

        std::vector<std::pair<Slot, float>> result;
        result.reserve(std::min(k, (int)candidates.size()));

        auto vec = candidates.top_vector();
        int count = 0;
        for (auto it = vec.begin(); it != vec.end() && count < k; ++it, ++count) {
            result.push_back(*it);
        }

        return result;
    }

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
    struct Candidate {
        Slot slot;
        float dist;

        bool operator>(const Candidate& other) const { return dist > other.dist; }
        bool operator<(const Candidate& other) const { return dist < other.dist; }
    };

    class SearchResultQueue {
    public:
        void push(Slot slot, float dist) {
            queue_.emplace_back(slot, dist);
        }

        std::vector<std::pair<Slot, float>> top_vector() const {
            auto sorted = queue_;
            std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            return sorted;
        }

        size_t size() const { return queue_.size(); }

    private:
        std::vector<std::pair<Slot, float>> queue_;
    };

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

    SearchResultQueue search_layer(
            const float* query,
            Slot enterpoint,
            int ef,
            int level,
            std::function<bool(Slot)> filter = nullptr) const {

        SearchResultQueue result;
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
        std::unordered_set<Slot> visited;
        std::priority_queue<Candidate> best;

        float enter_dist = distance(query, vector_data_->get_vector(enterpoint));
        candidates.push({enterpoint, enter_dist});
        visited.insert(enterpoint);
        best.push({enterpoint, enter_dist});

        while (!candidates.empty()) {
            auto curr = candidates.top();
            candidates.pop();

            if (!best.empty() && curr.dist > best.top().dist) {
                break;
            }

            if (curr.slot >= nodes_.size()) continue;

            const auto& neighbors = nodes_[curr.slot].neighbors;
            if (level >= (int)neighbors.size()) continue;

            for (Slot neighbor : neighbors[level]) {
                if (neighbor == INVALID_SLOT || visited.count(neighbor)) continue;

                visited.insert(neighbor);
                float dist = distance(query, vector_data_->get_vector(neighbor));

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

        while (!best.empty()) {
            auto c = best.top();
            best.pop();
            result.push(c.slot, c.dist);
        }

        return result;
    }

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

    void shrink_connections(Slot slot, int level) {
        if (slot >= nodes_.size()) return;

        auto& neighbors = nodes_[slot].neighbors[level];
        int M_max = (level == 0) ? M_max_ * 2 : M_max_;

        if ((int)neighbors.size() > M_max) {
            neighbors.resize(M_max);
        }
    }

    float distance(const float* a, const float* b) const {
        float sim = cosine_similarity(a, b, dim_);
        return 1.0f - sim;
    }

    int random_level() {
        int level = level_gen_(rng_);
        if (level < 0) level = 0;
        int max_possible_level = 16;
        if (level > max_possible_level) level = max_possible_level;
        return level;
    }

    size_t dim_;
    size_t max_elements_;
    int M_;
    int M_max_;
    int ef_construction_;

    Slot enterpoint_;
    int max_level_;
    size_t size_;

    std::vector<HNSWNode> nodes_;
    const VectorStorage* vector_data_;

    mutable std::shared_mutex global_mutex_;

    std::mt19937 rng_;
    std::geometric_distribution<int> level_gen_;
};

// ============================================================================
// Index 实现
// ============================================================================

struct Index::Impl {
    Impl(size_t dim, size_t max_elements, size_t delta_threshold)
        : dim_(dim),
          max_elements_(max_elements),
          delta_threshold_(delta_threshold),
          storage_(dim, max_elements),
          base_index_(dim, max_elements),
          rebuild_running_(false) {
        base_index_.set_vector_source(&storage_);
    }

    ~Impl() {
        if (rebuild_thread_.joinable()) {
            rebuild_thread_.join();
        }
    }

    bool put_with_data(Key key, const float* vector, const void* user_data, size_t user_data_len) {
        Vector normalized_vec(vector, vector + dim_);
        normalize_vector(normalized_vec.data(), dim_);

        std::unique_lock<std::shared_mutex> lock(write_mutex_);

        auto existing = key_manager_.get_meta(key);

        if (existing && !existing->tombstone) {
            Slot slot = existing->slot;
            storage_.set_vector(slot, normalized_vec.data());

            VectorMeta meta = *existing;
            meta.version++;
            if (user_data && user_data_len > 0) {
                const uint8_t* data_ptr = static_cast<const uint8_t*>(user_data);
                meta.user_data.assign(data_ptr, data_ptr + user_data_len);
            } else {
                meta.user_data.clear();
            }
            key_manager_.update_meta(key, meta);

            delta_keys_.insert(key);
            delta_slots_[key] = slot;

            return true;
        }

        Slot slot;
        if (existing) {
            slot = existing->slot;
        } else {
            slot = storage_.allocate_slot();
        }

        storage_.set_vector(slot, normalized_vec.data());

        VectorMeta meta(slot);
        if (user_data && user_data_len > 0) {
            const uint8_t* data_ptr = static_cast<const uint8_t*>(user_data);
            meta.user_data.assign(data_ptr, data_ptr + user_data_len);
        }
        key_manager_.put(key, meta);

        delta_keys_.insert(key);
        delta_slots_[key] = slot;

        return true;
    }

    bool del(Key key) {
        std::unique_lock<std::shared_mutex> lock(write_mutex_);

        if (!key_manager_.exists(key)) {
            return false;
        }

        key_manager_.del(key);
        delta_keys_.erase(key);
        delta_slots_.erase(key);
        return true;
    }

    bool exists(Key key) const {
        auto meta = key_manager_.get_meta(key);
        return meta.has_value() && !meta->tombstone;
    }

    std::vector<uint8_t> get_user_data(Key key) const {
        auto meta = key_manager_.get_meta(key);
        if (meta && !meta->tombstone) {
            return meta->user_data;
        }
        return {};
    }

    std::vector<SearchResult> search(const float* query, int topk) {
        Vector normalized_query(query, query + dim_);
        normalize_vector(normalized_query.data(), dim_);

        std::shared_lock<std::shared_mutex> lock(write_mutex_);

        std::vector<std::pair<Slot, float>> candidates;

        if (!base_index_.empty()) {
            auto base_candidates = base_index_.search(
                normalized_query.data(),
                topk * 2,
                64
            );
            candidates.insert(candidates.end(), base_candidates.begin(), base_candidates.end());
        }

        search_delta_brute_force(normalized_query.data(), topk * 2, candidates);

        return rerank(normalized_query.data(), candidates, topk);
    }

    void rebuild() {
        std::unique_lock<std::shared_mutex> lock(write_mutex_);

        if (rebuild_running_.load()) {
            return;
        }

        rebuild_running_ = true;
        lock.unlock();

        if (rebuild_thread_.joinable()) {
            rebuild_thread_.join();
        }

        rebuild_thread_ = std::thread([this]() { do_rebuild(); });
    }

    void wait_rebuild() {
        if (rebuild_thread_.joinable()) {
            rebuild_thread_.join();
        }
    }

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

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open file for writing: " + path);
        }

        size_t dim = dim_;
        size_t max_elements = max_elements_;
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(&max_elements), sizeof(max_elements));

        key_manager_.save(out);
        storage_.save(out);

        out.close();
    }

    void load_from_stream(std::ifstream& in) {
        key_manager_.load(in);
        storage_.load(in);
        rebuild_base_from_kv();
    }

private:
    void search_delta_brute_force(const float* query, int /* topk */,
                                  std::vector<std::pair<Slot, float>>& results) const {
        for (Key key : delta_keys_) {
            auto it = delta_slots_.find(key);
            if (it == delta_slots_.end()) continue;

            Slot slot = it->second;
            const float* vec = storage_.get_vector(slot);
            float sim = cosine_similarity(query, vec, dim_);

            results.emplace_back(slot, 1.0f - sim);
        }
    }

    std::vector<SearchResult> rerank(const float* query,
                                     const std::vector<std::pair<Slot, float>>& candidates,
                                     int topk) {
        std::unordered_set<Slot> seen;
        std::vector<std::pair<Slot, float>> unique_candidates;

        for (const auto& [slot, _] : candidates) {
            if (seen.insert(slot).second) {
                const float* vec = storage_.get_vector(slot);
                float sim = cosine_similarity(query, vec, dim_);
                unique_candidates.emplace_back(slot, sim);
            }
        }

        std::partial_sort(unique_candidates.begin(),
                          unique_candidates.begin() + std::min((size_t)topk, unique_candidates.size()),
                          unique_candidates.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });

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

    void do_rebuild() {
        std::cout << "[rebuild] Starting..." << std::endl;

        auto live_keys = key_manager_.get_all_live();

        HNSWIndex new_base(dim_, max_elements_);
        new_base.set_vector_source(&storage_);

        for (const auto& [key, slot] : live_keys) {
            new_base.add(slot);
        }

        std::unique_lock<std::shared_mutex> lock(write_mutex_);
        base_index_ = std::move(new_base);
        delta_keys_.clear();
        delta_slots_.clear();

        rebuild_running_ = false;
        std::cout << "[rebuild] Done. Base size: " << base_index_.size() << std::endl;
    }

    void rebuild_base_from_kv() {
        base_index_.clear();

        auto live_keys = key_manager_.get_all_live();
        for (const auto& [key, slot] : live_keys) {
            base_index_.add(slot);
        }

        delta_keys_.clear();
        delta_slots_.clear();
    }

private:
    size_t dim_;
    size_t max_elements_;
    size_t delta_threshold_;

    VectorStorage storage_;
    KeyManager key_manager_;

    HNSWIndex base_index_;

    std::unordered_set<Key> delta_keys_;
    std::unordered_map<Key, Slot> delta_slots_;

    mutable std::shared_mutex write_mutex_;

    std::atomic<bool> rebuild_running_;
    std::thread rebuild_thread_;
};

// ============================================================================
// Index 公共接口
// ============================================================================

Index::Index(size_t dim, size_t max_elements, size_t delta_threshold)
    : impl_(std::make_unique<Impl>(dim, max_elements, delta_threshold)) {}

Index::~Index() = default;

Index::Index(Index&& other) noexcept = default;
Index& Index::operator=(Index&& other) noexcept = default;

bool Index::put(Key key, const float* vector) {
    return impl_->put_with_data(key, vector, nullptr, 0);
}

bool Index::put_with_data(Key key, const float* vector, const void* user_data, size_t user_data_len) {
    return impl_->put_with_data(key, vector, user_data, user_data_len);
}

bool Index::del(Key key) {
    return impl_->del(key);
}

bool Index::exists(Key key) const {
    return impl_->exists(key);
}

std::vector<uint8_t> Index::get_user_data(Key key) const {
    return impl_->get_user_data(key);
}

std::vector<SearchResult> Index::search(const float* query, int topk) {
    return impl_->search(query, topk);
}

void Index::rebuild() {
    impl_->rebuild();
}

void Index::wait_rebuild() {
    impl_->wait_rebuild();
}

IndexStats Index::stats() const {
    return impl_->stats();
}

void Index::save(const std::string& path) const {
    impl_->save(path);
}

std::unique_ptr<Index> Index::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }

    size_t dim, max_elements;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&max_elements), sizeof(max_elements));

    auto index = std::make_unique<Index>(dim, max_elements);
    index->impl_->load_from_stream(in);
    return index;
}

} // namespace kvann
