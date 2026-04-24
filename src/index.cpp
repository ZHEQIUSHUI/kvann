// kvann - Index implementation
//
// Internal layering (top to bottom):
//   Index::Impl                  orchestration / public API
//   HnswGraph                    HNSW with arena-based layer-0 neighbors
//   DeltaSet                     alive-bitmap + member list for delta layer
//   KeyDir                       sharded Key -> {Slot, payload, version}
//   SlotKeyMap                   dense Slot -> Key (atomic)
//   VectorStore                  aligned per-slot vectors with seqlock writes
//   VisitedPool                  thread-local epoch-tagged visited buffers
//
// Concurrency model:
//   * Reads (search, get_payload, exists)  use shared/atomic paths.
//   * Writes (put, del)                    take per-stripe write locks for KeyDir
//                                          and per-slot mutexes for HNSW updates.
//   * VectorStore writes use a seqlock so concurrent searches never observe
//     torn data without retrying.
//   * rebuild() takes a snapshot of (key, slot, vector) so the new graph builds
//     against frozen data; live writes keep flowing into delta.

#include <kvann/core.h>
#include <kvann/index.h>
#include <kvann/status.h>
#include <kvann/detail/alloc.h>
#include <kvann/detail/arch.h>
#include <kvann/detail/log.h>
#include <kvann/detail/simd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kvann {
namespace {

// ============================================================================
// Helpers
// ============================================================================
constexpr size_t kVecAlign = 64;

inline size_t round_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

inline void emit_log(const IndexConfig& cfg, const char* level, const char* msg) {
    if (cfg.log_sink) cfg.log_sink(level, msg);
}

// ============================================================================
// VectorStore
//   - aligned 64B blocks
//   - per-slot seqlock so concurrent reads can detect torn writes and retry
//   - dot_with(slot, query) returns the SIMD dot product safely w/o copying
// ============================================================================
class VectorStore {
public:
    VectorStore(size_t dim, size_t max_elements, size_t block_size)
        : dim_(dim),
          dim_padded_(round_up(dim, 8)),  // 32-byte / 8-float alignment for AVX
          block_size_(block_size ? block_size : 4096),
          max_elements_(max_elements),
          version_(max_elements) {
        // Pre-allocate enough blocks to cover max_elements lazily on demand.
        size_t needed = (max_elements_ + block_size_ - 1) / block_size_;
        blocks_.reserve(needed);
        for (auto& v : version_) v.store(0, std::memory_order_relaxed);
    }

    ~VectorStore() {
        for (auto* b : blocks_) detail::aligned_free(b);
    }

    VectorStore(const VectorStore&) = delete;
    VectorStore& operator=(const VectorStore&) = delete;

    size_t dim() const { return dim_; }
    size_t max_elements() const { return max_elements_; }

    void set_vector(Slot slot, const float* src) {
        ensure_block(slot);
        auto& v = version_[slot];
        v.fetch_add(1, std::memory_order_acq_rel);  // -> odd (writing)
        float* dst = slot_ptr(slot);
        std::memcpy(dst, src, dim_ * sizeof(float));
        // pad bytes don't need init for dot; we sized arena to dim_padded_.
        v.fetch_add(1, std::memory_order_release);  // -> even (stable)
    }

    // Optimistic SIMD dot. Retries if a writer touched the slot mid-read.
    KVANN_FORCE_INLINE float dot_with(Slot slot, const float* query) const {
        auto& v = version_[slot];
        for (;;) {
            uint32_t v0 = v.load(std::memory_order_acquire);
            if (KVANN_UNLIKELY(v0 & 1u)) continue;
            float d = simd::dot_f32(slot_ptr_const(slot), query, dim_);
            std::atomic_thread_fence(std::memory_order_acquire);
            uint32_t v1 = v.load(std::memory_order_acquire);
            if (KVANN_LIKELY(v0 == v1)) return d;
        }
    }

    // Snapshot copy for rebuild: stable since we hold KeyDir read locks externally.
    void copy_vector(Slot slot, float* dst) const {
        auto& v = version_[slot];
        for (;;) {
            uint32_t v0 = v.load(std::memory_order_acquire);
            if (v0 & 1u) continue;
            std::memcpy(dst, slot_ptr_const(slot), dim_ * sizeof(float));
            std::atomic_thread_fence(std::memory_order_acquire);
            uint32_t v1 = v.load(std::memory_order_acquire);
            if (v0 == v1) return;
        }
    }

    // Persistence helpers
    void save_meta(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        out.write(reinterpret_cast<const char*>(&max_elements_), sizeof(max_elements_));
        out.write(reinterpret_cast<const char*>(&block_size_), sizeof(block_size_));
    }

    static VectorStore load(std::ifstream& in) {
        size_t dim, max_el, blk;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        in.read(reinterpret_cast<char*>(&max_el), sizeof(max_el));
        in.read(reinterpret_cast<char*>(&blk), sizeof(blk));
        return VectorStore(dim, max_el, blk);
    }

    // Write payload of all live slots (caller passes the slot list).
    void save_vectors(std::ofstream& out, const std::vector<Slot>& slots) const {
        size_t n = slots.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        std::vector<float> tmp(dim_);
        for (Slot s : slots) {
            copy_vector(s, tmp.data());
            out.write(reinterpret_cast<const char*>(tmp.data()), dim_ * sizeof(float));
        }
    }

    // Loads vectors into the given slot list. Caller must have ensured slots are valid.
    void load_vectors(std::ifstream& in, const std::vector<Slot>& slots) {
        size_t n;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (n != slots.size()) {
            throw std::runtime_error("vector count mismatch on load");
        }
        std::vector<float> tmp(dim_);
        for (Slot s : slots) {
            in.read(reinterpret_cast<char*>(tmp.data()), dim_ * sizeof(float));
            set_vector(s, tmp.data());
        }
    }

private:
    void ensure_block(Slot slot) {
        size_t block_idx = slot / block_size_;
        std::lock_guard<std::mutex> lk(grow_mutex_);
        while (blocks_.size() <= block_idx) {
            size_t bytes = round_up(block_size_ * dim_padded_ * sizeof(float), kVecAlign);
            blocks_.push_back(static_cast<float*>(detail::aligned_alloc_bytes(kVecAlign, bytes)));
        }
    }

    KVANN_FORCE_INLINE float* slot_ptr(Slot slot) {
        return blocks_[slot / block_size_] + (slot % block_size_) * dim_padded_;
    }
    KVANN_FORCE_INLINE const float* slot_ptr_const(Slot slot) const {
        return blocks_[slot / block_size_] + (slot % block_size_) * dim_padded_;
    }

    size_t dim_;
    size_t dim_padded_;
    size_t block_size_;
    size_t max_elements_;
    std::vector<float*> blocks_;
    mutable std::mutex grow_mutex_;
    mutable std::vector<std::atomic<uint32_t>> version_;
};

// ============================================================================
// SlotKeyMap (dense, atomic)
// ============================================================================
class SlotKeyMap {
public:
    explicit SlotKeyMap(size_t cap) : data_(cap) {
        for (auto& a : data_) a.store(kInvalidKey, std::memory_order_relaxed);
    }

    KVANN_FORCE_INLINE Key get(Slot s) const {
        return data_[s].load(std::memory_order_acquire);
    }
    void set(Slot s, Key k) {
        data_[s].store(k, std::memory_order_release);
    }
    void clear(Slot s) {
        data_[s].store(kInvalidKey, std::memory_order_release);
    }
    size_t capacity() const { return data_.size(); }

private:
    std::vector<std::atomic<Key>> data_;
};

// ============================================================================
// KeyDir (sharded)
//   Maps Key -> { slot, payload, version }.
//   Liveness is encoded in SlotKeyMap (slot_key == key  =>  live).
// ============================================================================
struct KeyEntry {
    Slot slot = kInvalidSlot;
    uint64_t version = 0;
    std::vector<uint8_t> payload;
};

class KeyDir {
public:
    explicit KeyDir(size_t stripes) : stripes_(stripes ? stripes : 1),
                                      maps_(stripes_), mutexes_(stripes_) {}

    // Read-side: returns a copy of the entry if present.
    bool find(Key k, KeyEntry& out) const {
        size_t s = stripe(k);
        std::shared_lock lk(mutexes_[s]);
        auto it = maps_[s].find(k);
        if (it == maps_[s].end()) return false;
        out = it->second;
        return true;
    }

    bool contains(Key k) const {
        size_t s = stripe(k);
        std::shared_lock lk(mutexes_[s]);
        return maps_[s].find(k) != maps_[s].end();
    }

    // Insert or update. cb is called under the stripe write lock with a mutable
    // reference to the (possibly new) entry. Returns whatever cb returns.
    template <class Fn>
    auto with_write(Key k, Fn&& cb) -> decltype(cb(std::declval<KeyEntry&>(), false)) {
        size_t s = stripe(k);
        std::unique_lock lk(mutexes_[s]);
        auto [it, inserted] = maps_[s].emplace(k, KeyEntry{});
        return cb(it->second, inserted);
    }

    bool erase(Key k) {
        size_t s = stripe(k);
        std::unique_lock lk(mutexes_[s]);
        return maps_[s].erase(k) > 0;
    }

    // Snapshot of all entries (under stripe read locks). Used by rebuild + save.
    std::vector<std::pair<Key, KeyEntry>> snapshot_all() const {
        std::vector<std::pair<Key, KeyEntry>> out;
        for (size_t i = 0; i < stripes_; ++i) {
            std::shared_lock lk(mutexes_[i]);
            for (const auto& [k, e] : maps_[i]) out.emplace_back(k, e);
        }
        return out;
    }

    size_t size() const {
        size_t total = 0;
        for (size_t i = 0; i < stripes_; ++i) {
            std::shared_lock lk(mutexes_[i]);
            total += maps_[i].size();
        }
        return total;
    }

    void clear() {
        for (size_t i = 0; i < stripes_; ++i) {
            std::unique_lock lk(mutexes_[i]);
            maps_[i].clear();
        }
    }

private:
    size_t stripe(Key k) const { return k % stripes_; }

    size_t stripes_;
    std::vector<std::unordered_map<Key, KeyEntry>> maps_;
    mutable std::vector<std::shared_mutex> mutexes_;
};

// ============================================================================
// HnswGraph
//   Layer 0  : flat arena of Slot[] + uint8_t degrees (hot, dominant cost)
//   Upper L  : sparse vector<vector<Slot>> per slot (rare)
//   Per-slot mutex pool (NUM_STRIPES) protects write-side; reads are lock-free
//   on degrees + neighbor cells (degrees use acquire/release).
// ============================================================================
class HnswGraph {
public:
    HnswGraph(size_t dim, size_t max_elements, int M, int M_max0, int ef_construction)
        : dim_(dim),
          max_elements_(max_elements),
          M_(M),
          M_max0_(M_max0),
          ef_construction_(ef_construction),
          enterpoint_(kInvalidSlot),
          max_layer_(-1),
          size_(0),
          mutex_stripes_(kStripeN),
          layer0_arena_(max_elements * static_cast<size_t>(M_max0)),
          layer0_deg_(max_elements),
          node_top_(max_elements),
          upper_(max_elements) {
        for (auto& d : layer0_deg_) d.store(0, std::memory_order_relaxed);
        for (auto& t : node_top_) t.store(-1, std::memory_order_relaxed);
        rng_.seed(0xC0FFEEULL);
        level_dist_ = std::geometric_distribution<int>(1.0 / std::log(std::max(2, M)));
    }

    HnswGraph(const HnswGraph&) = delete;
    HnswGraph& operator=(const HnswGraph&) = delete;

    HnswGraph(HnswGraph&&) noexcept = default;
    HnswGraph& operator=(HnswGraph&&) = delete;

    size_t size() const { return size_.load(std::memory_order_acquire); }
    bool empty() const  { return size() == 0; }
    int   dim() const   { return static_cast<int>(dim_); }

    // Vector source: caller-supplied function: (Slot) -> dot(slot, query)
    using DotFn = float(*)(const void* ctx, Slot s, const float* query);

    // Add a vector that is *already stored* in the vector source.
    void add(Slot slot, const float* vec, const void* ctx, DotFn dot_fn) {
        int level = sample_level();

        // Reserve slot in upper map
        bool first = false;
        {
            std::unique_lock lk(global_mutex_);
            if (slot >= max_elements_) {
                throw std::runtime_error("HNSW slot out of range");
            }
            if (size_.load(std::memory_order_relaxed) == 0) first = true;
            size_.fetch_add(1, std::memory_order_release);
            node_top_[slot].store(level, std::memory_order_release);
            if (level > 0) {
                upper_[slot].assign(level, {});
                for (auto& v : upper_[slot]) v.reserve(static_cast<size_t>(M_));
            }
            if (first) {
                enterpoint_ = slot;
                max_layer_ = level;
                return;
            }
        }

        Slot ep = enterpoint_;
        int top = max_layer_;
        // Greedy descent on upper layers
        for (int lc = top; lc > level; --lc) {
            ep = greedy_descend(ep, vec, lc, ctx, dot_fn);
        }

        for (int lc = std::min(level, top); lc >= 0; --lc) {
            auto cand = search_layer(vec, ep, ef_construction_, lc, ctx, dot_fn, nullptr);
            auto neighbors = select_neighbors_heuristic(vec, cand, M_, ctx, dot_fn);
            connect(slot, neighbors, lc, ctx, dot_fn);
            if (!neighbors.empty()) ep = neighbors[0].slot;
        }

        if (level > top) {
            std::unique_lock lk(global_mutex_);
            if (level > max_layer_) {
                max_layer_ = level;
                enterpoint_ = slot;
            }
        }
    }

    // ---- Search ----
    struct SearchHit {
        Slot  slot;
        float dist;     // 1 - cos
    };

    std::vector<SearchHit> search(const float* query, int ef, int /*topk*/,
                                  const void* ctx, DotFn dot_fn) const {
        if (empty()) return {};

        Slot ep;
        int top;
        {
            std::shared_lock lk(global_mutex_);
            ep = enterpoint_;
            top = max_layer_;
        }

        for (int lc = top; lc > 0; --lc) {
            ep = greedy_descend(ep, query, lc, ctx, dot_fn);
        }

        // Bottom layer with no slot-level filter; rerank/dedupe happens upstream.
        auto pool = search_layer(query, ep, ef, 0, ctx, dot_fn, nullptr);

        // Convert max-heap to sorted vector (ascending distance).
        std::vector<SearchHit> out;
        out.reserve(pool.size());
        while (!pool.empty()) {
            const auto& c = pool.top();
            out.push_back({c.slot, c.dist});
            pool.pop();
        }
        std::sort(out.begin(), out.end(),
                  [](const SearchHit& a, const SearchHit& b) { return a.dist < b.dist; });
        return out;
    }

    // Persist neighbors only (vectors are stored elsewhere).
    void save_meta(std::ofstream& out) const {
        std::shared_lock lk(global_mutex_);
        out.write(reinterpret_cast<const char*>(&enterpoint_), sizeof(enterpoint_));
        out.write(reinterpret_cast<const char*>(&max_layer_), sizeof(max_layer_));
        out.write(reinterpret_cast<const char*>(&max_elements_), sizeof(max_elements_));
        out.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
        out.write(reinterpret_cast<const char*>(&M_max0_), sizeof(M_max0_));
        out.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(ef_construction_));
        size_t sz = size_.load(std::memory_order_acquire);
        out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    }

    // (Persistence of full graph is M5; for now we rebuild on load.)

    void clear() {
        std::unique_lock lk(global_mutex_);
        for (auto& d : layer0_deg_) d.store(0, std::memory_order_relaxed);
        for (auto& t : node_top_) t.store(-1, std::memory_order_relaxed);
        for (auto& u : upper_) u.clear();
        enterpoint_ = kInvalidSlot;
        max_layer_ = -1;
        size_.store(0, std::memory_order_release);
    }

private:
    static constexpr size_t kStripeN = 1024;

    int sample_level() {
        std::lock_guard<std::mutex> lk(rng_mutex_);
        int lvl = level_dist_(rng_);
        if (lvl < 0) lvl = 0;
        if (lvl > 16) lvl = 16;
        return lvl;
    }

    KVANN_FORCE_INLINE std::mutex& slot_mutex(Slot s) const {
        return mutex_stripes_[s & (kStripeN - 1)];
    }

    KVANN_FORCE_INLINE Slot* layer0_neighbors(Slot s) {
        return layer0_arena_.data() + static_cast<size_t>(s) * static_cast<size_t>(M_max0_);
    }
    KVANN_FORCE_INLINE const Slot* layer0_neighbors(Slot s) const {
        return layer0_arena_.data() + static_cast<size_t>(s) * static_cast<size_t>(M_max0_);
    }

    KVANN_FORCE_INLINE int neighbors_count(Slot s, int layer) const {
        if (layer == 0) {
            return static_cast<int>(layer0_deg_[s].load(std::memory_order_acquire));
        }
        const auto& u = upper_[s];
        if (layer - 1 >= static_cast<int>(u.size())) return 0;
        return static_cast<int>(u[layer - 1].size());
    }

    // Returns a snapshot of neighbor slots at (s, layer). Single call sites copy
    // into a small local buffer to avoid holding any lock during distance calls.
    std::vector<Slot> snapshot_neighbors(Slot s, int layer) const {
        if (layer == 0) {
            uint8_t deg = layer0_deg_[s].load(std::memory_order_acquire);
            std::vector<Slot> out(deg);
            const Slot* src = layer0_neighbors(s);
            for (int i = 0; i < deg; ++i) {
                out[i] = src[i];
            }
            return out;
        }
        // Upper layers: take per-slot lock briefly to copy.
        std::lock_guard<std::mutex> lk(slot_mutex(s));
        const auto& u = upper_[s];
        if (layer - 1 >= static_cast<int>(u.size())) return {};
        return u[layer - 1];
    }

    Slot greedy_descend(Slot ep, const float* query, int layer,
                        const void* ctx, DotFn dot_fn) const {
        Slot curr = ep;
        float curr_d = 1.0f - dot_fn(ctx, curr, query);
        bool changed = true;
        while (changed) {
            changed = false;
            auto nbrs = snapshot_neighbors(curr, layer);
            for (Slot n : nbrs) {
                float d = 1.0f - dot_fn(ctx, n, query);
                if (d < curr_d) {
                    curr_d = d;
                    curr = n;
                    changed = true;
                }
            }
        }
        return curr;
    }

    struct Cand {
        Slot  slot;
        float dist;     // smaller = closer
    };
    struct CmpFar  { bool operator()(const Cand& a, const Cand& b) const { return a.dist < b.dist; } };  // max-heap on dist
    struct CmpNear { bool operator()(const Cand& a, const Cand& b) const { return a.dist > b.dist; } };  // min-heap on dist

    using SlotFilter = bool (*)(const void* ctx, Slot s);

    // Returns the upper-bound max-heap of best ef candidates.
    std::priority_queue<Cand, std::vector<Cand>, CmpFar>
    search_layer(const float* query, Slot ep, int ef, int layer,
                 const void* ctx, DotFn dot_fn, SlotFilter sfilter) const {
        // Per-thread visited buffer (epoch-tagged) — zero-cost reuse.
        auto& vp = visited_pool();
        if (vp.tags.size() < max_elements_) {
            vp.tags.assign(max_elements_, 0);
            vp.epoch = 0;
        }
        uint32_t epoch = ++vp.epoch;
        if (KVANN_UNLIKELY(epoch == 0)) {
            std::fill(vp.tags.begin(), vp.tags.end(), 0);
            epoch = ++vp.epoch;
        }
        auto try_visit = [&](Slot s) -> bool {
            if (vp.tags[s] == epoch) return false;
            vp.tags[s] = epoch;
            return true;
        };

        std::priority_queue<Cand, std::vector<Cand>, CmpNear> frontier;
        std::priority_queue<Cand, std::vector<Cand>, CmpFar>  best;

        float ed = 1.0f - dot_fn(ctx, ep, query);
        frontier.push({ep, ed});
        best.push({ep, ed});
        try_visit(ep);

        while (!frontier.empty()) {
            Cand c = frontier.top();
            if (!best.empty() && c.dist > best.top().dist) break;
            frontier.pop();

            auto nbrs = snapshot_neighbors(c.slot, layer);
            for (Slot n : nbrs) {
                if (!try_visit(n)) continue;
                float d = 1.0f - dot_fn(ctx, n, query);
                if (sfilter && !sfilter(ctx, n)) {
                    // visited but not eligible — still don't traverse from it
                    continue;
                }
                if (static_cast<int>(best.size()) < ef || d < best.top().dist) {
                    frontier.push({n, d});
                    best.push({n, d});
                    if (static_cast<int>(best.size()) > ef) best.pop();
                }
            }
        }
        return best;
    }

    // Neighbor selection — pick M closest. (Pairwise diversity heuristic
    // requires slot->vec ptr access via DotFn which currently only takes a
    // query buffer; revisit when DotFn is generalized.)
    std::vector<Cand> select_neighbors_heuristic(
            const float* /*query*/,
            std::priority_queue<Cand, std::vector<Cand>, CmpFar> cand,
            int M, const void* /*ctx*/, DotFn /*dot_fn*/) const {
        std::vector<Cand> sorted;
        sorted.reserve(cand.size());
        while (!cand.empty()) { sorted.push_back(cand.top()); cand.pop(); }
        std::sort(sorted.begin(), sorted.end(),
                  [](const Cand& a, const Cand& b) { return a.dist < b.dist; });
        if (static_cast<int>(sorted.size()) > M) sorted.resize(static_cast<size_t>(M));
        return sorted;
    }

    void connect(Slot src, const std::vector<Cand>& neighbors, int layer,
                 const void* ctx, DotFn dot_fn) {
        // Add src->n on this layer.
        for (const Cand& nc : neighbors) {
            insert_neighbor(src, nc.slot, layer, ctx, dot_fn);
            insert_neighbor(nc.slot, src, layer, ctx, dot_fn);
        }
    }

    void insert_neighbor(Slot s, Slot n, int layer,
                         const void* /*ctx*/, DotFn /*dot_fn*/) {
        if (s == n) return;
        std::lock_guard<std::mutex> lk(slot_mutex(s));
        if (layer == 0) {
            uint8_t deg = layer0_deg_[s].load(std::memory_order_relaxed);
            Slot* arr = layer0_neighbors(s);
            for (int i = 0; i < deg; ++i) if (arr[i] == n) return;
            if (deg < M_max0_) {
                arr[deg] = n;
                layer0_deg_[s].store(static_cast<uint8_t>(deg + 1),
                                     std::memory_order_release);
            } else {
                // Cap reached — replace the most-recent slot (FIFO-ish).
                // True heuristic eviction is a follow-up (needs slot->vec access).
                arr[deg - 1] = n;
            }
        } else {
            int li = layer - 1;
            auto& u = upper_[s];
            int top = node_top_[s].load(std::memory_order_acquire);
            if (top < layer) return;
            if (li >= static_cast<int>(u.size())) {
                u.resize(static_cast<size_t>(layer));
            }
            auto& vec = u[li];
            for (Slot existing : vec) if (existing == n) return;
            if (static_cast<int>(vec.size()) < M_) {
                vec.push_back(n);
            } else {
                vec.back() = n;
            }
        }
    }

    struct VisitedTLS {
        std::vector<uint32_t> tags;
        uint32_t epoch = 0;
    };
    static VisitedTLS& visited_pool() {
        thread_local VisitedTLS v;
        return v;
    }

    size_t dim_;
    size_t max_elements_;
    int M_;
    int M_max0_;
    int ef_construction_;

    Slot enterpoint_;
    int  max_layer_;
    std::atomic<size_t> size_;

    mutable std::shared_mutex global_mutex_;
    mutable std::vector<std::mutex> mutex_stripes_;

    std::vector<Slot> layer0_arena_;          // size = max_elements * M_max0_
    std::vector<std::atomic<uint8_t>> layer0_deg_;
    std::vector<std::atomic<int8_t>> node_top_;
    std::vector<std::vector<std::vector<Slot>>> upper_;  // upper_[slot][layer-1]

    std::mt19937_64 rng_;
    std::geometric_distribution<int> level_dist_;
    std::mutex rng_mutex_;
};

// ============================================================================
// DeltaSet
// ============================================================================
class DeltaSet {
public:
    explicit DeltaSet(size_t cap) : alive_(cap) {
        for (auto& a : alive_) a.store(0, std::memory_order_relaxed);
    }

    KVANN_FORCE_INLINE bool is_alive(Slot s) const {
        return alive_[s].load(std::memory_order_acquire) != 0;
    }

    void mark_alive(Slot s) {
        std::lock_guard lk(list_mutex_);
        if (alive_[s].load(std::memory_order_relaxed) == 0) {
            alive_[s].store(1, std::memory_order_release);
            members_.insert(s);
        }
    }

    void mark_dead(Slot s) {
        std::lock_guard lk(list_mutex_);
        if (alive_[s].load(std::memory_order_relaxed) != 0) {
            alive_[s].store(0, std::memory_order_release);
            members_.erase(s);
        }
    }

    size_t size() const {
        std::lock_guard lk(list_mutex_);
        return members_.size();
    }

    std::vector<Slot> snapshot() const {
        std::lock_guard lk(list_mutex_);
        return std::vector<Slot>(members_.begin(), members_.end());
    }

    void clear() {
        std::lock_guard lk(list_mutex_);
        for (Slot s : members_) alive_[s].store(0, std::memory_order_release);
        members_.clear();
    }

private:
    mutable std::mutex list_mutex_;
    std::vector<std::atomic<uint8_t>> alive_;
    std::unordered_set<Slot> members_;
};

} // anonymous namespace

// ============================================================================
// Index::Impl
// ============================================================================
struct Index::Impl {
    Impl(IndexConfig cfg)
        : config_(std::move(cfg)),
          storage_(config_.dim, config_.max_elements, config_.storage_block_size),
          slot_key_(config_.max_elements),
          key_dir_(config_.lock_stripes),
          base_graph_(std::make_unique<HnswGraph>(
              config_.dim, config_.max_elements,
              config_.hnsw_M, config_.hnsw_M_max0, config_.hnsw_ef_construction)),
          delta_graph_(std::make_unique<HnswGraph>(
              config_.dim, config_.max_elements,
              config_.hnsw_M, config_.hnsw_M_max0, config_.hnsw_ef_construction)),
          delta_(config_.max_elements),
          next_slot_(0),
          rebuild_running_(false) {
        if (config_.dim == 0) {
            throw std::invalid_argument("kvann: IndexConfig::dim must be > 0");
        }
        if (config_.max_elements == 0) {
            throw std::invalid_argument("kvann: IndexConfig::max_elements must be > 0");
        }
        if (config_.lock_stripes == 0) config_.lock_stripes = 1;
    }

    ~Impl() {
        if (rebuild_thread_.joinable()) rebuild_thread_.join();
    }

    // ------- ctx adapter for HnswGraph -------
    static float dot_via_storage(const void* ctx, Slot s, const float* query) {
        const Impl* self = static_cast<const Impl*>(ctx);
        return self->storage_.dot_with(s, query);
    }

    // ------- single-key ops -------
    Status put(Key key, const float* vector, const void* payload, size_t payload_len) {
        if (vector == nullptr) return Status::InvalidArgument("vector is null");

        // Normalize into a stack/heap buffer.
        std::vector<float> norm(config_.dim);
        std::memcpy(norm.data(), vector, config_.dim * sizeof(float));
        simd::normalize_f32(norm.data(), config_.dim);

        Slot slot = kInvalidSlot;
        bool is_new = false;

        Status st = key_dir_.with_write(key, [&](KeyEntry& e, bool inserted) -> Status {
            if (inserted) {
                Slot s = next_slot_.fetch_add(1, std::memory_order_acq_rel);
                if (s >= static_cast<Slot>(config_.max_elements)) {
                    // Slot is permanently leaked; concurrent fetch_add prevents
                    // a safe rollback. Acceptable: callers must respect cap.
                    return Status::Full();
                }
                e.slot = s;
                e.version = 1;
                is_new = true;
            } else {
                e.version++;
            }
            if (payload && payload_len > 0) {
                const auto* p = static_cast<const uint8_t*>(payload);
                e.payload.assign(p, p + payload_len);
            } else if (payload == nullptr) {
                // unspecified -> keep existing payload on update; clear on insert
                if (inserted) e.payload.clear();
            } else {
                e.payload.clear();
            }
            slot = e.slot;
            return Status::Ok();
        });
        if (!st.ok()) return st;

        storage_.set_vector(slot, norm.data());
        slot_key_.set(slot, key);

        // Add to delta graph (always until rebuild moves it to base).
        delta_.mark_alive(slot);

        if (is_new) live_count_.fetch_add(1, std::memory_order_acq_rel);

        maybe_extend_delta_hnsw(slot, norm.data());
        return Status::Ok();
    }

    void maybe_extend_delta_hnsw(Slot just_added, const float* vec) {
        if (delta_hnsw_active_.load(std::memory_order_acquire)) {
            delta_graph_->add(just_added, vec, this, &Impl::dot_via_storage);
            return;
        }
        if (delta_.size() <= config_.delta_hnsw_threshold) return;

        std::lock_guard<std::mutex> lk(delta_hnsw_build_mutex_);
        if (delta_hnsw_active_.load(std::memory_order_acquire)) {
            delta_graph_->add(just_added, vec, this, &Impl::dot_via_storage);
            return;
        }
        // Promotion: backfill ALL current delta members into the HNSW.
        auto members = delta_.snapshot();
        std::vector<float> tmp(config_.dim);
        for (Slot m : members) {
            const float* src = (m == just_added) ? vec : nullptr;
            if (src) {
                delta_graph_->add(m, src, this, &Impl::dot_via_storage);
            } else {
                storage_.copy_vector(m, tmp.data());
                delta_graph_->add(m, tmp.data(), this, &Impl::dot_via_storage);
            }
        }
        delta_hnsw_active_.store(true, std::memory_order_release);
    }

    Status del(Key key) {
        KeyEntry e;
        if (!key_dir_.find(key, e)) return Status::NotFound();
        Slot slot = e.slot;
        if (!key_dir_.erase(key)) return Status::NotFound();
        slot_key_.clear(slot);
        delta_.mark_dead(slot);
        live_count_.fetch_sub(1, std::memory_order_acq_rel);
        return Status::Ok();
    }

    bool exists(Key key) const { return key_dir_.contains(key); }

    Status get_payload(Key key, std::vector<uint8_t>& out) const {
        KeyEntry e;
        if (!key_dir_.find(key, e)) return Status::NotFound();
        out = std::move(e.payload);
        return Status::Ok();
    }

    // ------- search -------
    std::vector<SearchResult> search(const float* query, const SearchParams& params) const {
        if (query == nullptr) return {};
        int topk = std::max(1, params.topk);
        int ef = params.ef > 0 ? params.ef : config_.hnsw_ef_search;

        std::vector<float> q(config_.dim);
        std::memcpy(q.data(), query, config_.dim * sizeof(float));
        simd::normalize_f32(q.data(), config_.dim);

        // Collect candidates from base + delta.
        // Hold shared lock while we touch base_graph_ — rebuild swaps the
        // unique_ptr under unique_lock at the end of do_rebuild().
        std::vector<HnswGraph::SearchHit> base_hits;
        {
            std::shared_lock lk(base_swap_mutex_);
            if (!base_graph_->empty()) {
                base_hits = base_graph_->search(q.data(), ef, topk * 2,
                                                this, &Impl::dot_via_storage);
            }
        }

        std::vector<HnswGraph::SearchHit> delta_hits;
        size_t dsz = delta_.size();
        if (dsz > 0) {
            if (delta_hnsw_active_.load(std::memory_order_acquire)
                && !delta_graph_->empty()) {
                delta_hits = delta_graph_->search(q.data(), ef, topk * 2,
                                                  this, &Impl::dot_via_storage);
            } else {
                // Brute-force over delta members.
                auto members = delta_.snapshot();
                delta_hits.reserve(members.size());
                for (Slot s : members) {
                    float d = 1.0f - storage_.dot_with(s, q.data());
                    delta_hits.push_back({s, d});
                }
            }
        }

        // Merge + dedupe + alive-check + rerank with exact cosine on storage.
        // alive-check uses slot_key_ (kInvalidKey == dead).
        std::vector<SearchResult> out;
        out.reserve(static_cast<size_t>(topk));
        std::vector<std::pair<Slot, float>> merged;
        merged.reserve(base_hits.size() + delta_hits.size());
        for (const auto& h : base_hits)  merged.emplace_back(h.slot, h.dist);
        for (const auto& h : delta_hits) merged.emplace_back(h.slot, h.dist);

        // Dedupe by slot (keep the smaller dist).
        std::sort(merged.begin(), merged.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first || (a.first == b.first && a.second < b.second);
                  });
        merged.erase(std::unique(merged.begin(), merged.end(),
                                 [](const auto& a, const auto& b) { return a.first == b.first; }),
                     merged.end());

        // Re-score with exact cosine + apply user filter + alive check.
        std::vector<SearchResult> scored;
        scored.reserve(merged.size());
        for (auto& [slot, _] : merged) {
            Key k = slot_key_.get(slot);
            if (k == kInvalidKey) continue;
            if (params.filter && !params.filter(k)) continue;
            float sim = storage_.dot_with(slot, q.data());
            scored.emplace_back(k, sim);
        }

        // top-k by score desc.
        if (static_cast<int>(scored.size()) > topk) {
            std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                              [](const SearchResult& a, const SearchResult& b) {
                                  return a.score > b.score;
                              });
            scored.resize(static_cast<size_t>(topk));
        } else {
            std::sort(scored.begin(), scored.end(),
                      [](const SearchResult& a, const SearchResult& b) {
                          return a.score > b.score;
                      });
        }

        if (params.include_payload) {
            for (auto& r : scored) {
                KeyEntry e;
                if (key_dir_.find(r.key, e)) r.payload = std::move(e.payload);
            }
        }

        return scored;
    }

    // ------- batch -------
    Status put_batch(const Key* keys, const float* vectors, size_t n, size_t* first_err) {
        Status agg = Status::Ok();
        for (size_t i = 0; i < n; ++i) {
            Status s = put(keys[i], vectors + i * config_.dim, nullptr, 0);
            if (!s.ok() && agg.ok()) {
                agg = s;
                if (first_err) *first_err = i;
            }
        }
        return agg;
    }

    std::vector<std::vector<SearchResult>>
    search_batch(const float* queries, size_t n, const SearchParams& params) const {
        std::vector<std::vector<SearchResult>> out(n);
        for (size_t i = 0; i < n; ++i) {
            out[i] = search(queries + i * config_.dim, params);
        }
        return out;
    }

    // ------- rebuild -------
    Status rebuild_async() {
        bool expected = false;
        if (!rebuild_running_.compare_exchange_strong(expected, true)) {
            return Status::AlreadyExists("rebuild already running");
        }
        if (rebuild_thread_.joinable()) rebuild_thread_.join();
        rebuild_thread_ = std::thread([this]() { do_rebuild(); });
        return Status::Ok();
    }

    Status rebuild() {
        Status s = rebuild_async();
        if (!s.ok() && s.code() != StatusCode::kAlreadyExists) return s;
        wait_rebuild();
        return Status::Ok();
    }

    void wait_rebuild() const {
        if (rebuild_thread_.joinable()) {
            const_cast<std::thread&>(rebuild_thread_).join();
        }
    }

    void do_rebuild() {
        emit_log(config_, "info", "rebuild starting");
        KVANN_LOG_INFO("rebuild starting");

        // 1. Snapshot live (key, slot, vector) under stripe read locks.
        struct Snap { Key key; Slot slot; };
        std::vector<Snap> snap;
        std::vector<float> snap_vecs;
        {
            auto entries = key_dir_.snapshot_all();
            snap.reserve(entries.size());
            snap_vecs.resize(entries.size() * config_.dim);
            size_t i = 0;
            for (auto& [k, e] : entries) {
                snap.push_back({k, e.slot});
                storage_.copy_vector(e.slot, snap_vecs.data() + i * config_.dim);
                ++i;
            }
        }

        // 2. Build new base HNSW from snapshot. Vectors are already in storage_,
        //    so dot_fn = storage_.dot_with. Concurrent updates to those slots are
        //    protected by the seqlock (search will retry on torn reads).
        auto new_base = std::make_unique<HnswGraph>(
            config_.dim, config_.max_elements,
            config_.hnsw_M, config_.hnsw_M_max0, config_.hnsw_ef_construction);
        for (size_t i = 0; i < snap.size(); ++i) {
            new_base->add(snap[i].slot, snap_vecs.data() + i * config_.dim,
                          this, &Impl::dot_via_storage);
        }

        // 3. Atomic swap.
        {
            std::unique_lock lk(base_swap_mutex_);
            base_graph_ = std::move(new_base);
        }

        // 4. Drain delta of entries that are now in base.
        std::unordered_set<Slot> base_slots;
        base_slots.reserve(snap.size());
        for (auto& s : snap) base_slots.insert(s.slot);
        auto members = delta_.snapshot();
        for (Slot m : members) {
            if (base_slots.count(m)) delta_.mark_dead(m);
        }
        {
            std::lock_guard<std::mutex> lk(delta_hnsw_build_mutex_);
            delta_graph_->clear();
            delta_hnsw_active_.store(false, std::memory_order_release);
        }

        rebuild_running_.store(false, std::memory_order_release);
        emit_log(config_, "info", "rebuild done");
        KVANN_LOG_INFO("rebuild done");
    }

    // ------- stats -------
    IndexStats stats() const {
        IndexStats s;
        s.dim = config_.dim;
        size_t total = static_cast<size_t>(next_slot_.load(std::memory_order_acquire));
        size_t alive = live_count_.load(std::memory_order_acquire);
        s.total_keys = total;
        s.live_keys = alive;
        s.tombstone_count = total > alive ? total - alive : 0;
        s.tombstone_ratio = total > 0 ? static_cast<float>(s.tombstone_count) / total : 0.0f;
        {
            std::shared_lock lk(base_swap_mutex_);
            s.base_count = base_graph_->size();
        }
        s.delta_count = delta_.size();
        s.delta_ratio = alive > 0 ? static_cast<float>(s.delta_count) / alive : 0.0f;
        s.simd_backend = simd_backend();
        return s;
    }

    const IndexConfig& config() const { return config_; }

    // ------- persistence -------
    Status save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) return Status::Io("cannot open: " + path);

        const char magic[8] = {'K','V','A','N','N','0','2','\0'};
        uint32_t fmt_version = 2;
        uint32_t reserved = 0;
        out.write(magic, sizeof(magic));
        out.write(reinterpret_cast<const char*>(&fmt_version), sizeof(fmt_version));
        out.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));

        // Config
        out.write(reinterpret_cast<const char*>(&config_.dim), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&config_.max_elements), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&config_.storage_block_size), sizeof(size_t));

        // Snapshot live keys + payloads + vectors (slot order).
        auto entries = key_dir_.snapshot_all();
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.second.slot < b.second.slot; });

        size_t n = entries.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));

        std::vector<Slot> slots;
        slots.reserve(n);
        for (auto& [k, e] : entries) {
            out.write(reinterpret_cast<const char*>(&k), sizeof(k));
            out.write(reinterpret_cast<const char*>(&e.slot), sizeof(e.slot));
            out.write(reinterpret_cast<const char*>(&e.version), sizeof(e.version));
            size_t pl = e.payload.size();
            out.write(reinterpret_cast<const char*>(&pl), sizeof(pl));
            if (pl > 0) {
                out.write(reinterpret_cast<const char*>(e.payload.data()), pl);
            }
            slots.push_back(e.slot);
        }

        storage_.save_vectors(out, slots);
        out.flush();
        if (!out) return Status::Io("write failed");
        return Status::Ok();
    }

    static std::unique_ptr<Index> load(const std::string& path);

    void load_from_stream(std::ifstream& in) {
        size_t n;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));

        std::vector<Slot> slots;
        slots.reserve(n);
        Slot max_slot = 0;

        for (size_t i = 0; i < n; ++i) {
            Key k;
            KeyEntry e;
            in.read(reinterpret_cast<char*>(&k), sizeof(k));
            in.read(reinterpret_cast<char*>(&e.slot), sizeof(e.slot));
            in.read(reinterpret_cast<char*>(&e.version), sizeof(e.version));
            size_t pl;
            in.read(reinterpret_cast<char*>(&pl), sizeof(pl));
            if (pl > 0) {
                e.payload.resize(pl);
                in.read(reinterpret_cast<char*>(e.payload.data()), pl);
            }
            Slot slot = e.slot;
            slot_key_.set(slot, k);
            if (slot + 1 > max_slot) max_slot = slot + 1;
            key_dir_.with_write(k, [&](KeyEntry& slot_entry, bool /*inserted*/) {
                slot_entry = std::move(e);
            });
            slots.push_back(slot);
        }
        next_slot_.store(max_slot, std::memory_order_release);
        live_count_.store(slots.size(), std::memory_order_release);

        storage_.load_vectors(in, slots);

        // Rebuild base HNSW from the freshly loaded KV.
        rebuild();
    }

    IndexConfig config_;
    VectorStore storage_;
    SlotKeyMap  slot_key_;
    KeyDir      key_dir_;
    std::unique_ptr<HnswGraph> base_graph_;
    std::unique_ptr<HnswGraph> delta_graph_;
    DeltaSet    delta_;
    std::atomic<size_t> live_count_{0};
    std::atomic<bool>   delta_hnsw_active_{false};
    std::mutex          delta_hnsw_build_mutex_;

    std::atomic<Slot> next_slot_;
    std::atomic<bool> rebuild_running_;
    std::thread       rebuild_thread_;
    mutable std::shared_mutex base_swap_mutex_;
};

// ============================================================================
// Index — public API forwarding
// ============================================================================
Index::Index(const IndexConfig& config) : impl_(std::make_unique<Impl>(config)) {}
Index::~Index() = default;
Index::Index(Index&&) noexcept = default;
Index& Index::operator=(Index&&) noexcept = default;

Status Index::put(Key key, const float* vector) {
    return impl_->put(key, vector, nullptr, 0);
}
Status Index::put(Key key, const float* vector, const void* payload, size_t payload_len) {
    return impl_->put(key, vector, payload, payload_len);
}
Status Index::del(Key key)            { return impl_->del(key); }
bool   Index::exists(Key key) const   { return impl_->exists(key); }
Status Index::get_payload(Key key, std::vector<uint8_t>& out) const {
    return impl_->get_payload(key, out);
}

Status Index::put_batch(const Key* keys, const float* vectors, size_t n, size_t* first_err) {
    return impl_->put_batch(keys, vectors, n, first_err);
}

std::vector<SearchResult> Index::search(const float* query, const SearchParams& params) const {
    return impl_->search(query, params);
}

std::vector<std::vector<SearchResult>>
Index::search_batch(const float* queries, size_t n, const SearchParams& params) const {
    return impl_->search_batch(queries, n, params);
}

Status Index::rebuild()                 { return impl_->rebuild(); }
Status Index::rebuild_async()           { return impl_->rebuild_async(); }
void   Index::wait_rebuild() const      { impl_->wait_rebuild(); }

IndexStats         Index::stats() const  { return impl_->stats(); }
const IndexConfig& Index::config() const { return impl_->config(); }

Status Index::save(const std::string& path) const { return impl_->save(path); }

std::unique_ptr<Index> Index::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open: " + path);

    char magic[8] = {};
    in.read(magic, sizeof(magic));
    static const char expected[8] = {'K','V','A','N','N','0','2','\0'};
    if (std::memcmp(magic, expected, sizeof(magic)) != 0) {
        throw std::runtime_error("kvann: unsupported file format (need v2)");
    }
    uint32_t fmt_version = 0;
    uint32_t reserved = 0;
    in.read(reinterpret_cast<char*>(&fmt_version), sizeof(fmt_version));
    in.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
    if (fmt_version != 2) {
        throw std::runtime_error("kvann: unsupported format version");
    }

    IndexConfig cfg;
    in.read(reinterpret_cast<char*>(&cfg.dim), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cfg.max_elements), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cfg.storage_block_size), sizeof(size_t));

    auto idx = std::make_unique<Index>(cfg);
    idx->impl_->load_from_stream(in);
    return idx;
}

} // namespace kvann
