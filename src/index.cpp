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
#include <kvann/detail/crc32.h>
#include <kvann/detail/log.h>
#include <kvann/detail/simd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
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
          version_(max_elements),
          slot_mutexes_(kSlotMutexN) {
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
        // Serialize concurrent writers to the same slot so the seqlock window
        // (odd version + memcpy + even version) is atomic per writer. Without
        // this, two put()s targeting the same key would have overlapping
        // memcpys and a torn version sequence.
        std::lock_guard<std::mutex> lk(slot_mutexes_[slot & (kSlotMutexN - 1)]);
        auto& v = version_[slot];
        v.fetch_add(1, std::memory_order_acq_rel);  // -> odd (writing)
        float* dst = slot_ptr(slot);
        std::memcpy(dst, src, dim_ * sizeof(float));
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

    static constexpr size_t kSlotMutexN = 1024;

    size_t dim_;
    size_t dim_padded_;
    size_t block_size_;
    size_t max_elements_;
    std::vector<float*> blocks_;
    mutable std::mutex grow_mutex_;
    mutable std::vector<std::atomic<uint32_t>> version_;
    mutable std::vector<std::mutex> slot_mutexes_;
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
        Slot ep;
        int top;
        {
            // Re-check empty() inside the lock so a concurrent clear()/swap
            // between the user's empty() check and our read of enterpoint_
            // can't hand us an invalid slot.
            std::shared_lock lk(global_mutex_);
            if (size_.load(std::memory_order_relaxed) == 0 ||
                enterpoint_ == kInvalidSlot) {
                return {};
            }
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

    // ---- Persistence ----
    // Full graph state -> stream. Vectors are stored elsewhere.
    // Format:
    //   u32 enterpoint, i32 max_layer, u64 size
    //   u64 n_nodes  (slots with top >= 0)
    //   for each node:
    //     u32 slot
    //     i8  top_layer
    //     u8  layer0_deg
    //     Slot[layer0_deg] layer0_neighbors
    //     for layer in 1..top:
    //       u8 deg
    //       Slot[deg] neighbors
    void save_graph(std::vector<uint8_t>& out) const {
        std::shared_lock lk(global_mutex_);
        auto put = [&](const void* p, std::size_t n) {
            const uint8_t* b = static_cast<const uint8_t*>(p);
            out.insert(out.end(), b, b + n);
        };

        uint32_t ep = enterpoint_;
        int32_t  ml = max_layer_;
        uint64_t sz = size_.load(std::memory_order_acquire);
        put(&ep, sizeof(ep));
        put(&ml, sizeof(ml));
        put(&sz, sizeof(sz));

        // Count nodes and remember which to dump.
        uint64_t n_nodes = 0;
        for (size_t i = 0; i < node_top_.size(); ++i) {
            if (node_top_[i].load(std::memory_order_acquire) >= 0) ++n_nodes;
        }
        put(&n_nodes, sizeof(n_nodes));

        for (size_t i = 0; i < node_top_.size(); ++i) {
            int8_t top = node_top_[i].load(std::memory_order_acquire);
            if (top < 0) continue;

            uint32_t s = static_cast<uint32_t>(i);
            put(&s, sizeof(s));
            put(&top, sizeof(top));

            uint8_t deg0 = layer0_deg_[i].load(std::memory_order_acquire);
            put(&deg0, sizeof(deg0));
            const Slot* nbrs = layer0_neighbors(static_cast<Slot>(i));
            put(nbrs, sizeof(Slot) * deg0);

            const auto& upper = upper_[i];
            for (int L = 1; L <= top; ++L) {
                int li = L - 1;
                uint8_t deg = (li < static_cast<int>(upper.size()))
                              ? static_cast<uint8_t>(upper[li].size()) : 0;
                put(&deg, sizeof(deg));
                if (deg > 0) put(upper[li].data(), sizeof(Slot) * deg);
            }
        }
    }

    void load_graph(const uint8_t* data, std::size_t len) {
        std::unique_lock lk(global_mutex_);
        std::size_t off = 0;
        auto take = [&](void* dst, std::size_t n) {
            if (off + n > len) throw std::runtime_error("hnsw graph: short read");
            std::memcpy(dst, data + off, n);
            off += n;
        };

        // Reset
        for (auto& d : layer0_deg_) d.store(0, std::memory_order_relaxed);
        for (auto& t : node_top_) t.store(-1, std::memory_order_relaxed);
        for (auto& u : upper_) u.clear();

        uint32_t ep;
        int32_t ml;
        uint64_t sz, n_nodes;
        take(&ep, sizeof(ep));
        take(&ml, sizeof(ml));
        take(&sz, sizeof(sz));
        take(&n_nodes, sizeof(n_nodes));
        enterpoint_ = ep;
        max_layer_  = ml;
        size_.store(sz, std::memory_order_release);

        for (uint64_t i = 0; i < n_nodes; ++i) {
            uint32_t s;
            int8_t top;
            take(&s, sizeof(s));
            take(&top, sizeof(top));
            if (s >= max_elements_) throw std::runtime_error("hnsw: slot out of range");
            node_top_[s].store(top, std::memory_order_release);

            uint8_t deg0;
            take(&deg0, sizeof(deg0));
            if (deg0 > M_max0_) throw std::runtime_error("hnsw: deg0 out of range");
            Slot* nbrs = layer0_neighbors(static_cast<Slot>(s));
            take(nbrs, sizeof(Slot) * deg0);
            layer0_deg_[s].store(deg0, std::memory_order_release);

            if (top > 0) {
                upper_[s].resize(static_cast<std::size_t>(top));
                for (int L = 1; L <= top; ++L) {
                    int li = L - 1;
                    uint8_t deg;
                    take(&deg, sizeof(deg));
                    upper_[s][li].resize(deg);
                    if (deg > 0) take(upper_[s][li].data(), sizeof(Slot) * deg);
                }
            }
        }
        if (off != len) throw std::runtime_error("hnsw: trailing bytes");
    }

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
          next_slot_(0) {
        if (config_.dim == 0) {
            throw std::invalid_argument("kvann: IndexConfig::dim must be > 0");
        }
        if (config_.max_elements == 0) {
            throw std::invalid_argument("kvann: IndexConfig::max_elements must be > 0");
        }
        if (config_.lock_stripes == 0) config_.lock_stripes = 1;
    }

    ~Impl() {
        // Wait for any in-flight rebuild before destruction so the detached
        // worker thread doesn't outlive `this`.
        wait_rebuild();
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
    //
    // We use a shared_future so multiple threads can safely wait for the same
    // rebuild without racing on std::thread::join (which is UB if called from
    // multiple threads). Worker is detached and signals completion via a
    // promise; the destructor waits for any in-flight rebuild.
    Status rebuild_async() {
        std::lock_guard<std::mutex> lk(rebuild_mu_);
        if (rebuild_future_.valid() &&
            rebuild_future_.wait_for(std::chrono::seconds(0))
                != std::future_status::ready) {
            return Status::AlreadyExists("rebuild already running");
        }
        auto pr = std::make_shared<std::promise<void>>();
        rebuild_future_ = pr->get_future().share();
        std::thread([this, pr]() {
            try { do_rebuild(); } catch (...) {}
            pr->set_value();
        }).detach();
        return Status::Ok();
    }

    Status rebuild() {
        Status s = rebuild_async();
        if (!s.ok() && s.code() != StatusCode::kAlreadyExists) return s;
        wait_rebuild();
        return Status::Ok();
    }

    void wait_rebuild() const {
        std::shared_future<void> f;
        {
            std::lock_guard<std::mutex> lk(rebuild_mu_);
            f = rebuild_future_;
        }
        if (f.valid()) f.wait();
    }

    void do_rebuild() {
        emit_log(config_, "info", "rebuild starting");
        KVANN_LOG_INFO("rebuild starting");
        // No need for rebuild_running_ flag; rebuild_future_ ready state is
        // the source of truth (checked under rebuild_mu_).

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

    // ------- persistence (file format v3) -------
    //
    // Layout:
    //   [0..32)        Header (magic, fmt_version, flags, num_sections)
    //   [32..288)      Section table: 8 entries x 32 bytes (zero-padded)
    //   [288..)        Sections in order: meta, keys, vectors, [hnsw_graph]
    //
    // Section IDs:
    //   1 META, 2 KEYS, 3 VECTORS, 4 HNSW_GRAPH (optional)
    //
    // Each section is checksummed with CRC32 (IEEE 802.3, reflected).
    Status save(const std::string& path) const {
        constexpr uint32_t kFmtVersion   = 3;
        constexpr uint32_t kSecMeta      = 1;
        constexpr uint32_t kSecKeys      = 2;
        constexpr uint32_t kSecVectors   = 3;
        constexpr uint32_t kSecHnswGraph = 4;
        constexpr size_t   kHeaderSize   = 32;
        constexpr size_t   kMaxSections  = 8;
        constexpr size_t   kTableSize    = kMaxSections * 32;
        constexpr size_t   kPrefix       = kHeaderSize + kTableSize;

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) return Status::Io("cannot open: " + path);

        // Reserve prefix.
        std::vector<char> zeros(kPrefix, 0);
        out.write(zeros.data(), zeros.size());

        struct SecEntry {
            uint32_t id; uint32_t pad0;
            uint64_t offset; uint64_t length;
            uint32_t crc; uint32_t pad1;
        };
        std::vector<SecEntry> sections;

        auto write_section = [&](uint32_t id, const std::vector<uint8_t>& bytes) {
            SecEntry e{};
            e.id     = id;
            e.offset = static_cast<uint64_t>(out.tellp());
            e.length = bytes.size();
            e.crc    = detail::crc32(bytes.data(), bytes.size());
            out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            sections.push_back(e);
        };

        // ---- META section ----
        {
            std::vector<uint8_t> buf;
            auto put = [&](const void* p, size_t n) {
                const uint8_t* b = static_cast<const uint8_t*>(p);
                buf.insert(buf.end(), b, b + n);
            };
            uint64_t dim = config_.dim;
            uint64_t maxe = config_.max_elements;
            uint64_t blk = config_.storage_block_size;
            uint64_t ns = next_slot_.load(std::memory_order_acquire);
            int32_t M = config_.hnsw_M, M0 = config_.hnsw_M_max0,
                    efc = config_.hnsw_ef_construction;
            put(&dim, 8); put(&maxe, 8); put(&blk, 8); put(&ns, 8);
            put(&M, 4); put(&M0, 4); put(&efc, 4);
            uint32_t pad = 0; put(&pad, 4);
            write_section(kSecMeta, buf);
        }

        // Snapshot live keys, sorted by slot.
        auto entries = key_dir_.snapshot_all();
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.second.slot < b.second.slot; });

        // ---- KEYS section ----
        {
            std::vector<uint8_t> buf;
            auto put = [&](const void* p, size_t n) {
                const uint8_t* b = static_cast<const uint8_t*>(p);
                buf.insert(buf.end(), b, b + n);
            };
            uint64_t n = entries.size();
            put(&n, 8);
            for (auto& [k, e] : entries) {
                put(&k, 8);
                put(&e.slot, 4);
                uint64_t ver = e.version;
                put(&ver, 8);
                uint64_t pl = e.payload.size();
                put(&pl, 8);
                if (pl > 0) put(e.payload.data(), pl);
            }
            write_section(kSecKeys, buf);
        }

        // ---- VECTORS section (streamed, large) ----
        {
            SecEntry e{};
            e.id = kSecVectors;
            e.offset = static_cast<uint64_t>(out.tellp());
            uint32_t crc = 0;

            uint64_t dim = config_.dim;
            uint64_t n = entries.size();
            out.write(reinterpret_cast<const char*>(&dim), 8);
            crc = detail::crc32_update(crc, &dim, 8);
            out.write(reinterpret_cast<const char*>(&n), 8);
            crc = detail::crc32_update(crc, &n, 8);

            std::vector<float> tmp(config_.dim);
            for (auto& [k, en] : entries) {
                storage_.copy_vector(en.slot, tmp.data());
                size_t b = config_.dim * sizeof(float);
                out.write(reinterpret_cast<const char*>(tmp.data()), b);
                crc = detail::crc32_update(crc, tmp.data(), b);
            }
            e.length = static_cast<uint64_t>(out.tellp()) - e.offset;
            e.crc = crc;
            sections.push_back(e);
        }

        // ---- HNSW_GRAPH section (only if base is non-empty) ----
        bool has_hnsw = false;
        {
            std::shared_lock lk(base_swap_mutex_);
            if (!base_graph_->empty()) {
                has_hnsw = true;
                std::vector<uint8_t> buf;
                base_graph_->save_graph(buf);
                write_section(kSecHnswGraph, buf);
            }
        }

        // Seek back, write header + section table.
        out.seekp(0, std::ios::beg);
        char magic[8] = {'K','V','A','N','N','0','3','\0'};
        uint32_t flags = has_hnsw ? 1u : 0u;
        uint32_t num_sections = static_cast<uint32_t>(sections.size());
        uint32_t reserved = 0;
        uint64_t reserved2 = 0;
        out.write(magic, 8);
        out.write(reinterpret_cast<const char*>(&kFmtVersion), 4);
        out.write(reinterpret_cast<const char*>(&flags), 4);
        out.write(reinterpret_cast<const char*>(&num_sections), 4);
        out.write(reinterpret_cast<const char*>(&reserved), 4);
        out.write(reinterpret_cast<const char*>(&reserved2), 8);

        for (const auto& s : sections) {
            out.write(reinterpret_cast<const char*>(&s), sizeof(SecEntry));
        }
        // pad table to fixed size
        for (size_t i = sections.size(); i < kMaxSections; ++i) {
            char zero[32] = {};
            out.write(zero, 32);
        }

        out.flush();
        if (!out) return Status::Io("write failed");
        return Status::Ok();
    }

    static std::unique_ptr<Index> load(const std::string& path);

    // Loads sections from an open input stream positioned at byte 0.
    // Returns false on any error (status set by caller).
    void load_from_stream(std::ifstream& in) {
        constexpr size_t kMaxSections = 8;

        char magic[8] = {};
        in.read(magic, 8);
        static const char expected[8] = {'K','V','A','N','N','0','3','\0'};
        if (std::memcmp(magic, expected, 8) != 0) {
            throw std::runtime_error("kvann: bad magic (need v3)");
        }
        uint32_t fmt_version = 0, flags = 0, num_sections = 0, reserved = 0;
        uint64_t reserved2 = 0;
        in.read(reinterpret_cast<char*>(&fmt_version), 4);
        in.read(reinterpret_cast<char*>(&flags), 4);
        in.read(reinterpret_cast<char*>(&num_sections), 4);
        in.read(reinterpret_cast<char*>(&reserved), 4);
        in.read(reinterpret_cast<char*>(&reserved2), 8);
        if (fmt_version != 3) {
            throw std::runtime_error("kvann: unsupported fmt_version");
        }

        struct SecEntry {
            uint32_t id; uint32_t pad0;
            uint64_t offset; uint64_t length;
            uint32_t crc; uint32_t pad1;
        };
        std::vector<SecEntry> table(kMaxSections);
        in.read(reinterpret_cast<char*>(table.data()), kMaxSections * 32);
        table.resize(num_sections);

        auto find_section = [&](uint32_t id) -> const SecEntry* {
            for (auto& e : table) if (e.id == id) return &e;
            return nullptr;
        };

        auto read_section = [&](const SecEntry& e) -> std::vector<uint8_t> {
            std::vector<uint8_t> buf(e.length);
            in.seekg(static_cast<std::streamoff>(e.offset), std::ios::beg);
            in.read(reinterpret_cast<char*>(buf.data()), e.length);
            uint32_t got = detail::crc32(buf.data(), buf.size());
            if (got != e.crc) {
                throw std::runtime_error("kvann: section CRC mismatch (id=" +
                                         std::to_string(e.id) + ")");
            }
            return buf;
        };

        // KEYS
        const SecEntry* sk = find_section(2);
        if (!sk) throw std::runtime_error("kvann: missing KEYS section");
        auto kbuf = read_section(*sk);

        // Parse keys
        size_t off = 0;
        auto take = [&](void* dst, size_t n) {
            if (off + n > kbuf.size()) throw std::runtime_error("KEYS short");
            std::memcpy(dst, kbuf.data() + off, n);
            off += n;
        };
        uint64_t n_keys = 0;
        take(&n_keys, 8);
        std::vector<Slot> slots;
        slots.reserve(n_keys);
        Slot max_slot = 0;
        for (uint64_t i = 0; i < n_keys; ++i) {
            Key k = 0;
            Slot slot = 0;
            uint64_t ver = 0, pl = 0;
            take(&k, 8); take(&slot, 4); take(&ver, 8); take(&pl, 8);
            KeyEntry e;
            e.slot = slot;
            e.version = ver;
            if (pl > 0) {
                e.payload.resize(pl);
                take(e.payload.data(), pl);
            }
            slot_key_.set(slot, k);
            if (slot + 1 > max_slot) max_slot = slot + 1;
            key_dir_.with_write(k, [&](KeyEntry& slot_entry, bool) {
                slot_entry = std::move(e);
            });
            slots.push_back(slot);
        }
        next_slot_.store(max_slot, std::memory_order_release);
        live_count_.store(slots.size(), std::memory_order_release);

        // VECTORS
        const SecEntry* sv = find_section(3);
        if (!sv) throw std::runtime_error("kvann: missing VECTORS section");
        auto vbuf = read_section(*sv);
        size_t voff = 0;
        auto vtake = [&](void* dst, size_t n) {
            if (voff + n > vbuf.size()) throw std::runtime_error("VECTORS short");
            std::memcpy(dst, vbuf.data() + voff, n);
            voff += n;
        };
        uint64_t vdim = 0, vn = 0;
        vtake(&vdim, 8); vtake(&vn, 8);
        if (vdim != config_.dim || vn != slots.size()) {
            throw std::runtime_error("kvann: vectors header mismatch");
        }
        for (uint64_t i = 0; i < vn; ++i) {
            const float* src = reinterpret_cast<const float*>(vbuf.data() + voff);
            voff += config_.dim * sizeof(float);
            storage_.set_vector(slots[i], src);
        }

        // HNSW_GRAPH (optional)
        if (flags & 1u) {
            const SecEntry* sh = find_section(4);
            if (sh) {
                auto hbuf = read_section(*sh);
                base_graph_->load_graph(hbuf.data(), hbuf.size());
            }
        }

        // If no HNSW graph in file, rebuild from KV.
        if ((flags & 1u) == 0) {
            rebuild();
        }
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
    mutable std::mutex          rebuild_mu_;
    mutable std::shared_future<void> rebuild_future_;
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

    // Read header magic to verify before constructing.
    char magic[8] = {};
    in.read(magic, 8);
    static const char expected[8] = {'K','V','A','N','N','0','3','\0'};
    if (std::memcmp(magic, expected, 8) != 0) {
        throw std::runtime_error("kvann: bad magic (need v3)");
    }

    // Skip rest of header to reach section table, then read META.
    uint32_t fmt_version = 0, flags = 0, num_sections = 0, reserved = 0;
    uint64_t reserved2 = 0;
    in.read(reinterpret_cast<char*>(&fmt_version), 4);
    in.read(reinterpret_cast<char*>(&flags), 4);
    in.read(reinterpret_cast<char*>(&num_sections), 4);
    in.read(reinterpret_cast<char*>(&reserved), 4);
    in.read(reinterpret_cast<char*>(&reserved2), 8);
    if (fmt_version != 3) {
        throw std::runtime_error("kvann: unsupported fmt_version");
    }

    struct SecEntry {
        uint32_t id; uint32_t pad0;
        uint64_t offset; uint64_t length;
        uint32_t crc; uint32_t pad1;
    };
    std::vector<SecEntry> table(8);
    in.read(reinterpret_cast<char*>(table.data()), 8 * 32);
    table.resize(num_sections);

    // Find META, parse minimal fields needed for IndexConfig.
    const SecEntry* sm = nullptr;
    for (auto& e : table) if (e.id == 1) { sm = &e; break; }
    if (!sm) throw std::runtime_error("kvann: missing META section");

    std::vector<uint8_t> mbuf(sm->length);
    in.seekg(static_cast<std::streamoff>(sm->offset), std::ios::beg);
    in.read(reinterpret_cast<char*>(mbuf.data()), sm->length);

    if (detail::crc32(mbuf.data(), mbuf.size()) != sm->crc) {
        throw std::runtime_error("kvann: META CRC mismatch");
    }

    IndexConfig cfg;
    size_t off = 0;
    auto take = [&](void* dst, size_t n) {
        std::memcpy(dst, mbuf.data() + off, n);
        off += n;
    };
    uint64_t dim, maxe, blk, ns;
    int32_t M, M0, efc;
    take(&dim, 8); take(&maxe, 8); take(&blk, 8); take(&ns, 8);
    take(&M, 4); take(&M0, 4); take(&efc, 4);
    cfg.dim                  = static_cast<size_t>(dim);
    cfg.max_elements         = static_cast<size_t>(maxe);
    cfg.storage_block_size   = static_cast<size_t>(blk);
    cfg.hnsw_M               = M;
    cfg.hnsw_M_max0          = M0;
    cfg.hnsw_ef_construction = efc;

    auto idx = std::make_unique<Index>(cfg);
    in.clear();
    in.seekg(0, std::ios::beg);
    idx->impl_->load_from_stream(in);
    return idx;
}

} // namespace kvann
