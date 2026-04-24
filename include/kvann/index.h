// kvann - 主索引接口
//
// 设计原则：KV-first, ANN-second。Key 是外部唯一 ID，Slot 是内部稠密编号。
// HNSW 索引仅作为加速结构；删除/更新立即生效，搜索结果总是按精确余弦再 rerank。
#pragma once

#include "core.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace kvann {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct IndexConfig {
    // ---- mandatory ----
    size_t dim          = 0;            // 向量维度（必填）
    size_t max_elements = 1'000'000;    // 容量上限

    // ---- HNSW ----
    int hnsw_M               = 16;
    int hnsw_M_max0          = 32;       // layer-0 最大度数（推荐 ~2*M）
    int hnsw_ef_construction = 200;
    int hnsw_ef_search       = 64;       // 默认 ef，可被 SearchParams 覆盖

    // ---- delta layer ----
    size_t delta_bruteforce_limit = 1'000;   // <= 这个走 brute-force
    size_t delta_hnsw_threshold   = 5'000;   // > 这个自动建 delta HNSW

    // ---- storage ----
    size_t storage_block_size = 4096;        // 每块包含的 slot 数

    // ---- concurrency ----
    size_t lock_stripes = 64;                // KeyDir 分桶数（必须 >0）

    // ---- automatic rebuild thresholds (informational) ----
    float auto_rebuild_tombstone_ratio = 0.30f;
    float auto_rebuild_delta_ratio     = 0.50f;

    // Parallel rebuild: 0 = serial, 1 = auto (hardware_concurrency),
    // >1 = exact thread count. Serial build gives best recall; parallel
    // is much faster but recall may drop ~3-5% due to insertion-order
    // interleaving. Default = auto.
    int rebuild_threads = 1;

    // ---- logger ----
    // 不为空时所有内部日志（rebuild 等）都走此回调。默认 no-op。
    // level: "info" | "warn" | "error"
    std::function<void(const char* level, const char* msg)> log_sink;
};

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
struct SearchParams {
    int  topk            = 10;
    int  ef              = 0;       // 0 = 用 IndexConfig::hnsw_ef_search
    bool include_payload = false;   // 结果是否携带 user_data
    // 可选自定义过滤：返回 false 则该 key 不进入结果
    std::function<bool(Key)> filter;
};

struct SearchResult {
    Key   key   = 0;
    float score = 0;                      // 余弦相似度（越大越相似）
    std::vector<uint8_t> payload;         // include_payload=true 时填充

    SearchResult() = default;
    SearchResult(Key k, float s) : key(k), score(s) {}
};

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
struct IndexStats {
    size_t dim              = 0;
    size_t total_keys       = 0;   // 包括 tombstone
    size_t live_keys        = 0;
    size_t tombstone_count  = 0;
    size_t base_count       = 0;
    size_t delta_count      = 0;
    float  tombstone_ratio  = 0;
    float  delta_ratio      = 0;
    const char* simd_backend = "scalar";
};

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------
class Index {
public:
    // 构造
    explicit Index(const IndexConfig& config);
    ~Index();

    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;
    Index(Index&&) noexcept;
    Index& operator=(Index&&) noexcept;

    // ---- single-key ops ----
    Status put(Key key, const float* vector);
    Status put(Key key, const float* vector, const void* payload, size_t payload_len);
    Status del(Key key);
    bool   exists(Key key) const;
    Status get_payload(Key key, std::vector<uint8_t>& out) const;

    // ---- batch ----
    // vectors: row-major, n * dim floats. 任一项失败不影响其他项；
    // 返回的 Status 反映"是否全部成功"，详细错误可看返回的 first_error 索引。
    Status put_batch(const Key* keys, const float* vectors, size_t n,
                     size_t* first_error_index = nullptr);

    // ---- search ----
    std::vector<SearchResult> search(const float* query,
                                     const SearchParams& params = {}) const;

    // queries: n * dim floats; 返回 n 个结果列表
    std::vector<std::vector<SearchResult>>
    search_batch(const float* queries, size_t n,
                 const SearchParams& params = {}) const;

    // ---- maintenance ----
    Status rebuild();          // 同步：触发 + 等待
    Status rebuild_async();    // 后台触发，立即返回
    void   wait_rebuild() const;

    // ---- introspection ----
    IndexStats stats() const;
    const IndexConfig& config() const;

    // ---- persistence ----
    // 写入失败抛 std::runtime_error；后续 M5 升级到 Status。
    Status save(const std::string& path) const;

    // 加载失败抛 std::runtime_error。
    static std::unique_ptr<Index> load(const std::string& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kvann
