/**
 * kvann - 主索引类
 * 单一版本（KV-first, ANN-second）
 */

#pragma once

#include "core.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kvann {

struct IndexConfig {
    size_t max_elements = 1000000;
    size_t delta_bruteforce_limit = 1000;
    size_t delta_hnsw_threshold = 5000;

    int hnsw_M = 16;
    int hnsw_ef_construction = 200;
    int hnsw_ef_search = 64;

    size_t storage_block_size = 4096;
    size_t lock_stripes = 64;
};

class Index {
public:
    /**
     * 构造函数
     * @param dim: 向量维度
     * @param max_elements: 最大元素数
     * @param delta_threshold: Delta层使用brute-force的最大数量，超过则触发rebuild建议
     */
    Index(size_t dim, size_t max_elements = 1000000, size_t delta_threshold = 1000);
    Index(size_t dim, const IndexConfig& config);
    ~Index();

    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;
    Index(Index&&) noexcept;
    Index& operator=(Index&&) noexcept;

    // 插入/更新向量
    bool put(Key key, const float* vector);

    // 插入/更新向量（带user_data）
    bool put_with_data(Key key, const float* vector, const void* user_data, size_t user_data_len);

    // 删除向量（tombstone机制）
    bool del(Key key);

    // 检查key是否存在（且未删除）
    bool exists(Key key) const;

    // 获取key对应的user_data
    std::vector<uint8_t> get_user_data(Key key) const;

    // 搜索（多线程安全）
    std::vector<SearchResult> search(const float* query, int topk);

    // 手动重建索引
    void rebuild();

    // 等待重建完成
    void wait_rebuild();

    // 获取统计信息
    IndexStats stats() const;

    // 保存索引到文件
    void save(const std::string& path) const;

    // 从文件加载索引
    static std::unique_ptr<Index> load(const std::string& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kvann
