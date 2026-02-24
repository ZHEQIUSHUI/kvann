/**
 * kvann 使用示例
 * 展示基本功能 + user_data 功能
 */

#include <kvann/index.h>
#include <kvann/index_v3.h>
#include <kvann/distributed.h>
#include <iostream>
#include <random>
#include <vector>

// 生成随机归一化向量
std::vector<float> random_vector(size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> vec(dim);
    for (auto& v : vec) v = dist(rng);
    kvann::normalize_vector(vec.data(), dim);
    return vec;
}

void example_basic() {
    std::cout << "\n=== 示例1: 基础功能 ===" << std::endl;
    
    const size_t DIM = 128;
    const int N = 100;
    
    kvann::Index index(DIM, N * 2);
    std::mt19937 rng(42);
    
    // 插入向量
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        index.put(i, vec.data());
    }
    
    // 重建
    index.rebuild();
    index.wait_rebuild();
    
    // 搜索
    auto results = index.search(vectors[0].data(), 5);
    
    std::cout << "Top-5 搜索结果:" << std::endl;
    for (const auto& r : results) {
        std::cout << "  key=" << r.key << " score=" << r.score << std::endl;
    }
}

void example_user_data() {
    std::cout << "\n=== 示例2: 带 user_data 的向量 ===" << std::endl;
    
    const size_t DIM = 128;
    const int N = 50;
    
    kvann::Index index(DIM, N * 2);
    std::mt19937 rng(42);
    
    // 插入带 user_data 的向量（例如：文本、标签、元数据等）
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        
        // user_data 可以是任意二进制数据
        std::string metadata = "document_" + std::to_string(i) + ":这是一个示例文档";
        index.put_with_data(i, vec.data(), metadata.c_str(), metadata.size() + 1);
    }
    
    index.rebuild();
    index.wait_rebuild();
    
    // 搜索并获取 user_data
    auto query = random_vector(DIM, rng);
    auto results = index.search(query.data(), 5);
    
    std::cout << "搜索结果（含user_data）:" << std::endl;
    for (const auto& r : results) {
        std::string user_data_str(reinterpret_cast<const char*>(r.user_data.data()));
        std::cout << "  key=" << r.key << " score=" << r.score 
                  << " data=" << user_data_str << std::endl;
    }
}

void example_v3_tiered() {
    std::cout << "\n=== 示例3: V3 多层索引 ===" << std::endl;
    
    const size_t DIM = 128;
    const int N = 500;
    
    // 配置分层策略
    kvann::TieringPolicy tiering;
    tiering.hot_threshold = 100;    // Delta层最多100个
    tiering.warm_threshold = 300;   // Base层最多300个
    tiering.enable_gpu = false;
    tiering.enable_pq = false;
    
    kvann::IndexV3 index(DIM, N * 2, kvann::QuantizeType::FP32, 
                         kvann::AutoRebuildPolicy(), tiering);
    
    std::mt19937 rng(42);
    
    // 插入数据
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        index.put(i, vec.data());
    }
    
    // 分层重建
    index.rebuild_with_tiering();
    
    // 查看分层统计
    auto tier_stats = index.tier_stats();
    std::cout << "分层统计:" << std::endl;
    std::cout << "  Delta (热数据): " << tier_stats.delta_count << std::endl;
    std::cout << "  Base  (温数据): " << tier_stats.base_count << std::endl;
    std::cout << "  Cold  (冷数据): " << tier_stats.cold_count << std::endl;
}

void example_distributed() {
    std::cout << "\n=== 示例4: 分布式索引 ===" << std::endl;
    
    const size_t DIM = 128;
    const int N = 200;
    const int NSHARDS = 4;
    
    // 创建4个分片的分布式索引
    kvann::DistributedIndex dist_index(DIM, NSHARDS);
    std::mt19937 rng(42);
    
    // 插入数据（自动按key哈希分片）
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        dist_index.put(i, vec.data());
    }
    
    // 重建所有分片
    dist_index.rebuild_all();
    dist_index.wait_all_rebuilds();
    
    // 查看分片统计
    auto stats = dist_index.stats();
    std::cout << "分布式统计:" << std::endl;
    std::cout << "  总分片数: " << dist_index.nshards() << std::endl;
    std::cout << "  总向量数: " << stats.total_live << std::endl;
    for (size_t i = 0; i < stats.shard_stats.size(); ++i) {
        std::cout << "  分片[" << i << "]: " << stats.shard_stats[i].live_vectors << " 向量" << std::endl;
    }
    
    // 搜索（fan-out到所有分片，合并结果）
    auto query = random_vector(DIM, rng);
    auto results = dist_index.search(query.data(), 5);
    
    std::cout << "分布式搜索结果:" << std::endl;
    for (const auto& r : results) {
        std::cout << "  key=" << r.key << " score=" << r.score << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "kvann 向量检索引擎 - 使用示例" << std::endl;
    std::cout << "========================================" << std::endl;
    
    example_basic();
    example_user_data();
    example_v3_tiered();
    example_distributed();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "所有示例完成！" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
