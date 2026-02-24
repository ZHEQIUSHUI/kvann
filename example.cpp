/**
 * kvann 使用示例
 * 展示基本功能 + user_data 功能
 */

#include <kvann/index.h>
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

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "kvann 向量检索引擎 - 使用示例" << std::endl;
    std::cout << "========================================" << std::endl;
    
    example_basic();
    example_user_data();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "所有示例完成！" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
