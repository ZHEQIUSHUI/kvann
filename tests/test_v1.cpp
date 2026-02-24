/**
 * kvann 测试
 * 测试内容：
 * 1. 基本CRUD操作
 * 2. 搜索功能
 * 3. Delta层正确性
 * 4. Tombstone删除
 * 5. 重建索引
 * 6. 持久化
 * 7. 多线程安全
 */

#include <kvann/index.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <thread>
#include <chrono>

using namespace kvann;

// ============================================================================
// 测试工具
// ============================================================================

class TestTimer {
public:
    TestTimer(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
    
    ~TestTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << "  [Timer] " << name_ << ": " << ms << "ms" << std::endl;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "ASSERTION FAILED: " << msg << " at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

#define RUN_TEST(name) \
    do { \
        std::cout << "\n[TEST] " << #name << "..." << std::endl; \
        name(); \
        std::cout << "[PASS] " << #name << std::endl; \
    } while(0)

// 生成随机向量（归一化）
std::vector<float> random_vector(size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> vec(dim);
    for (auto& v : vec) v = dist(rng);
    normalize_vector(vec.data(), dim);
    return vec;
}

// 计算余弦相似度
float compute_sim(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;
}

// ============================================================================
// 测试用例
// ============================================================================

/**
 * 测试1: 基本CRUD
 */
void test_basic_crud() {
    const size_t DIM = 128;
    Index index(DIM, 10000);
    
    // Put
    std::vector<float> vec1 = {1, 0, 0, 0};  // 简化为4维测试
    vec1.resize(DIM, 0);
    normalize_vector(vec1.data(), DIM);
    
    TEST_ASSERT(index.put(1, vec1.data()), "Put failed");
    TEST_ASSERT(index.exists(1), "Key should exist");
    
    // Update
    std::vector<float> vec2 = {0, 1, 0, 0};
    vec2.resize(DIM, 0);
    normalize_vector(vec2.data(), DIM);
    
    TEST_ASSERT(index.put(1, vec2.data()), "Update failed");
    
    // Delete
    TEST_ASSERT(index.del(1), "Delete failed");
    TEST_ASSERT(!index.exists(1), "Key should not exist after delete");
    TEST_ASSERT(!index.del(1), "Delete tombstoned key should return false");
    
    // Delete non-existent
    TEST_ASSERT(!index.del(999), "Delete non-existent should return false");
    
    // Put after delete
    TEST_ASSERT(index.put(1, vec1.data()), "Put after delete failed");
    TEST_ASSERT(index.exists(1), "Key should exist after re-put");
}

/**
 * 测试2: 基本搜索
 */
void test_basic_search() {
    const size_t DIM = 128;
    const int N = 100;
    Index index(DIM, 10000);
    std::mt19937 rng(42);
    
    // 插入向量
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        TEST_ASSERT(index.put(i, vec.data()), "Put failed");
    }
    
    // 重建索引使base层生效
    index.rebuild();
    index.wait_rebuild();
    
    // 搜索
    auto query = vectors[0];
    auto results = index.search(query.data(), 10);
    
    TEST_ASSERT(!results.empty(), "Search should return results");
    TEST_ASSERT(results[0].key == 0, "First result should be the query itself");
    TEST_ASSERT(results[0].score > 0.99f, "Self similarity should be ~1.0");
    
    // 验证分数正确性
    for (const auto& r : results) {
        float expected = compute_sim(query, vectors[r.key]);
        TEST_ASSERT(std::abs(r.score - expected) < 0.001f, "Score mismatch");
    }
}

/**
 * 测试3: Delta层（未重建时的搜索）
 */
void test_delta_layer() {
    const size_t DIM = 128;
    const int N = 50;
    Index index(DIM, 10000);
    std::mt19937 rng(42);
    
    // 插入向量（不重建）
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        TEST_ASSERT(index.put(i, vec.data()), "Put failed");
    }
    
    // 不重建直接搜索（走delta层）
    auto query = vectors[10];
    auto results = index.search(query.data(), 5);
    
    TEST_ASSERT(!results.empty(), "Search should work with delta layer");
    
    // 检查所有结果都在插入的向量中
    for (const auto& r : results) {
        TEST_ASSERT(r.key < static_cast<Key>(N), "Result key out of range");
    }
}

/**
 * 测试4: Tombstone删除语义
 */
void test_tombstone() {
    const size_t DIM = 128;
    Index index(DIM, 10000);
    std::mt19937 rng(42);
    
    // 插入向量
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < 20; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        index.put(i, vec.data());
    }
    
    index.rebuild();
    index.wait_rebuild();
    
    // 删除一些向量
    index.del(5);
    index.del(10);
    index.del(15);
    
    // 搜索不应返回已删除的向量
    auto query = vectors[5];
    auto results = index.search(query.data(), 20);
    
    for (const auto& r : results) {
        TEST_ASSERT(r.key != 5, "Deleted key 5 should not appear");
        TEST_ASSERT(r.key != 10, "Deleted key 10 should not appear");
        TEST_ASSERT(r.key != 15, "Deleted key 15 should not appear");
    }
    
    // 重新插入被删除的key
    auto new_vec = random_vector(DIM, rng);
    index.put(5, new_vec.data());
    TEST_ASSERT(index.exists(5), "Re-inserted key should exist");
}

/**
 * 测试5: 更新语义
 */
void test_update() {
    const size_t DIM = 128;
    Index index(DIM, 10000);
    std::mt19937 rng(42);
    
    auto vec1 = random_vector(DIM, rng);
    auto vec2 = random_vector(DIM, rng);
    
    // 第一次插入
    index.put(1, vec1.data());
    index.rebuild();
    index.wait_rebuild();
    
    // 更新
    index.put(1, vec2.data());
    
    // 搜索应该找到新的向量
    auto results = index.search(vec2.data(), 1);
    TEST_ASSERT(results[0].key == 1, "Should find updated vector");
    
    float expected = compute_sim(vec2, vec2);
    TEST_ASSERT(std::abs(results[0].score - expected) < 0.001f, "Score should be ~1.0");
}

/**
 * 测试6: 重建索引
 */
void test_rebuild() {
    const size_t DIM = 128;
    const int N = 1000;
    Index index(DIM, 10000);
    std::mt19937 rng(42);
    
    // 插入大量向量
    {
        TestTimer timer("Insert " + std::to_string(N) + " vectors");
        for (int i = 0; i < N; ++i) {
            auto vec = random_vector(DIM, rng);
            index.put(i, vec.data());
        }
    }
    
    // 重建前：delta层有大量数据
    auto stats_before = index.stats();
    TEST_ASSERT(stats_before.delta_count == (size_t)N, "All vectors should be in delta");
    
    // 重建
    {
        TestTimer timer("Rebuild");
        index.rebuild();
        index.wait_rebuild();
    }
    
    // 重建后：delta层清空，base层有数据
    auto stats_after = index.stats();
    TEST_ASSERT(stats_after.base_count == (size_t)N, "All vectors should be in base");
    TEST_ASSERT(stats_after.delta_count == 0, "Delta should be empty");
}

/**
 * 测试7: 持久化
 */
void test_persistence() {
    const size_t DIM = 128;
    const int N = 100;
    const char* TEST_FILE = "/tmp/kvann_test.index";
    std::mt19937 rng(42);
    
    std::vector<std::vector<float>> vectors;
    
    // 创建并保存索引
    {
        Index index(DIM, 10000);
        
        for (int i = 0; i < N; ++i) {
            auto vec = random_vector(DIM, rng);
            vectors.push_back(vec);
            index.put(i, vec.data());
        }
        
        index.rebuild();
        index.wait_rebuild();
        
        index.save(TEST_FILE);
    }
    
    // 加载索引
    {
        auto index = Index::load(TEST_FILE);
        
        // 验证数据
        for (int i = 0; i < N; ++i) {
            TEST_ASSERT(index->exists(i), "Key should exist after load");
        }
        
        // 验证搜索
        auto query = vectors[0];
        auto results = index->search(query.data(), 5);
        TEST_ASSERT(!results.empty(), "Search should work after load");
        
        // 验证删除
        index->del(50);
        TEST_ASSERT(!index->exists(50), "Delete should work after load");
    }
    
    // 清理
    std::remove(TEST_FILE);
}

/**
 * 测试8: 多线程查询
 */
void test_concurrent_search() {
    const size_t DIM = 128;
    const int N = 1000;
    const int NUM_THREADS = 4;
    const int QUERIES_PER_THREAD = 100;
    
    Index index(DIM, 10000);
    std::mt19937 rng(42);
    
    // 插入数据
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        index.put(i, vec.data());
    }
    
    index.rebuild();
    index.wait_rebuild();
    
    // 并发查询
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    {
        TestTimer timer("Concurrent search (" + std::to_string(NUM_THREADS) + " threads)");
        
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&index, &vectors, &success_count, t, QUERIES_PER_THREAD]() {
                std::mt19937 local_rng(42 + t);
                std::uniform_int_distribution<int> dist(0, vectors.size() - 1);
                
                for (int i = 0; i < QUERIES_PER_THREAD; ++i) {
                    int idx = dist(local_rng);
                    auto results = index.search(vectors[idx].data(), 10);
                    if (!results.empty() && results[0].score > 0.9f) {
                        success_count++;
                    }
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
    }
    
    TEST_ASSERT(success_count == NUM_THREADS * QUERIES_PER_THREAD, 
                "All concurrent queries should succeed");
}

/**
 * 测试9: 多线程读写
 */
void test_concurrent_readwrite() {
    const size_t DIM = 128;
    const int N = 500;
    const int NUM_WRITERS = 2;
    const int NUM_READERS = 4;
    
    Index index(DIM, 10000);
    std::atomic<bool> stop{false};
    std::atomic<int> write_count{0};
    std::atomic<int> read_count{0};
    
    std::vector<std::thread> threads;
    
    // 写入线程
    for (int t = 0; t < NUM_WRITERS; ++t) {
        threads.emplace_back([&index, &write_count, &stop, t, N]() {
            std::mt19937 rng(42 + t);
            int i = t;
            while (i < N && !stop.load()) {
                auto vec = random_vector(128, rng);
                index.put(i, vec.data());
                write_count++;
                i += NUM_WRITERS;
            }
        });
    }
    
    // 读取线程
    for (int t = 0; t < NUM_READERS; ++t) {
        threads.emplace_back([&index, &read_count, &stop, t]() {
            std::mt19937 rng(100 + t);
            std::vector<float> query(128);
            std::normal_distribution<float> dist(0, 1);
            
            while (!stop.load()) {
                for (auto& v : query) v = dist(rng);
                normalize_vector(query.data(), 128);
                
                auto results = index.search(query.data(), 5);
                read_count++;
                
                if (read_count > 1000) break;  // 限制查询次数
            }
        });
    }
    
    {
        TestTimer timer("Concurrent read/write");
        for (auto& t : threads) {
            t.join();
        }
    }
    
    TEST_ASSERT(write_count > 0, "Should have writes");
    TEST_ASSERT(read_count > 0, "Should have reads");
    
    std::cout << "  Writes: " << write_count << ", Reads: " << read_count << std::endl;
}

/**
 * 测试10: Recall 测试（验证ANN质量）
 */
void test_recall() {
    const size_t DIM = 128;
    const int N = 1000;
    const int K = 10;
    Index index(DIM, 10000);
    std::mt19937 rng(42);
    
    // 生成数据
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        index.put(i, vec.data());
    }
    
    index.rebuild();
    index.wait_rebuild();
    
    // 测试recall
    int total_hits = 0;
    int num_queries = 50;
    
    for (int q = 0; q < num_queries; ++q) {
        auto query = vectors[q];
        
        // ANN搜索
        auto ann_results = index.search(query.data(), K);
        
        // 暴力搜索（ground truth）
        std::vector<std::pair<int, float>> all_sims;
        for (int i = 0; i < N; ++i) {
            float sim = compute_sim(query, vectors[i]);
            all_sims.emplace_back(i, sim);
        }
        std::partial_sort(all_sims.begin(), all_sims.begin() + K, all_sims.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // 计算hits
        std::unordered_set<int> ann_keys;
        for (const auto& r : ann_results) {
            ann_keys.insert(r.key);
        }
        
        for (int i = 0; i < K; ++i) {
            if (ann_keys.count(all_sims[i].first)) {
                total_hits++;
            }
        }
    }
    
    float recall = (float)total_hits / (num_queries * K);
    std::cout << "  Recall@" << K << " = " << recall << std::endl;
    
    // 宽松要求：recall > 0.7（V1版本）
    TEST_ASSERT(recall > 0.7f, "Recall too low");
}

/**
 * 测试11: 空索引行为
 */
void test_empty_index() {
    const size_t DIM = 128;
    Index index(DIM, 10000);
    
    // 搜索空索引
    std::vector<float> query(DIM);
    auto results = index.search(query.data(), 10);
    TEST_ASSERT(results.empty(), "Search on empty index should return empty");
    
    // 不存在的key
    TEST_ASSERT(!index.exists(999), "Non-existent key");
    
    auto stats = index.stats();
    TEST_ASSERT(stats.total_vectors == 0, "Empty index stats");
}

/**
 * 测试12: 大规模数据
 */
void test_large_scale() {
    const size_t DIM = 128;
    const int N = 10000;
    Index index(DIM, 20000);
    std::mt19937 rng(42);
    
    {
        TestTimer timer("Insert " + std::to_string(N) + " vectors");
        for (int i = 0; i < N; ++i) {
            auto vec = random_vector(DIM, rng);
            index.put(i, vec.data());
        }
    }
    
    {
        TestTimer timer("Rebuild " + std::to_string(N) + " vectors");
        index.rebuild();
        index.wait_rebuild();
    }
    
    // 搜索性能测试
    auto query = random_vector(DIM, rng);
    
    {
        TestTimer timer("Search 100 times");
        for (int i = 0; i < 100; ++i) {
            auto results = index.search(query.data(), 10);
        }
    }
    
    auto stats = index.stats();
    TEST_ASSERT(stats.base_count == (size_t)N, "All vectors in base");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "kvann Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        RUN_TEST(test_basic_crud);
        RUN_TEST(test_basic_search);
        RUN_TEST(test_delta_layer);
        RUN_TEST(test_tombstone);
        RUN_TEST(test_update);
        RUN_TEST(test_rebuild);
        RUN_TEST(test_persistence);
        RUN_TEST(test_concurrent_search);
        RUN_TEST(test_concurrent_readwrite);
        RUN_TEST(test_recall);
        RUN_TEST(test_empty_index);
        RUN_TEST(test_large_scale);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED WITH EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
