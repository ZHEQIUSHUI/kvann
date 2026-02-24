/**
 * kvann V3 测试
 * 测试内容：
 * 1. GPU 接口（CPU 回退模式）
 * 2. Product Quantization (PQ)
 * 3. 多层索引（Delta + Base + Cold）
 * 4. 分布式支持
 */

#include <kvann/index_v3.h>
#include <kvann/distributed.h>
#include <kvann/gpu.h>
#include <kvann/pq.h>
#include <iostream>
#include <random>

using namespace kvann;

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

std::vector<float> random_vector(size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> vec(dim);
    for (auto& v : vec) v = dist(rng);
    normalize_vector(vec.data(), dim);
    return vec;
}

// ============================================================================
// 测试用例
// ============================================================================

/**
 * 测试1: GPU 可用性检查
 */
void test_gpu_availability() {
    bool available = gpu_available();
    auto info = get_gpu_info();
    
    std::cout << "  GPU available: " << (available ? "yes" : "no") << std::endl;
    std::cout << "  Device count: " << info.device_count << std::endl;
    
    // 在 CPU 模式下应该返回 false
    TEST_ASSERT(!available || info.device_count > 0, "GPU info inconsistent");
}

/**
 * 测试2: GPU Brute-force 索引（CPU 回退）
 */
void test_gpu_brute_force() {
    const size_t DIM = 128;
    const int N = 100;
    
    GPUBruteForceIndex gpu_index(DIM, N * 2);
    std::mt19937 rng(42);
    
    // 添加向量
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        gpu_index.add(i, vec.data());
    }
    
    // 搜索
    auto query = vectors[0];
    auto results = gpu_index.search(query.data(), 10);
    
    TEST_ASSERT(!results.empty(), "GPU index should return results");
    
    // 第一个应该是查询向量本身
    bool found_self = false;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i] == 0) {
            found_self = true;
            break;
        }
    }
    TEST_ASSERT(found_self, "Should find self in GPU index");
    
    std::cout << "  GPU Brute-force index: " << N << " vectors, top-" << results.size() << " results" << std::endl;
}

/**
 * 测试3: GPU IVF 索引（CPU 回退）
 */
void test_gpu_ivf() {
    const size_t DIM = 128;
    const int N = 500;
    const int NCLUSTERS = 10;
    
    GPUIVFIndex ivf_index(DIM, NCLUSTERS, 5);  // nprobe=5
    std::mt19937 rng(42);
    
    // 生成训练数据
    std::vector<std::vector<float>> vectors;
    std::vector<float> train_data;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        train_data.insert(train_data.end(), vec.begin(), vec.end());
    }
    
    // 训练
    ivf_index.train(train_data, N);
    
    // 添加向量
    for (int i = 0; i < N; ++i) {
        ivf_index.add(i, vectors[i].data());
    }
    
    // 搜索
    auto query = vectors[0];
    auto results = ivf_index.search(query.data(), 10);
    
    TEST_ASSERT(!results.empty(), "IVF index should return results");
    std::cout << "  IVF index: " << N << " vectors, " << NCLUSTERS << " clusters, top-" << results.size() << " results" << std::endl;
}

/**
 * 测试4: Product Quantization
 */
void test_pq() {
    const int DIM = 128;
    const int M = 8;        // 子空间数
    const int KSUB = 256;   // 每子空间聚类数
    const int N = 1000;     // 训练向量数
    
    ProductQuantizer pq(M, KSUB, DIM);
    std::mt19937 rng(42);
    
    // 生成训练数据
    std::vector<float> train_data;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        train_data.insert(train_data.end(), vec.begin(), vec.end());
    }
    
    // 训练
    std::cout << "  Training PQ (M=" << M << ", ksub=" << KSUB << ")..." << std::endl;
    pq.train(train_data.data(), N, 5);  // 5 次迭代
    
    TEST_ASSERT(pq.is_trained(), "PQ should be trained");
    
    // 量化一个向量
    auto vec = random_vector(DIM, rng);
    auto codes = pq.encode(vec.data());
    
    TEST_ASSERT((int)codes.size() == M, "Code size should be M");
    std::cout << "  Quantized vector: " << M << " bytes (original: " << DIM * 4 << " bytes, compression: " << (DIM * 4.0 / M) << "x)" << std::endl;
    
    // 解码（近似重建）
    auto reconstructed = pq.decode(codes.data());
    
    // 计算重建误差
    float error = 0;
    for (int i = 0; i < DIM; ++i) {
        float diff = vec[i] - reconstructed[i];
        error += diff * diff;
    }
    error = std::sqrt(error / DIM);
    std::cout << "  Reconstruction RMSE: " << error << std::endl;
}

/**
 * 测试5: PQ 索引
 */
void test_pq_index() {
    const int DIM = 128;
    const int N = 500;
    
    PQIndex pq_index(8, 256, DIM);
    std::mt19937 rng(42);
    
    // 生成数据
    std::vector<std::vector<float>> vectors;
    std::vector<float> train_data;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        train_data.insert(train_data.end(), vec.begin(), vec.end());
    }
    
    // 训练
    pq_index.train(train_data, N);
    TEST_ASSERT(pq_index.is_trained(), "PQ index should be trained");
    
    // 添加向量
    for (int i = 0; i < N; ++i) {
        pq_index.add(i, vectors[i].data());
    }
    
    // 搜索
    auto query = vectors[0];
    auto results = pq_index.search(query.data(), 10);
    
    TEST_ASSERT(!results.empty(), "PQ index should return results");
    std::cout << "  PQ index: " << N << " vectors, top-" << results.size() << " results" << std::endl;
}

/**
 * 测试6: V3 多层索引
 */
void test_v3_tiered_index() {
    const size_t DIM = 128;
    const int N = 500;
    
    TieringPolicy tiering;
    tiering.hot_threshold = 100;
    tiering.warm_threshold = 300;
    tiering.enable_gpu = false;  // CPU 模式
    tiering.enable_pq = false;
    
    IndexV3 index(DIM, N * 2, QuantizeType::FP32, AutoRebuildPolicy(), tiering);
    std::mt19937 rng(42);
    
    // 插入数据
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        index.put(i, vec.data());
    }
    
    // 分层重建
    index.rebuild_with_tiering();
    
    // 检查分层统计
    auto tier_stats = index.tier_stats();
    std::cout << "  Tier stats: delta=" << tier_stats.delta_count 
              << ", base=" << tier_stats.base_count 
              << ", cold=" << tier_stats.cold_count << std::endl;
    
    // 搜索
    auto query = vectors[0];
    auto results = index.search(query.data(), 10);
    
    TEST_ASSERT(!results.empty(), "V3 index should return results");
    std::cout << "  Search returned " << results.size() << " results" << std::endl;
}

/**
 * 测试7: 分布式索引
 */
void test_distributed_index() {
    const size_t DIM = 128;
    const int N = 400;
    const int NSHARDS = 4;
    
    DistributedIndex dist_index(DIM, NSHARDS);
    std::mt19937 rng(42);
    
    // 插入数据（会自动分片）
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        dist_index.put(i, vec.data());
    }
    
    // 重建所有分片
    dist_index.rebuild_all();
    dist_index.wait_all_rebuilds();
    
    // 检查统计
    auto stats = dist_index.stats();
    std::cout << "  Distributed index: " << NSHARDS << " shards" << std::endl;
    std::cout << "  Total vectors: " << stats.total_vectors << std::endl;
    std::cout << "  Per shard: ";
    for (size_t i = 0; i < stats.shard_stats.size(); ++i) {
        std::cout << "[" << i << "]" << stats.shard_stats[i].live_vectors << " ";
    }
    std::cout << std::endl;
    
    // 搜索（fan-out/merge）
    auto query = vectors[0];
    auto results = dist_index.search(query.data(), 10);
    
    TEST_ASSERT(!results.empty(), "Distributed search should return results");
    std::cout << "  Search returned " << results.size() << " results" << std::endl;
}

/**
 * 测试8: V3 带 user_data
 */
void test_v3_with_user_data() {
    const size_t DIM = 128;
    const int N = 50;
    
    IndexV3 index(DIM, N * 2);
    std::mt19937 rng(42);
    
    // 插入带 user_data 的向量
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        std::string data = "v3_data_" + std::to_string(i);
        index.put_with_data(i, vec.data(), data.c_str(), data.size() + 1);
    }
    
    index.rebuild_with_tiering();
    
    // 搜索并验证 user_data
    auto query = random_vector(DIM, rng);
    auto results = index.search(query.data(), 10);
    
    for (const auto& r : results) {
        TEST_ASSERT(!r.user_data.empty(), "V3 search should return user_data");
        std::string data_str(reinterpret_cast<const char*>(r.user_data.data()));
        TEST_ASSERT(data_str.find("v3_data_") == 0, "user_data content mismatch");
    }
    
    std::cout << "  V3 with user_data: OK" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "kvann V3 Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        RUN_TEST(test_gpu_availability);
        RUN_TEST(test_gpu_brute_force);
        RUN_TEST(test_gpu_ivf);
        RUN_TEST(test_pq);
        RUN_TEST(test_pq_index);
        RUN_TEST(test_v3_tiered_index);
        RUN_TEST(test_distributed_index);
        RUN_TEST(test_v3_with_user_data);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL V3 TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED WITH EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
