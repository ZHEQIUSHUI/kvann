/**
 * kvann V2 测试
 * 测试内容：
 * 1. 自动重建策略
 * 2. 量化支持（FP16, INT8）
 * 3. SIMD优化（如果启用）
 */

#include <kvann/index_v2.h>
#include <kvann/quantization.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

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
 * 测试1: FP16量化/反量化精度
 */
void test_fp16_conversion() {
    const size_t DIM = 128;
    std::mt19937 rng(42);
    auto vec = random_vector(DIM, rng);
    
    // FP32 -> FP16 -> FP32
    std::vector<uint16_t> fp16(DIM);
    std::vector<float> back_to_fp32(DIM);
    
#ifdef ENABLE_AVX2
    float32_to_float16_avx2(vec.data(), fp16.data(), DIM);
    float16_to_float32_avx2(fp16.data(), back_to_fp32.data(), DIM);
#else
    float32_to_float16_scalar(vec.data(), fp16.data(), DIM);
    float16_to_float32_scalar(fp16.data(), back_to_fp32.data(), DIM);
#endif
    
    // 检查精度损失
    float max_diff = 0;
    for (size_t i = 0; i < DIM; ++i) {
        max_diff = std::max(max_diff, std::abs(vec[i] - back_to_fp32[i]));
    }
    
    std::cout << "  FP16 max diff: " << max_diff << std::endl;
    // FP16有约3-4位有效数字，精度约为1e-3
    TEST_ASSERT(max_diff < 0.001f, "FP16 conversion error too large");
}

/**
 * 测试2: INT8量化/反量化
 */
void test_int8_quantization() {
    const size_t DIM = 128;
    std::mt19937 rng(42);
    auto vec = random_vector(DIM, rng);
    
    Int8Vector int8_vec;
    int8_vec.quantize(vec.data(), DIM);
    
    std::vector<float> back_to_fp32(DIM);
    int8_vec.dequantize(back_to_fp32.data(), DIM);
    
    // 检查精度
    float max_diff = 0;
    for (size_t i = 0; i < DIM; ++i) {
        max_diff = std::max(max_diff, std::abs(vec[i] - back_to_fp32[i]));
    }
    
    std::cout << "  INT8 max diff: " << max_diff << " (scale=" << int8_vec.scale << ")" << std::endl;
    // INT8误差应该小于scale
    TEST_ASSERT(max_diff <= int8_vec.scale * 1.5f, "INT8 quantization error too large");
}

/**
 * 测试3: SIMD点积（如果启用AVX2）
 */
void test_simd_dot_product() {
    const size_t DIM = 128;
    
    std::mt19937 rng(42);
    std::vector<float> a = random_vector(DIM, rng);
    std::vector<float> b = random_vector(DIM, rng);
    
    // 标量版本
    float dot_scalar = 0;
    for (size_t i = 0; i < DIM; ++i) {
        dot_scalar += a[i] * b[i];
    }
    
#ifdef ENABLE_AVX2
    float dot_avx2 = dot_product_avx2(a.data(), b.data(), DIM);
    
    std::cout << "  Scalar dot: " << dot_scalar << std::endl;
    std::cout << "  AVX2 dot: " << dot_avx2 << std::endl;
    std::cout << "  Diff: " << std::abs(dot_scalar - dot_avx2) << std::endl;
    
    TEST_ASSERT(std::abs(dot_scalar - dot_avx2) < 0.0001f, "AVX2 dot product mismatch");
    
    // 性能测试
    const int N = 100000;
    {
        auto start = std::chrono::high_resolution_clock::now();
        volatile float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum = dot_product_avx2(a.data(), b.data(), DIM);
        }
        (void)sum;
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  AVX2: " << us << " us for " << N << " iterations" << std::endl;
    }
    
    // 标量版本性能测试
    {
        auto start = std::chrono::high_resolution_clock::now();
        volatile float sum = 0;
        for (int i = 0; i < N; ++i) {
            float s = 0;
            for (size_t j = 0; j < DIM; ++j) s += a[j] * b[j];
            sum = s;
        }
        (void)sum;
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  Scalar: " << us << " us for " << N << " iterations" << std::endl;
    }
#else
    float dot_scalar2 = dot_product_scalar(a.data(), b.data(), DIM);
    std::cout << "  AVX2 not enabled, scalar dot: " << dot_scalar2 << std::endl;
    (void)dot_scalar;  // 避免警告
#endif
}

/**
 * 测试4: 自动重建策略 - 检查阈值
 */
void test_auto_rebuild_policy() {
    const size_t DIM = 128;
    const int N = 200;
    
    AutoRebuildPolicy policy;
    policy.auto_rebuild = false;  // 手动检查
    policy.tombstone_ratio_threshold = 0.25f;
    policy.delta_ratio_threshold = 0.3f;
    
    IndexV2 index(DIM, 10000, QuantizeType::FP32, policy);
    std::mt19937 rng(42);
    
    // 插入数据
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        index.put(i, vec.data());
    }
    
    index.rebuild();
    index.wait_rebuild();
    
    TEST_ASSERT(!index.needs_rebuild(), "Should not need rebuild after fresh rebuild");
    
    // 删除30%的数据（超过25%阈值）
    for (int i = 0; i < N * 0.3; ++i) {
        index.del(i);
    }
    
    TEST_ASSERT(index.needs_rebuild(), "Should need rebuild after many deletes");
    
    std::cout << "  needs_rebuild=true after 30% deletes" << std::endl;
}

/**
 * 测试5: V2与V1兼容性
 */
void test_v2_v1_compatibility() {
    const size_t DIM = 128;
    const int N = 100;
    
    // 使用V2但使用FP32（应该等同于V1）
    IndexV2 index(DIM, 10000, QuantizeType::FP32);
    std::mt19937 rng(42);
    std::vector<std::vector<float>> vectors;
    
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        index.put(i, vec.data());
    }
    
    index.rebuild();
    index.wait_rebuild();
    
    // 基本CRUD测试
    TEST_ASSERT(index.exists(0), "Key should exist");
    TEST_ASSERT(index.exists(50), "Key should exist");
    
    index.del(50);
    TEST_ASSERT(!index.exists(50), "Key should be deleted");
    
    // 搜索测试
    auto query = vectors[0];
    auto results = index.search(query.data(), 10);
    TEST_ASSERT(!results.empty(), "Search should return results");
    
    // recall测试
    bool self_found = false;
    for (const auto& r : results) {
        if (r.key == 0) {
            self_found = true;
            break;
        }
    }
    TEST_ASSERT(self_found, "Should find self");
    
    std::cout << "  V2 basic functionality OK" << std::endl;
}

/**
 * 测试6: V2量化类型设置
 */
void test_v2_quantize_type() {
    const size_t DIM = 128;
    
    IndexV2 index_fp32(DIM, 10000, QuantizeType::FP32);
    IndexV2 index_fp16(DIM, 10000, QuantizeType::FP16);
    IndexV2 index_int8(DIM, 10000, QuantizeType::INT8);
    
    TEST_ASSERT(index_fp32.quantize_type() == QuantizeType::FP32, "Type should be FP32");
    TEST_ASSERT(index_fp16.quantize_type() == QuantizeType::FP16, "Type should be FP16");
    TEST_ASSERT(index_int8.quantize_type() == QuantizeType::INT8, "Type should be INT8");
    
    std::cout << "  Quantize types set correctly" << std::endl;
}

/**
 * 测试7: QuantizedVectorStorage 点积与更新
 */
void test_quantized_storage_dot_and_update() {
    const size_t DIM = 128;
    std::mt19937 rng(42);
    auto vec1 = random_vector(DIM, rng);
    auto vec2 = random_vector(DIM, rng);
    
    // FP16
    {
        QuantizedVectorStorage qs(DIM, QuantizeType::FP16);
        qs.store(0, vec1.data());
        float dot1 = qs.dot(0, vec1.data());
        TEST_ASSERT(dot1 > 0.98f, "FP16 dot with same vector should be high");
        
        qs.store(0, vec2.data());
        float dot2 = qs.dot(0, vec2.data());
        TEST_ASSERT(dot2 > 0.98f, "FP16 update should take effect");
    }
    
    // INT8
    {
        QuantizedVectorStorage qs(DIM, QuantizeType::INT8);
        qs.store(0, vec1.data());
        float dot1 = qs.dot(0, vec1.data());
        TEST_ASSERT(dot1 > 0.9f, "INT8 dot with same vector should be high");
        
        qs.store(0, vec2.data());
        float dot2 = qs.dot(0, vec2.data());
        TEST_ASSERT(dot2 > 0.9f, "INT8 update should take effect");
    }
    
    std::cout << "  QuantizedVectorStorage dot/update OK" << std::endl;
}

/**
 * 测试8: V2量化搜索（Delta + Base）
 */
void test_v2_quantized_search() {
    const size_t DIM = 128;
    const int N = 200;
    std::mt19937 rng(42);
    
    // INT8 量化
    IndexV2 index(DIM, 10000, QuantizeType::INT8);
    std::vector<std::vector<float>> vectors;
    vectors.reserve(N);
    
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        index.put(i, vec.data());
    }
    
    // Delta 层搜索（未重建）
    {
        auto results = index.search(vectors[10].data(), 5);
        TEST_ASSERT(!results.empty(), "INT8 delta search should return results");
        bool found = false;
        for (const auto& r : results) {
            if (r.key == 10) {
                found = true;
                break;
            }
        }
        TEST_ASSERT(found, "INT8 delta search should find self");
    }
    
    // Base 层搜索（重建后）
    index.rebuild();
    index.wait_rebuild();
    {
        auto results = index.search(vectors[20].data(), 5);
        TEST_ASSERT(!results.empty(), "INT8 base search should return results");
        bool found = false;
        for (const auto& r : results) {
            if (r.key == 20) {
                found = true;
                break;
            }
        }
        TEST_ASSERT(found, "INT8 base search should find self");
    }
    
    std::cout << "  V2 INT8 search OK" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "kvann V2 Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
#ifdef ENABLE_AVX2
    std::cout << "AVX2: ENABLED" << std::endl;
#else
    std::cout << "AVX2: DISABLED" << std::endl;
#endif
    
    try {
        RUN_TEST(test_fp16_conversion);
        RUN_TEST(test_int8_quantization);
        RUN_TEST(test_simd_dot_product);
        RUN_TEST(test_auto_rebuild_policy);
        RUN_TEST(test_v2_v1_compatibility);
        RUN_TEST(test_v2_quantize_type);
        RUN_TEST(test_quantized_storage_dot_and_update);
        RUN_TEST(test_v2_quantized_search);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL V2 TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED WITH EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
