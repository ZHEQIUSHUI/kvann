/**
 * user_data 功能测试
 */

#include <kvann/index.h>
#include <iostream>
#include <cstring>
#include <random>

using namespace kvann;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "ASSERTION FAILED: " << msg << " at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

void test_put_with_data() {
    const size_t DIM = 128;
    Index index(DIM, 1000);
    
    // 创建向量
    std::vector<float> vec(DIM);
    for (size_t i = 0; i < DIM; ++i) vec[i] = 1.0f;
    normalize_vector(vec.data(), DIM);
    
    // 创建 user_data
    const char* text = "Hello, kvann!";
    size_t text_len = strlen(text) + 1;
    
    // 插入带 user_data 的向量
    index.put_with_data(1, vec.data(), text, text_len);
    
    // 验证可以通过 get_user_data 获取
    auto retrieved = index.get_user_data(1);
    TEST_ASSERT(!retrieved.empty(), "user_data should not be empty");
    TEST_ASSERT(retrieved.size() == text_len, "user_data size mismatch");
    TEST_ASSERT(strcmp(reinterpret_cast<const char*>(retrieved.data()), text) == 0, 
                "user_data content mismatch");
    
    std::cout << "  [PASS] put_with_data and get_user_data" << std::endl;
}

void test_search_with_user_data() {
    const size_t DIM = 128;
    Index index(DIM, 1000);
    
    // 插入多个带 user_data 的向量
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);
    
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(DIM);
        for (auto& v : vec) v = dist(rng);
        normalize_vector(vec.data(), DIM);
        
        // 每个key有不同的user_data
        std::string data = "data_for_key_" + std::to_string(i);
        index.put_with_data(i, vec.data(), data.c_str(), data.size() + 1);
    }
    
    index.rebuild();
    index.wait_rebuild();
    
    // 搜索
    std::vector<float> query(DIM);
    for (auto& v : query) v = dist(rng);
    normalize_vector(query.data(), DIM);
    
    auto results = index.search(query.data(), 5);
    
    // 验证返回的结果包含 user_data
    for (const auto& r : results) {
        TEST_ASSERT(!r.user_data.empty(), "SearchResult should contain user_data");
        std::string expected_prefix = "data_for_key_";
        std::string actual(reinterpret_cast<const char*>(r.user_data.data()));
        TEST_ASSERT(actual.find(expected_prefix) == 0, 
                    "user_data should start with expected prefix");
    }
    
    std::cout << "  [PASS] search returns user_data" << std::endl;
}

void test_update_user_data() {
    const size_t DIM = 128;
    Index index(DIM, 1000);
    
    std::vector<float> vec(DIM);
    for (size_t i = 0; i < DIM; ++i) vec[i] = 1.0f;
    normalize_vector(vec.data(), DIM);
    
    // 第一次插入
    const char* data1 = "original_data";
    index.put_with_data(1, vec.data(), data1, strlen(data1) + 1);
    
    auto retrieved1 = index.get_user_data(1);
    TEST_ASSERT(strcmp(reinterpret_cast<const char*>(retrieved1.data()), data1) == 0,
                "Original data mismatch");
    
    // 更新 user_data（保持向量不变）
    const char* data2 = "updated_data";
    index.put_with_data(1, vec.data(), data2, strlen(data2) + 1);
    
    auto retrieved2 = index.get_user_data(1);
    TEST_ASSERT(strcmp(reinterpret_cast<const char*>(retrieved2.data()), data2) == 0,
                "Updated data mismatch");
    
    std::cout << "  [PASS] update user_data" << std::endl;
}

void test_persistence_with_user_data() {
    const size_t DIM = 128;
    const char* TEST_FILE = "/tmp/kvann_userdata_test.index";
    
    // 创建并保存
    {
        Index index(DIM, 1000);
        
        std::vector<float> vec(DIM);
        for (size_t i = 0; i < DIM; ++i) vec[i] = 0.5f;
        normalize_vector(vec.data(), DIM);
        
        const char* data = "persistent_data_12345";
        index.put_with_data(1, vec.data(), data, strlen(data) + 1);
        
        index.save(TEST_FILE);
    }
    
    // 加载并验证
    {
        auto index = Index::load(TEST_FILE);
        
        auto retrieved = index->get_user_data(1);
        TEST_ASSERT(!retrieved.empty(), "user_data should persist");
        TEST_ASSERT(strcmp(reinterpret_cast<const char*>(retrieved.data()), 
                          "persistent_data_12345") == 0,
                    "user_data content should persist correctly");
    }
    
    std::remove(TEST_FILE);
    std::cout << "  [PASS] persistence with user_data" << std::endl;
}

void test_empty_user_data() {
    const size_t DIM = 128;
    Index index(DIM, 1000);
    
    std::vector<float> vec(DIM);
    for (size_t i = 0; i < DIM; ++i) vec[i] = 1.0f;
    normalize_vector(vec.data(), DIM);
    
    // 使用普通 put（无 user_data）
    index.put(1, vec.data());
    
    auto retrieved = index.get_user_data(1);
    TEST_ASSERT(retrieved.empty(), "user_data should be empty for put without data");
    
    // 搜索也应该返回空的 user_data
    index.rebuild();
    index.wait_rebuild();
    
    auto results = index.search(vec.data(), 1);
    TEST_ASSERT(!results.empty(), "Should find the vector");
    TEST_ASSERT(results[0].user_data.empty(), "SearchResult user_data should be empty");
    
    std::cout << "  [PASS] empty user_data handling" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "user_data Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        std::cout << "\n[TEST] test_put_with_data..." << std::endl;
        test_put_with_data();
        
        std::cout << "\n[TEST] test_search_with_user_data..." << std::endl;
        test_search_with_user_data();
        
        std::cout << "\n[TEST] test_update_user_data..." << std::endl;
        test_update_user_data();
        
        std::cout << "\n[TEST] test_persistence_with_user_data..." << std::endl;
        test_persistence_with_user_data();
        
        std::cout << "\n[TEST] test_empty_user_data..." << std::endl;
        test_empty_user_data();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL USER_DATA TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED WITH EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
