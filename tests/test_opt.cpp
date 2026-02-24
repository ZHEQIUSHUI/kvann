/**
 * 优化项测试（单版本）
 */

#include <kvann/index.h>
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

static std::vector<float> random_vector(size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> vec(dim);
    for (auto& v : vec) v = dist(rng);
    normalize_vector(vec.data(), dim);
    return vec;
}

void test_delta_hnsw_switch() {
    const size_t DIM = 128;
    IndexConfig cfg;
    cfg.max_elements = 10000;
    cfg.delta_bruteforce_limit = 16;
    cfg.delta_hnsw_threshold = 32;
    cfg.hnsw_ef_search = 64;

    Index index(DIM, cfg);
    std::mt19937 rng(42);
    std::vector<std::vector<float>> vectors;

    const int N = 100;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        index.put(i, vec.data());
    }

    auto stats = index.stats();
    TEST_ASSERT(stats.delta_count == (size_t)N, "All vectors should be in delta");

    // search should still work and find self
    auto results = index.search(vectors[10].data(), 5);
    TEST_ASSERT(!results.empty(), "Search should return results");
    bool found = false;
    for (const auto& r : results) {
        if (r.key == 10) {
            found = true;
            break;
        }
    }
    TEST_ASSERT(found, "Delta search should find self after HNSW switch");
}

void test_block_storage() {
    const size_t DIM = 128;
    IndexConfig cfg;
    cfg.max_elements = 1000;
    cfg.storage_block_size = 8; // force multi-block
    Index index(DIM, cfg);

    std::mt19937 rng(7);
    const int N = 30;
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        vectors.push_back(vec);
        index.put(i, vec.data());
    }

    index.rebuild();
    index.wait_rebuild();

    auto results = index.search(vectors[25].data(), 5);
    TEST_ASSERT(!results.empty(), "Search should return results across blocks");
    TEST_ASSERT(results[0].score > 0.9f, "Top score should be high");
}

void test_persistence_header() {
    const size_t DIM = 128;
    const char* TEST_FILE = "/tmp/kvann_opt_test.index";
    std::mt19937 rng(123);
    std::vector<std::vector<float>> vectors;

    {
        IndexConfig cfg;
        cfg.max_elements = 1000;
        cfg.storage_block_size = 16;
        Index index(DIM, cfg);
        for (int i = 0; i < 50; ++i) {
            auto vec = random_vector(DIM, rng);
            vectors.push_back(vec);
            index.put(i, vec.data());
        }
        index.rebuild();
        index.wait_rebuild();
        index.save(TEST_FILE);
    }

    {
        auto index = Index::load(TEST_FILE);
        TEST_ASSERT(index->exists(0), "Key should exist after load");
        auto results = index->search(vectors[0].data(), 5);
        TEST_ASSERT(!results.empty(), "Search should work after load");
    }

    std::remove(TEST_FILE);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "kvann Optimization Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        RUN_TEST(test_delta_hnsw_switch);
        RUN_TEST(test_block_storage);
        RUN_TEST(test_persistence_header);

        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL OPT TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED WITH EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
