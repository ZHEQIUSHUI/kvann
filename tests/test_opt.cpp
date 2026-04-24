// kvann optimization-path tests — v0.2 API
//
// Exercises:
//   - delta brute-force <-> delta HNSW switch
//   - aligned block storage at small block sizes
//   - persistence header (v2 magic)
//   - SIMD backend reporting

#include <kvann/core.h>
#include <kvann/index.h>

#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

using namespace kvann;

#define TEST_ASSERT(cond, msg)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "ASSERT FAILED: " << msg << " @line " << __LINE__     \
                      << std::endl;                                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

std::vector<float> random_vector(std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    normalize_vector(v.data(), dim);
    return v;
}

void test_delta_hnsw_switch() {
    constexpr std::size_t DIM = 128;
    IndexConfig cfg;
    cfg.dim = DIM;
    cfg.max_elements = 10000;
    cfg.delta_bruteforce_limit = 16;
    cfg.delta_hnsw_threshold = 32;
    cfg.hnsw_ef_search = 64;

    Index index(cfg);
    std::mt19937 rng(42);

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < 100; ++i) {
        auto v = random_vector(DIM, rng);
        vecs.push_back(v);
        index.put(i, v.data());
    }

    auto s = index.stats();
    TEST_ASSERT(s.delta_count > cfg.delta_hnsw_threshold,
                "delta should exceed threshold");

    SearchParams sp; sp.topk = 5;
    auto r = index.search(vecs[10].data(), sp);
    TEST_ASSERT(!r.empty(), "search returns");
    bool found = false;
    for (auto& x : r) if (x.key == 10) { found = true; break; }
    TEST_ASSERT(found, "self in top-5 after switch");
    std::cout << "  [PASS] delta HNSW switch\n";
}

void test_block_storage() {
    constexpr std::size_t DIM = 128;
    IndexConfig cfg;
    cfg.dim = DIM;
    cfg.max_elements = 1000;
    cfg.storage_block_size = 8;  // many blocks

    Index index(cfg);
    std::mt19937 rng(7);

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < 30; ++i) {
        auto v = random_vector(DIM, rng);
        vecs.push_back(v);
        index.put(i, v.data());
    }
    index.rebuild();

    SearchParams sp; sp.topk = 5;
    auto r = index.search(vecs[25].data(), sp);
    TEST_ASSERT(!r.empty(), "search across blocks");
    TEST_ASSERT(r[0].score > 0.9f, "top score high");
    std::cout << "  [PASS] block storage\n";
}

void test_persistence_header() {
    constexpr std::size_t DIM = 128;
    const char* PATH = "/tmp/kvann_opt.index";
    std::mt19937 rng(123);

    std::vector<std::vector<float>> vecs;
    {
        IndexConfig cfg; cfg.dim = DIM; cfg.max_elements = 1000;
        cfg.storage_block_size = 16;
        Index index(cfg);
        for (int i = 0; i < 50; ++i) {
            auto v = random_vector(DIM, rng);
            vecs.push_back(v);
            index.put(i, v.data());
        }
        index.rebuild();
        TEST_ASSERT(index.save(PATH).ok(), "save ok");
    }
    {
        auto idx = Index::load(PATH);
        TEST_ASSERT(idx->exists(0), "key 0 after load");
        SearchParams sp; sp.topk = 5;
        auto r = idx->search(vecs[0].data(), sp);
        TEST_ASSERT(!r.empty(), "search after load");
    }
    std::remove(PATH);
    std::cout << "  [PASS] persistence header\n";
}

void test_simd_backend_reported() {
    const char* be = simd_backend();
    TEST_ASSERT(be != nullptr, "backend non-null");
    std::cout << "  SIMD backend: " << be << "\n";
    std::cout << "  [PASS] simd backend reporting\n";
}

} // namespace

int main() {
    std::cout << "==== kvann opt tests ====\n";
    try {
        test_delta_hnsw_switch();
        test_block_storage();
        test_persistence_header();
        test_simd_backend_reported();
        std::cout << "\nALL OPT TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
