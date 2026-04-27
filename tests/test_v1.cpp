// kvann 功能测试（v0.2 API）
//
// 覆盖：CRUD / 搜索 / delta / tombstone / 更新 / rebuild / 持久化 /
// 多线程查询 / 多线程读写 / recall / 空索引 / 大规模

#include <kvann/core.h>
#include <kvann/index.h>

#include "test_paths.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace kvann;

namespace {

class Timer {
public:
    explicit Timer(std::string n)
        : name_(std::move(n)),
          start_(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << "  [Timer] " << name_ << ": " << ms << "ms\n";
    }
private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

#define TEST_ASSERT(cond, msg)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "ASSERT FAILED: " << msg << " @line " << __LINE__     \
                      << std::endl;                                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define RUN_TEST(name)                                                         \
    do {                                                                       \
        std::cout << "\n[TEST] " << #name << "...\n";                          \
        name();                                                                \
        std::cout << "[PASS] " << #name << "\n";                               \
    } while (0)

std::vector<float> random_vector(std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    normalize_vector(v.data(), dim);
    return v;
}

float compute_sim(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

IndexConfig cfg_for(std::size_t dim, std::size_t cap) {
    IndexConfig c;
    c.dim = dim;
    c.max_elements = cap;
    return c;
}

void test_basic_crud() {
    constexpr std::size_t DIM = 128;
    Index index(cfg_for(DIM, 10000));

    std::vector<float> v1(DIM, 0); v1[0] = 1.0f;
    std::vector<float> v2(DIM, 0); v2[1] = 1.0f;
    normalize_vector(v1.data(), DIM);
    normalize_vector(v2.data(), DIM);

    TEST_ASSERT(index.put(1, v1.data()).ok(), "put failed");
    TEST_ASSERT(index.exists(1), "should exist");
    TEST_ASSERT(index.put(1, v2.data()).ok(), "update failed");
    TEST_ASSERT(index.del(1).ok(), "delete failed");
    TEST_ASSERT(!index.exists(1), "should not exist after delete");
    TEST_ASSERT(!index.del(1).ok(), "second delete should fail");
    TEST_ASSERT(!index.del(999).ok(), "delete non-existent should fail");
    TEST_ASSERT(index.put(1, v1.data()).ok(), "put after delete failed");
    TEST_ASSERT(index.exists(1), "should exist after re-put");
}

void test_basic_search() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 100;
    Index index(cfg_for(DIM, 10000));
    std::mt19937 rng(42);

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < N; ++i) {
        auto v = random_vector(DIM, rng);
        vecs.push_back(v);
        TEST_ASSERT(index.put(i, v.data()).ok(), "put failed");
    }
    index.rebuild();

    SearchParams sp; sp.topk = 10;
    auto results = index.search(vecs[0].data(), sp);

    TEST_ASSERT(!results.empty(), "should return results");
    TEST_ASSERT(results[0].key == 0, "first should be self");
    TEST_ASSERT(results[0].score > 0.99f, "self similarity ~1.0");
    for (const auto& r : results) {
        float expected = compute_sim(vecs[0], vecs[r.key]);
        TEST_ASSERT(std::abs(r.score - expected) < 0.001f, "score mismatch");
    }
}

void test_delta_layer() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 50;
    Index index(cfg_for(DIM, 10000));
    std::mt19937 rng(42);

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < N; ++i) {
        auto v = random_vector(DIM, rng);
        vecs.push_back(v);
        index.put(i, v.data());
    }

    SearchParams sp; sp.topk = 5;
    auto results = index.search(vecs[10].data(), sp);
    TEST_ASSERT(!results.empty(), "delta search should work");
    for (const auto& r : results) {
        TEST_ASSERT(r.key < (Key)N, "result key range");
    }
}

void test_tombstone() {
    constexpr std::size_t DIM = 128;
    Index index(cfg_for(DIM, 10000));
    std::mt19937 rng(42);

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < 20; ++i) {
        auto v = random_vector(DIM, rng);
        vecs.push_back(v);
        index.put(i, v.data());
    }
    index.rebuild();

    index.del(5); index.del(10); index.del(15);

    SearchParams sp; sp.topk = 20;
    auto results = index.search(vecs[5].data(), sp);
    for (const auto& r : results) {
        TEST_ASSERT(r.key != 5,  "5 deleted");
        TEST_ASSERT(r.key != 10, "10 deleted");
        TEST_ASSERT(r.key != 15, "15 deleted");
    }
    auto v = random_vector(DIM, rng);
    index.put(5, v.data());
    TEST_ASSERT(index.exists(5), "re-insert");
}

void test_update() {
    constexpr std::size_t DIM = 128;
    Index index(cfg_for(DIM, 10000));
    std::mt19937 rng(42);

    auto v1 = random_vector(DIM, rng);
    auto v2 = random_vector(DIM, rng);

    index.put(1, v1.data());
    index.rebuild();
    index.put(1, v2.data());

    SearchParams sp; sp.topk = 1;
    auto results = index.search(v2.data(), sp);
    TEST_ASSERT(results[0].key == 1, "should find updated vec");
    TEST_ASSERT(std::abs(results[0].score - 1.0f) < 0.001f, "self ~1.0");
}

void test_rebuild() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 1000;
    Index index(cfg_for(DIM, 10000));
    std::mt19937 rng(42);

    {
        Timer t("Insert 1000");
        for (int i = 0; i < N; ++i) {
            auto v = random_vector(DIM, rng);
            index.put(i, v.data());
        }
    }
    auto sb = index.stats();
    TEST_ASSERT(sb.delta_count > 0, "delta should have entries");

    {
        Timer t("Rebuild 1000");
        index.rebuild();
    }
    auto sa = index.stats();
    TEST_ASSERT(sa.base_count == (std::size_t)N, "all in base");
    TEST_ASSERT(sa.delta_count == 0, "delta cleared");
}

void test_persistence() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 100;
    std::string PATH = kvann_test::tmp_path("kvann_v2_test.index");
    std::mt19937 rng(42);

    std::vector<std::vector<float>> vecs;
    {
        Index index(cfg_for(DIM, 10000));
        for (int i = 0; i < N; ++i) {
            auto v = random_vector(DIM, rng);
            vecs.push_back(v);
            index.put(i, v.data());
        }
        index.rebuild();
        TEST_ASSERT(index.save(PATH).ok(), "save failed");
    }
    {
        auto idx = Index::load(PATH);
        for (int i = 0; i < N; ++i) {
            TEST_ASSERT(idx->exists(i), "key after load");
        }
        SearchParams sp; sp.topk = 5;
        auto results = idx->search(vecs[0].data(), sp);
        TEST_ASSERT(!results.empty(), "search after load");
        idx->del(50);
        TEST_ASSERT(!idx->exists(50), "delete after load");
    }
    std::remove(PATH.c_str());
}

void test_concurrent_search() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 1000;
    constexpr int T = 4;
    constexpr int Q = 100;

    Index index(cfg_for(DIM, 10000));
    std::mt19937 rng(42);

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < N; ++i) {
        auto v = random_vector(DIM, rng);
        vecs.push_back(v);
        index.put(i, v.data());
    }
    index.rebuild();

    std::vector<std::thread> th;
    std::atomic<int> ok{0};
    {
        Timer t("Concurrent search " + std::to_string(T) + " threads");
        for (int i = 0; i < T; ++i) {
            th.emplace_back([&, tid=i]() {
                std::mt19937 lr(42 + tid);
                std::uniform_int_distribution<int> d(0, vecs.size()-1);
                SearchParams sp; sp.topk = 10;
                for (int j = 0; j < Q; ++j) {
                    int idx = d(lr);
                    auto r = index.search(vecs[idx].data(), sp);
                    if (!r.empty() && r[0].score > 0.9f) ok++;
                }
            });
        }
        for (auto& x : th) x.join();
    }
    TEST_ASSERT(ok == T * Q, "all should succeed");
}

void test_concurrent_readwrite() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 500;
    constexpr int W = 2, R = 4;

    Index index(cfg_for(DIM, 10000));
    std::atomic<bool> stop{false};
    std::atomic<int> wc{0}, rc{0};

    std::vector<std::thread> th;
    for (int i = 0; i < W; ++i) {
        th.emplace_back([&, tid=i]() {
            std::mt19937 rng(42 + tid);
            int k = tid;
            while (k < N && !stop.load()) {
                auto v = random_vector(DIM, rng);
                index.put(k, v.data());
                wc++;
                k += W;
            }
        });
    }
    for (int i = 0; i < R; ++i) {
        th.emplace_back([&, tid=i]() {
            std::mt19937 rng(100 + tid);
            std::vector<float> q(DIM);
            std::normal_distribution<float> d(0, 1);
            SearchParams sp; sp.topk = 5;
            while (!stop.load()) {
                for (auto& x : q) x = d(rng);
                normalize_vector(q.data(), DIM);
                index.search(q.data(), sp);
                if (++rc > 1000) break;
            }
        });
    }
    {
        Timer t("Concurrent r/w");
        for (auto& x : th) x.join();
    }
    TEST_ASSERT(wc > 0 && rc > 0, "had reads + writes");
    std::cout << "  W=" << wc << " R=" << rc << "\n";
}

void test_recall() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 1000;
    constexpr int K = 10;
    Index index(cfg_for(DIM, 10000));
    std::mt19937 rng(42);

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < N; ++i) {
        auto v = random_vector(DIM, rng);
        vecs.push_back(v);
        index.put(i, v.data());
    }
    index.rebuild();

    int hits = 0;
    SearchParams sp; sp.topk = K;
    for (int q = 0; q < 50; ++q) {
        auto& query = vecs[q];
        auto ann = index.search(query.data(), sp);

        std::vector<std::pair<int, float>> all;
        all.reserve(N);
        for (int i = 0; i < N; ++i) {
            all.emplace_back(i, compute_sim(query, vecs[i]));
        }
        std::partial_sort(all.begin(), all.begin() + K, all.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });

        std::unordered_set<int> ann_set;
        for (auto& r : ann) ann_set.insert(r.key);
        for (int i = 0; i < K; ++i) {
            if (ann_set.count(all[i].first)) ++hits;
        }
    }
    float recall = float(hits) / (50 * K);
    std::cout << "  Recall@" << K << " = " << recall << "\n";
    TEST_ASSERT(recall > 0.7f, "recall too low");
}

void test_empty() {
    Index index(cfg_for(128, 10000));
    std::vector<float> q(128);
    SearchParams sp; sp.topk = 10;
    auto r = index.search(q.data(), sp);
    TEST_ASSERT(r.empty(), "empty index returns empty");
    TEST_ASSERT(!index.exists(999), "no key");
    auto s = index.stats();
    TEST_ASSERT(s.total_keys == 0, "empty stats");
}

void test_large_scale() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 10000;
    Index index(cfg_for(DIM, 20000));
    std::mt19937 rng(42);

    {
        Timer t("Insert 10k");
        for (int i = 0; i < N; ++i) {
            auto v = random_vector(DIM, rng);
            index.put(i, v.data());
        }
    }
    {
        Timer t("Rebuild 10k");
        index.rebuild();
    }
    auto query = random_vector(DIM, rng);
    SearchParams sp; sp.topk = 10;
    {
        Timer t("Search 100x");
        for (int i = 0; i < 100; ++i) index.search(query.data(), sp);
    }
    TEST_ASSERT(index.stats().base_count == (std::size_t)N, "all in base");
}

void test_batch() {
    constexpr std::size_t DIM = 64;
    constexpr int N = 200;
    Index index(cfg_for(DIM, 10000));
    std::mt19937 rng(42);

    std::vector<Key> keys(N);
    std::vector<float> data(N * DIM);
    for (int i = 0; i < N; ++i) {
        keys[i] = i;
        auto v = random_vector(DIM, rng);
        std::copy(v.begin(), v.end(), data.begin() + i * DIM);
    }
    auto st = index.put_batch(keys.data(), data.data(), N);
    TEST_ASSERT(st.ok(), "put_batch ok");
    index.rebuild();

    std::vector<float> q(DIM);
    std::copy(data.begin(), data.begin() + DIM, q.begin());
    SearchParams sp; sp.topk = 5;
    auto r = index.search(q.data(), sp);
    TEST_ASSERT(!r.empty() && r[0].key == 0, "self");
}

} // namespace

int main() {
    std::cout << "==== kvann test (SIMD: " << simd_backend() << ") ====\n";
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
        RUN_TEST(test_empty);
        RUN_TEST(test_large_scale);
        RUN_TEST(test_batch);
        std::cout << "\nALL TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
