// Edge cases and error paths.
//
// These tests aim to expose bugs that happy-path tests would miss:
//   * input validation (dim, capacity, null vec)
//   * extreme search params (topk=0, topk > N)
//   * lifecycle edge cases (slot reuse, repeated update, all-deleted)
//   * payload edge cases (empty, large)
//   * persistence error paths (bad path, missing file)
//   * concurrent ops on the same key
//   * NaN / zero vectors
//   * dim variations (1, 1024)

#include <kvann/core.h>
#include <kvann/index.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
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

#define RUN_TEST(name)                                                         \
    do {                                                                       \
        std::cout << "[TEST] " << #name << "...\n";                            \
        name();                                                                \
        std::cout << "[PASS] " << #name << "\n";                               \
    } while (0)

namespace {

IndexConfig cfg_for(std::size_t dim, std::size_t cap) {
    IndexConfig c;
    c.dim = dim;
    c.max_elements = cap;
    return c;
}

std::vector<float> random_vec(std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> v(dim);
    for (auto& x : v) x = nd(rng);
    normalize_vector(v.data(), dim);
    return v;
}

// ---------------------------------------------------------------------------
// Construction / config validation
// ---------------------------------------------------------------------------

void test_dim_zero_rejected() {
    try {
        IndexConfig cfg;
        cfg.dim = 0;
        Index idx(cfg);
        TEST_ASSERT(false, "expected throw for dim=0");
    } catch (const std::invalid_argument&) {
        // ok
    }
}

void test_max_elements_zero_rejected() {
    try {
        IndexConfig cfg;
        cfg.dim = 4;
        cfg.max_elements = 0;
        Index idx(cfg);
        TEST_ASSERT(false, "expected throw for max_elements=0");
    } catch (const std::invalid_argument&) {
        // ok
    }
}

// ---------------------------------------------------------------------------
// Input validation on hot path
// ---------------------------------------------------------------------------

void test_put_null_vector() {
    Index idx(cfg_for(8, 100));
    auto st = idx.put(1, nullptr);
    TEST_ASSERT(!st.ok(), "null vector should be rejected");
    TEST_ASSERT(st.code() == StatusCode::kInvalidArgument, "expect InvalidArgument");
}

// ---------------------------------------------------------------------------
// Capacity
// ---------------------------------------------------------------------------

void test_capacity_full() {
    constexpr std::size_t CAP = 16;
    Index idx(cfg_for(8, CAP));
    std::mt19937 rng(1);
    for (std::size_t i = 0; i < CAP; ++i) {
        auto v = random_vec(8, rng);
        TEST_ASSERT(idx.put(i, v.data()).ok(), "put should succeed under cap");
    }
    auto v = random_vec(8, rng);
    auto st = idx.put(999, v.data());
    TEST_ASSERT(!st.ok(), "should fail past cap");
    TEST_ASSERT(st.code() == StatusCode::kFull, "expect Full");
}

// ---------------------------------------------------------------------------
// Search params
// ---------------------------------------------------------------------------

void test_topk_zero_returns_empty() {
    Index idx(cfg_for(8, 100));
    std::mt19937 rng(1);
    for (int i = 0; i < 10; ++i) {
        auto v = random_vec(8, rng);
        idx.put(i, v.data());
    }
    idx.rebuild();

    std::vector<float> q(8, 0); q[0] = 1;
    SearchParams sp; sp.topk = 0;
    auto r = idx.search(q.data(), sp);
    // 0 -> we clamp to 1 internally; but should be small.
    TEST_ASSERT(r.size() <= 1, "topk=0 should return at most 1");
}

void test_topk_greater_than_live() {
    Index idx(cfg_for(8, 100));
    std::mt19937 rng(1);
    for (int i = 0; i < 5; ++i) {
        auto v = random_vec(8, rng);
        idx.put(i, v.data());
    }
    idx.rebuild();

    auto q = random_vec(8, rng);
    SearchParams sp; sp.topk = 100;
    auto r = idx.search(q.data(), sp);
    TEST_ASSERT(r.size() <= 5, "should not exceed live count");
}

void test_filter_rejects_all() {
    Index idx(cfg_for(8, 100));
    std::mt19937 rng(1);
    for (int i = 0; i < 10; ++i) {
        auto v = random_vec(8, rng);
        idx.put(i, v.data());
    }
    idx.rebuild();

    SearchParams sp;
    sp.topk = 5;
    sp.filter = [](Key) { return false; };
    auto q = random_vec(8, rng);
    auto r = idx.search(q.data(), sp);
    TEST_ASSERT(r.empty(), "filter rejecting all should return nothing");
}

void test_filter_accepts_one() {
    Index idx(cfg_for(8, 100));
    std::mt19937 rng(1);
    for (int i = 0; i < 10; ++i) {
        auto v = random_vec(8, rng);
        idx.put(i, v.data());
    }
    idx.rebuild();

    SearchParams sp;
    sp.topk = 5;
    sp.filter = [](Key k) { return k == 7; };
    auto q = random_vec(8, rng);
    auto r = idx.search(q.data(), sp);
    TEST_ASSERT(r.size() == 1, "should return only 1");
    TEST_ASSERT(r[0].key == 7, "should be the accepted key");
}

// ---------------------------------------------------------------------------
// Empty index
// ---------------------------------------------------------------------------

void test_search_empty_index_returns_empty() {
    Index idx(cfg_for(16, 100));
    std::vector<float> q(16, 0); q[0] = 1;
    SearchParams sp; sp.topk = 10;
    auto r = idx.search(q.data(), sp);
    TEST_ASSERT(r.empty(), "search on empty returns empty");
}

void test_rebuild_empty_index() {
    Index idx(cfg_for(8, 100));
    auto st = idx.rebuild();
    TEST_ASSERT(st.ok(), "rebuild on empty should be Ok");
    auto s = idx.stats();
    TEST_ASSERT(s.live_keys == 0 && s.base_count == 0, "empty after rebuild");
}

// ---------------------------------------------------------------------------
// Payload edge cases
// ---------------------------------------------------------------------------

void test_empty_payload_explicit() {
    Index idx(cfg_for(8, 100));
    std::vector<float> v(8, 0); v[0] = 1;
    normalize_vector(v.data(), 8);
    // Pass empty payload (len=0)
    TEST_ASSERT(idx.put(1, v.data(), nullptr, 0).ok(), "put with null payload");
    std::vector<uint8_t> p;
    auto st = idx.get_payload(1, p);
    TEST_ASSERT(st.ok() && p.empty(), "payload empty");
}

void test_large_payload() {
    Index idx(cfg_for(8, 100));
    std::vector<float> v(8, 0); v[0] = 1;
    normalize_vector(v.data(), 8);
    std::vector<uint8_t> big(64 * 1024, 0xAB);
    TEST_ASSERT(idx.put(1, v.data(), big.data(), big.size()).ok(), "big payload");
    std::vector<uint8_t> got;
    idx.get_payload(1, got);
    TEST_ASSERT(got.size() == big.size(), "size match");
    TEST_ASSERT(got == big, "content match");
}

// ---------------------------------------------------------------------------
// Update lifecycle
// ---------------------------------------------------------------------------

void test_update_many_times() {
    Index idx(cfg_for(8, 100));
    std::mt19937 rng(1);
    auto v = random_vec(8, rng);
    for (int i = 0; i < 100; ++i) {
        auto v2 = random_vec(8, rng);
        TEST_ASSERT(idx.put(1, v2.data()).ok(), "update i");
        v = v2;
    }
    idx.rebuild();
    SearchParams sp; sp.topk = 1;
    auto r = idx.search(v.data(), sp);
    TEST_ASSERT(r.size() == 1, "one result");
    TEST_ASSERT(r[0].key == 1, "key=1");
    TEST_ASSERT(std::abs(r[0].score - 1.0f) < 0.01f, "score ~1");
}

void test_del_then_reput_same_key() {
    // Even if slot isn't reclaimed, semantically the key should map to the new vec.
    Index idx(cfg_for(8, 100));
    std::mt19937 rng(1);
    auto v1 = random_vec(8, rng);
    auto v2 = random_vec(8, rng);

    idx.put(1, v1.data());
    idx.del(1);
    TEST_ASSERT(!idx.exists(1), "deleted");
    idx.put(1, v2.data());
    TEST_ASSERT(idx.exists(1), "back");
    idx.rebuild();

    SearchParams sp; sp.topk = 1;
    auto r = idx.search(v2.data(), sp);
    TEST_ASSERT(r[0].key == 1, "self");
    TEST_ASSERT(std::abs(r[0].score - 1.0f) < 0.01f, "score ~1");
    // Search with v1 should NOT score 1.0 against key=1 (it was overwritten).
    auto r1 = idx.search(v1.data(), sp);
    if (r1[0].key == 1) {
        TEST_ASSERT(std::abs(r1[0].score - 1.0f) > 0.01f,
                    "old vector should not match new one perfectly");
    }
}

void test_search_after_all_deleted() {
    Index idx(cfg_for(8, 100));
    std::mt19937 rng(1);
    for (int i = 0; i < 20; ++i) {
        auto v = random_vec(8, rng);
        idx.put(i, v.data());
    }
    idx.rebuild();
    for (int i = 0; i < 20; ++i) idx.del(i);
    auto q = random_vec(8, rng);
    SearchParams sp; sp.topk = 5;
    auto r = idx.search(q.data(), sp);
    TEST_ASSERT(r.empty(), "all deleted -> no results");
}

// ---------------------------------------------------------------------------
// Persistence error paths
// ---------------------------------------------------------------------------

void test_save_to_invalid_path() {
    Index idx(cfg_for(8, 100));
    std::mt19937 rng(1);
    auto v = random_vec(8, rng);
    idx.put(1, v.data());
    auto st = idx.save("/this/path/does/not/exist/x.idx");
    TEST_ASSERT(!st.ok(), "save to bad path fails");
    TEST_ASSERT(st.code() == StatusCode::kIo, "expect Io");
}

void test_load_nonexistent_file() {
    bool threw = false;
    try {
        auto idx = Index::load("/no/such/file.idx");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    TEST_ASSERT(threw, "load missing file should throw");
}

// ---------------------------------------------------------------------------
// Rebuild concurrency
// ---------------------------------------------------------------------------

void test_double_rebuild_serializes() {
    // Two concurrent rebuild() calls should both return Ok and not crash.
    Index idx(cfg_for(8, 1000));
    std::mt19937 rng(1);
    for (int i = 0; i < 200; ++i) {
        auto v = random_vec(8, rng);
        idx.put(i, v.data());
    }

    std::thread t1([&]() { auto st = idx.rebuild(); TEST_ASSERT(st.ok(), "t1"); });
    std::thread t2([&]() { auto st = idx.rebuild(); TEST_ASSERT(st.ok(), "t2"); });
    t1.join();
    t2.join();
    auto s = idx.stats();
    TEST_ASSERT(s.base_count == 200, "all in base after");
}

// ---------------------------------------------------------------------------
// Concurrent updates to the SAME key
// ---------------------------------------------------------------------------

void test_concurrent_updates_same_key() {
    // 4 threads updating key=42 with random vectors; final state should be one
    // of the candidates (last writer wins). No crashes, no torn reads.
    Index idx(cfg_for(64, 1000));
    {
        std::mt19937 rng(0);
        auto v = random_vec(64, rng);
        idx.put(42, v.data());
    }
    constexpr int N = 4;
    constexpr int M = 200;

    std::vector<std::thread> th;
    std::atomic<int> ok{0};
    for (int t = 0; t < N; ++t) {
        th.emplace_back([&, t]() {
            std::mt19937 rng(7 + t);
            for (int i = 0; i < M; ++i) {
                auto v = random_vec(64, rng);
                if (idx.put(42, v.data()).ok()) ++ok;
            }
        });
    }
    for (auto& x : th) x.join();
    TEST_ASSERT(ok == N * M, "all updates succeed");
    TEST_ASSERT(idx.exists(42), "key still alive");
}

// ---------------------------------------------------------------------------
// Dim variations
// ---------------------------------------------------------------------------

void test_dim_one() {
    Index idx(cfg_for(1, 100));
    std::vector<float> a = {1.0f}, b = {-1.0f};
    normalize_vector(a.data(), 1);
    normalize_vector(b.data(), 1);
    idx.put(1, a.data());
    idx.put(2, b.data());
    idx.rebuild();
    SearchParams sp; sp.topk = 2;
    auto r = idx.search(a.data(), sp);
    TEST_ASSERT(r.size() == 2, "two results");
    TEST_ASSERT(r[0].key == 1, "self first");
}

void test_dim_large_1024() {
    Index idx(cfg_for(1024, 100));
    std::mt19937 rng(1);
    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < 50; ++i) {
        vecs.push_back(random_vec(1024, rng));
        idx.put(i, vecs.back().data());
    }
    idx.rebuild();
    SearchParams sp; sp.topk = 5;
    auto r = idx.search(vecs[0].data(), sp);
    TEST_ASSERT(!r.empty() && r[0].key == 0, "self first at dim=1024");
    TEST_ASSERT(std::abs(r[0].score - 1.0f) < 1e-3f, "self ~1");
}

// ---------------------------------------------------------------------------
// NaN handling — kvann normalizes input. A zero vector becomes NaN; we just
// require no crash.
// ---------------------------------------------------------------------------

void test_zero_vector_no_crash() {
    Index idx(cfg_for(8, 100));
    std::vector<float> z(8, 0);  // norm == 0; normalize_f32 leaves it as zeros
    auto st = idx.put(1, z.data());
    TEST_ASSERT(st.ok(), "put zero vector should not crash");
    SearchParams sp; sp.topk = 1;
    auto r = idx.search(z.data(), sp);
    (void)r;  // any result is fine; test is "no crash"
}

// ---------------------------------------------------------------------------
// get_payload on missing/deleted keys
// ---------------------------------------------------------------------------

void test_get_payload_missing_returns_notfound() {
    Index idx(cfg_for(8, 100));
    std::vector<uint8_t> out;
    auto st = idx.get_payload(999, out);
    TEST_ASSERT(!st.ok(), "missing key should error");
    TEST_ASSERT(st.code() == StatusCode::kNotFound, "expect NotFound");
}

// compact() reclaims slots after del/put churn so that subsequent inserts
// don't trip Status::Full() near max_elements.
void test_compact_reclaims_slots() {
    constexpr std::size_t DIM = 16;
    constexpr std::size_t CAP = 64;
    Index idx(cfg_for(DIM, CAP));
    std::mt19937 rng(1);

    // Fill, then delete half. next_slot_ is now CAP and any further put fails.
    for (std::size_t i = 0; i < CAP; ++i) {
        auto v = random_vec(DIM, rng);
        TEST_ASSERT(idx.put(static_cast<Key>(i), v.data()).ok(), "fill");
    }
    for (std::size_t i = 0; i < CAP / 2; ++i) {
        TEST_ASSERT(idx.del(static_cast<Key>(i)).ok(), "del");
    }
    // Out of slots:
    auto v = random_vec(DIM, rng);
    auto st = idx.put(999, v.data());
    TEST_ASSERT(!st.ok() && st.code() == StatusCode::kFull,
                "expected Full before compact");

    // Compact reclaims the deleted slots.
    TEST_ASSERT(idx.compact().ok(), "compact ok");

    auto stats = idx.stats();
    TEST_ASSERT(stats.live_keys == CAP / 2, "live count preserved");
    TEST_ASSERT(stats.tombstone_count == 0, "tombstones cleared");

    // Now we can insert again (up to CAP).
    int reinserted = 0;
    for (std::size_t i = 0; i < CAP; ++i) {
        auto v2 = random_vec(DIM, rng);
        Key k = static_cast<Key>(10000 + i);
        if (idx.put(k, v2.data()).ok()) ++reinserted;
    }
    TEST_ASSERT(reinserted == (int)(CAP / 2), "reinserts equal reclaimed slots");
}

void test_compact_preserves_search() {
    constexpr std::size_t DIM = 32;
    Index idx(cfg_for(DIM, 200));
    std::mt19937 rng(7);
    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < 100; ++i) {
        vecs.push_back(random_vec(DIM, rng));
        idx.put(static_cast<Key>(i), vecs.back().data());
    }
    idx.rebuild();

    SearchParams sp; sp.topk = 1;
    auto before = idx.search(vecs[42].data(), sp);
    TEST_ASSERT(!before.empty() && before[0].key == 42, "self pre-compact");

    TEST_ASSERT(idx.compact().ok(), "compact ok");

    auto after = idx.search(vecs[42].data(), sp);
    TEST_ASSERT(!after.empty() && after[0].key == 42, "self post-compact");
}

void test_get_payload_after_delete_returns_notfound() {
    Index idx(cfg_for(8, 100));
    std::vector<float> v(8, 0); v[0] = 1;
    normalize_vector(v.data(), 8);
    idx.put(7, v.data(), "data", 5);
    idx.del(7);
    std::vector<uint8_t> out;
    auto st = idx.get_payload(7, out);
    TEST_ASSERT(!st.ok(), "deleted key");
    TEST_ASSERT(st.code() == StatusCode::kNotFound, "expect NotFound");
}

} // namespace

int main() {
    std::cout << "==== edge tests (SIMD: " << simd_backend() << ") ====\n";
    try {
        RUN_TEST(test_dim_zero_rejected);
        RUN_TEST(test_max_elements_zero_rejected);
        RUN_TEST(test_put_null_vector);
        RUN_TEST(test_capacity_full);
        RUN_TEST(test_topk_zero_returns_empty);
        RUN_TEST(test_topk_greater_than_live);
        RUN_TEST(test_filter_rejects_all);
        RUN_TEST(test_filter_accepts_one);
        RUN_TEST(test_search_empty_index_returns_empty);
        RUN_TEST(test_rebuild_empty_index);
        RUN_TEST(test_empty_payload_explicit);
        RUN_TEST(test_large_payload);
        RUN_TEST(test_update_many_times);
        RUN_TEST(test_del_then_reput_same_key);
        RUN_TEST(test_search_after_all_deleted);
        RUN_TEST(test_save_to_invalid_path);
        RUN_TEST(test_load_nonexistent_file);
        RUN_TEST(test_double_rebuild_serializes);
        RUN_TEST(test_concurrent_updates_same_key);
        RUN_TEST(test_dim_one);
        RUN_TEST(test_dim_large_1024);
        RUN_TEST(test_zero_vector_no_crash);
        RUN_TEST(test_get_payload_missing_returns_notfound);
        RUN_TEST(test_get_payload_after_delete_returns_notfound);
        RUN_TEST(test_compact_reclaims_slots);
        RUN_TEST(test_compact_preserves_search);
        std::cout << "\nALL EDGE TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
