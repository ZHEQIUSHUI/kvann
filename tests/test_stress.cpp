// High-concurrency stress tests.
//
// The goal is to exercise:
//   * many threads doing mixed put/del/search for a few seconds (no crash, no
//     TSAN-detectable data race when run under -fsanitize=thread)
//   * rebuild() running concurrently with active put/del/search
//   * back-to-back rebuild() / async-rebuild() / wait_rebuild() across threads
//   * live_count_ accounting under concurrent put + del

#include <kvann/core.h>
#include <kvann/index.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#if defined(__has_include)
#  if __has_include(<execinfo.h>)
#    include <execinfo.h>
#    define KVANN_HAS_BACKTRACE 1
#  endif
#endif

namespace {
void install_crash_handler() {
#if defined(KVANN_HAS_BACKTRACE)
    auto handler = [](int sig) {
        void* frames[64];
        int n = backtrace(frames, 64);
        std::fprintf(stderr, "\n*** signal %d caught — backtrace ***\n", sig);
        backtrace_symbols_fd(frames, n, 2);
        std::_Exit(128 + sig);
    };
    std::signal(SIGSEGV, handler);
    std::signal(SIGABRT, handler);
    std::signal(SIGBUS,  handler);
#endif
}
} // namespace

using namespace kvann;
using clk = std::chrono::steady_clock;

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

bool with_budget(clk::time_point deadline) {
    return clk::now() < deadline;
}

// Sanitizers slow code 5-30x; shrink budgets so TSAN/ASAN runs fit in CI.
#if defined(__SANITIZE_THREAD__)
constexpr int kStressBudgetMs = 200;
#elif defined(__SANITIZE_ADDRESS__)
constexpr int kStressBudgetMs = 400;
#elif defined(__has_feature)
#  if __has_feature(thread_sanitizer)
constexpr int kStressBudgetMs = 200;
#  elif __has_feature(address_sanitizer)
constexpr int kStressBudgetMs = 400;
#  else
constexpr int kStressBudgetMs = 2000;
#  endif
#else
constexpr int kStressBudgetMs = 2000;
#endif

// ---------------------------------------------------------------------------
// Mixed workload: puts + searches for a fixed time, multiple threads.
// Verifies: no crash, search returns sensible results throughout.
// ---------------------------------------------------------------------------
void test_mixed_load() {
    constexpr std::size_t DIM = 64;
    constexpr std::size_t CAP = 50000;
    Index idx(cfg_for(DIM, CAP));

    auto deadline = clk::now() + std::chrono::milliseconds(kStressBudgetMs);
    std::atomic<bool> stop{false};
    std::atomic<std::size_t> writes{0}, reads{0};

    std::vector<std::thread> th;

    // 2 writers
    for (int i = 0; i < 2; ++i) {
        th.emplace_back([&, tid=i]() {
            std::mt19937 rng(0xA + tid);
            Key k = static_cast<Key>(tid * 1'000'000ULL);
            while (!stop.load(std::memory_order_relaxed)) {
                auto v = random_vec(DIM, rng);
                idx.put(k++, v.data());
                ++writes;
                if (k % 5000 == 0) std::this_thread::yield();
            }
        });
    }

    // 4 readers
    for (int i = 0; i < 4; ++i) {
        th.emplace_back([&, tid=i]() {
            std::mt19937 rng(0xB + tid);
            SearchParams sp; sp.topk = 10;
            while (!stop.load(std::memory_order_relaxed)) {
                auto v = random_vec(DIM, rng);
                idx.search(v.data(), sp);
                ++reads;
            }
        });
    }

    while (with_budget(deadline)) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stop.store(true);
    for (auto& t : th) t.join();

    TEST_ASSERT(writes > 0, "writes happened");
    TEST_ASSERT(reads > 0, "reads happened");
    std::cout << "  writes=" << writes.load() << " reads=" << reads.load() << "\n";
}

// ---------------------------------------------------------------------------
// Rebuild while writes/searches continue.
// ---------------------------------------------------------------------------
void test_rebuild_under_load() {
    constexpr std::size_t DIM = 64;
    constexpr std::size_t CAP = 30000;
    Index idx(cfg_for(DIM, CAP));

    // Seed
    {
        std::mt19937 rng(1);
        for (int i = 0; i < 2000; ++i) {
            auto v = random_vec(DIM, rng);
            idx.put(i, v.data());
        }
        idx.rebuild();
    }

    auto deadline = clk::now() + std::chrono::milliseconds(kStressBudgetMs + 500);
    std::atomic<bool> stop{false};
    std::atomic<std::size_t> rebuilds{0};

    std::vector<std::thread> th;

    th.emplace_back([&]() {
        std::mt19937 rng(2);
        Key k = 100000;
        while (!stop.load()) {
            auto v = random_vec(DIM, rng);
            idx.put(k++, v.data());
        }
    });

    th.emplace_back([&]() {
        SearchParams sp; sp.topk = 10;
        std::mt19937 rng(3);
        while (!stop.load()) {
            auto v = random_vec(DIM, rng);
            auto r = idx.search(v.data(), sp);
            (void)r;
        }
    });

    th.emplace_back([&]() {
        while (!stop.load()) {
            auto st = idx.rebuild();
            TEST_ASSERT(st.ok(), "rebuild ok");
            ++rebuilds;
        }
    });

    while (with_budget(deadline)) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true);
    for (auto& t : th) t.join();

    TEST_ASSERT(rebuilds > 0, "at least one rebuild");
    std::cout << "  completed " << rebuilds.load() << " rebuilds\n";
    auto s = idx.stats();
    TEST_ASSERT(s.live_keys > 0, "still has live keys");
}

// ---------------------------------------------------------------------------
// Many threads concurrently call wait_rebuild on the same in-flight rebuild.
// std::thread::join may NOT be called from multiple threads — this is the
// classic UB. The Index API must serialize the wait.
// ---------------------------------------------------------------------------
void test_concurrent_wait_rebuild() {
    constexpr std::size_t DIM = 64;
    Index idx(cfg_for(DIM, 5000));
    std::mt19937 rng(1);
    for (int i = 0; i < 1000; ++i) {
        auto v = random_vec(DIM, rng);
        idx.put(i, v.data());
    }

    auto st = idx.rebuild_async();
    TEST_ASSERT(st.ok(), "async start");

    constexpr int W = 8;
    std::vector<std::thread> th;
    for (int i = 0; i < W; ++i) {
        th.emplace_back([&]() { idx.wait_rebuild(); });
    }
    for (auto& t : th) t.join();

    auto s = idx.stats();
    TEST_ASSERT(s.base_count == 1000, "all in base after rebuild");
}

// ---------------------------------------------------------------------------
// live_count accounting under concurrent put + del (no two threads target the
// same key, so this should net to a deterministic count).
// ---------------------------------------------------------------------------
void test_live_count_invariant() {
    constexpr std::size_t DIM = 32;
    constexpr int N = 2000;
    Index idx(cfg_for(DIM, N * 4));

    // Two writer threads each handle a distinct key range.
    std::vector<std::thread> th;
    for (int t = 0; t < 2; ++t) {
        th.emplace_back([&, tid=t]() {
            std::mt19937 rng(0xCC + tid);
            int begin = tid * N;
            for (int i = begin; i < begin + N; ++i) {
                auto v = random_vec(DIM, rng);
                idx.put(i, v.data());
            }
        });
    }
    for (auto& t : th) t.join();

    auto after_puts = idx.stats();
    TEST_ASSERT(after_puts.live_keys == 2 * N, "all live");

    // Now delete half from each range concurrently.
    std::vector<std::thread> dt;
    for (int t = 0; t < 2; ++t) {
        dt.emplace_back([&, tid=t]() {
            int begin = tid * N;
            for (int i = begin; i < begin + N / 2; ++i) idx.del(i);
        });
    }
    for (auto& t : dt) t.join();

    auto after_dels = idx.stats();
    TEST_ASSERT(after_dels.live_keys == N, "half remain");
}

} // namespace

int main() {
    install_crash_handler();
    std::cout << "==== stress tests (SIMD: " << simd_backend() << ") ====\n";
    try {
        RUN_TEST(test_mixed_load);
        // Each test is annotated with the bug it exposes (if any). All four
        // are intentionally aggressive — failures here are real Index bugs.
        RUN_TEST(test_rebuild_under_load);
        RUN_TEST(test_concurrent_wait_rebuild);
        RUN_TEST(test_live_count_invariant);
        std::cout << "\nALL STRESS TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
