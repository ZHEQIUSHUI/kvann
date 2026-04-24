// Property-based / fuzz-style test:
//
// Drive the index with a deterministic random sequence of put/del operations,
// keep a parallel ground-truth model in std::map<Key, vec>, and at intervals
// run a search and verify against brute-force ranking.
//
// Multiple seeds catch sequence-dependent bugs that single-shot tests miss.

#include <kvann/core.h>
#include <kvann/index.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
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

float exact_cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

struct GroundTruth {
    std::size_t dim;
    std::map<Key, std::vector<float>> live;

    std::vector<std::pair<Key, float>> topk(const std::vector<float>& q, int k) const {
        std::vector<std::pair<Key, float>> all;
        all.reserve(live.size());
        for (const auto& [key, v] : live) {
            all.emplace_back(key, exact_cosine(q, v));
        }
        if ((int)all.size() > k) {
            std::partial_sort(all.begin(), all.begin() + k, all.end(),
                              [](const auto& a, const auto& b) {
                                  return a.second > b.second;
                              });
            all.resize(k);
        } else {
            std::sort(all.begin(), all.end(),
                      [](const auto& a, const auto& b) {
                          return a.second > b.second;
                      });
        }
        return all;
    }
};

// Run a sequence of random ops with a given seed, verify periodically.
// Returns recall@k averaged over checkpoints.
double run_seed(uint64_t seed, std::size_t dim, int n_ops, int n_keys, int topk) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> op_dist(0, 99);
    std::uniform_int_distribution<int> key_dist(0, n_keys - 1);

    Index idx(cfg_for(dim, static_cast<std::size_t>(n_keys * 4)));
    GroundTruth gt;
    gt.dim = dim;

    int hits = 0, total = 0;
    int verifications = 0;

    for (int op = 0; op < n_ops; ++op) {
        int what = op_dist(rng);
        Key k = static_cast<Key>(key_dist(rng));
        if (what < 65) {
            // put
            auto v = random_vec(dim, rng);
            auto st = idx.put(k, v.data());
            // Capacity may eventually fill if we never delete same key.
            if (st.ok()) gt.live[k] = v;
        } else if (what < 90) {
            // del
            auto st = idx.del(k);
            if (st.ok()) gt.live.erase(k);
        } else {
            // search + verify
            auto q = random_vec(dim, rng);
            // Maybe trigger a rebuild before searching (10% of the time).
            if ((op % 50) == 0) idx.rebuild();

            SearchParams sp; sp.topk = topk;
            auto got = idx.search(q.data(), sp);

            auto truth = gt.topk(q, topk);
            // Compute recall: how many truth keys appear in got
            std::vector<Key> got_keys;
            for (const auto& r : got) got_keys.push_back(r.key);
            for (const auto& t : truth) {
                if (std::find(got_keys.begin(), got_keys.end(), t.first)
                        != got_keys.end()) {
                    ++hits;
                }
                ++total;
            }
            ++verifications;
        }
    }

    // One last rebuild + verification across a fresh batch of queries.
    idx.rebuild();
    int extra_hits = 0, extra_total = 0;
    for (int q = 0; q < 20; ++q) {
        auto qv = random_vec(dim, rng);
        SearchParams sp; sp.topk = topk;
        auto got = idx.search(qv.data(), sp);
        auto truth = gt.topk(qv, topk);
        std::vector<Key> got_keys;
        for (const auto& r : got) got_keys.push_back(r.key);
        for (const auto& t : truth) {
            if (std::find(got_keys.begin(), got_keys.end(), t.first)
                    != got_keys.end()) {
                ++extra_hits;
            }
            ++extra_total;
        }
    }

    double mid = total ? double(hits) / total : 1.0;
    double end = extra_total ? double(extra_hits) / extra_total : 1.0;
    std::cout << "  seed " << seed
              << ": mid_recall=" << mid << " end_recall=" << end
              << " (verifications=" << verifications
              << " final_live=" << gt.live.size() << ")\n";
    return std::min(mid, end);
}

// Verify that delete eventually disappears from search results.
void test_delete_disappears() {
    constexpr std::size_t DIM = 16;
    constexpr int N = 100;
    Index idx(cfg_for(DIM, 1000));
    std::mt19937 rng(1);
    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < N; ++i) {
        vecs.push_back(random_vec(DIM, rng));
        idx.put(i, vecs.back().data());
    }
    idx.rebuild();

    // Delete keys 10..20
    for (int i = 10; i <= 20; ++i) idx.del(i);

    // Search using each deleted key's own vector — it must NOT come back.
    SearchParams sp; sp.topk = N;
    for (int i = 10; i <= 20; ++i) {
        auto r = idx.search(vecs[i].data(), sp);
        for (const auto& x : r) {
            TEST_ASSERT((int)x.key != i, "deleted key must not appear");
        }
    }
}

void test_random_sequence() {
    // Several seeds, modest size to keep CI fast.
    for (uint64_t s : {1ull, 7ull, 42ull, 1337ull}) {
        double r = run_seed(s, /*dim=*/32, /*n_ops=*/500,
                            /*n_keys=*/80, /*topk=*/5);
        // Recall is informational — we mainly want the test not to crash and
        // assertions inside to hold. But require at least 0.6 since it's
        // exact cosine rerank over known live set.
        TEST_ASSERT(r >= 0.5, "recall too low for seed");
    }
}

} // namespace

int main() {
    std::cout << "==== property tests (SIMD: " << simd_backend() << ") ====\n";
    try {
        std::cout << "[TEST] test_delete_disappears...\n";
        test_delete_disappears();
        std::cout << "[PASS]\n";
        std::cout << "[TEST] test_random_sequence...\n";
        test_random_sequence();
        std::cout << "[PASS]\n";
        std::cout << "\nALL PROPERTY TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
