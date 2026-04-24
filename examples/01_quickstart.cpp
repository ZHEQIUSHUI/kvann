// 01_quickstart.cpp
//
// Build a small index, run a search, print results. The minimum viable
// integration of kvann.

#include <kvann/core.h>
#include <kvann/index.h>

#include <iostream>
#include <random>
#include <vector>

int main() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 1000;

    kvann::IndexConfig cfg;
    cfg.dim          = DIM;
    cfg.max_elements = N * 2;

    kvann::Index index(cfg);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> vecs(N, std::vector<float>(DIM));
    for (int i = 0; i < N; ++i) {
        for (auto& x : vecs[i]) x = dist(rng);
        kvann::normalize_vector(vecs[i].data(), DIM);
        index.put(static_cast<kvann::Key>(i), vecs[i].data());
    }

    // Building base eagerly is optional but improves search recall.
    index.rebuild();

    kvann::SearchParams sp;
    sp.topk = 5;
    auto results = index.search(vecs[0].data(), sp);

    std::cout << "kvann backend: " << kvann::simd_backend() << "\n";
    std::cout << "top-5 for query=key_0:\n";
    for (const auto& r : results) {
        std::cout << "  key=" << r.key << "  score=" << r.score << "\n";
    }
    return 0;
}
