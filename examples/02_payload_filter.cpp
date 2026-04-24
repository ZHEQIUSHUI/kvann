// 02_payload_filter.cpp
//
// Demonstrates two production-grade features:
//   1. Per-vector payload  (arbitrary bytes attached to each Key)
//   2. Search-side filter  (e.g. tag/permission-based result narrowing)

#include <kvann/core.h>
#include <kvann/index.h>

#include <iostream>
#include <random>
#include <string>
#include <vector>

int main() {
    constexpr std::size_t DIM = 64;
    constexpr int N = 200;

    kvann::IndexConfig cfg;
    cfg.dim          = DIM;
    cfg.max_elements = N * 2;
    kvann::Index index(cfg);

    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);

    for (int i = 0; i < N; ++i) {
        std::vector<float> v(DIM);
        for (auto& x : v) x = nd(rng);
        kvann::normalize_vector(v.data(), DIM);

        // Payload can be anything: serialized struct, JSON, pointer to row id, ...
        std::string payload = "doc_" + std::to_string(i);
        index.put(i, v.data(), payload.data(), payload.size() + 1);
    }
    index.rebuild();

    // Query with a filter that only allows even-numbered keys.
    std::vector<float> q(DIM);
    for (auto& x : q) x = nd(rng);
    kvann::normalize_vector(q.data(), DIM);

    kvann::SearchParams sp;
    sp.topk            = 5;
    sp.include_payload = true;
    sp.filter          = [](kvann::Key k) { return k % 2 == 0; };

    auto results = index.search(q.data(), sp);
    for (const auto& r : results) {
        std::cout << "key=" << r.key
                  << "  score=" << r.score
                  << "  payload=" << reinterpret_cast<const char*>(r.payload.data())
                  << "\n";
    }
    return 0;
}
