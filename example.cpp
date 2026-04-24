// kvann 使用示例（v0.2 API）

#include <kvann/core.h>
#include <kvann/index.h>

#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<float> random_vector(std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    kvann::normalize_vector(v.data(), dim);
    return v;
}

void example_basic() {
    std::cout << "\n=== example: basic put/search ===\n";

    constexpr std::size_t DIM = 128;
    constexpr int N = 100;

    kvann::IndexConfig cfg;
    cfg.dim = DIM;
    cfg.max_elements = N * 2;

    kvann::Index index(cfg);
    std::mt19937 rng(42);

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < N; ++i) {
        auto v = random_vector(DIM, rng);
        vecs.push_back(v);
        auto st = index.put(static_cast<kvann::Key>(i), v.data());
        if (!st.ok()) std::cerr << "put failed: " << st.code_str() << "\n";
    }
    index.rebuild();

    kvann::SearchParams sp;
    sp.topk = 5;
    auto results = index.search(vecs[0].data(), sp);

    std::cout << "top-5 (SIMD backend = " << kvann::simd_backend() << "):\n";
    for (const auto& r : results) {
        std::cout << "  key=" << r.key << " score=" << r.score << "\n";
    }
}

void example_payload() {
    std::cout << "\n=== example: payload + filter + batch ===\n";

    constexpr std::size_t DIM = 64;
    constexpr int N = 50;

    kvann::IndexConfig cfg;
    cfg.dim = DIM;
    cfg.max_elements = N * 2;

    kvann::Index index(cfg);
    std::mt19937 rng(7);

    for (int i = 0; i < N; ++i) {
        auto v = random_vector(DIM, rng);
        std::string meta = "doc_" + std::to_string(i);
        index.put(i, v.data(), meta.data(), meta.size() + 1);
    }
    index.rebuild();

    kvann::SearchParams sp;
    sp.topk = 5;
    sp.include_payload = true;
    sp.filter = [](kvann::Key k) { return k % 2 == 0; };  // 仅偶数 key

    auto query = random_vector(DIM, rng);
    auto results = index.search(query.data(), sp);

    for (const auto& r : results) {
        std::cout << "  key=" << r.key << " score=" << r.score
                  << " payload=" << reinterpret_cast<const char*>(r.payload.data())
                  << "\n";
    }
}

} // namespace

int main() {
    std::cout << "kvann " << kvann::simd_backend() << " backend\n";
    example_basic();
    example_payload();
    return 0;
}
