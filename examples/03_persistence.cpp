// 03_persistence.cpp
//
// Builds an index, persists it to disk (file format v3 with HNSW graph dump),
// then loads it back. Cold load is O(file size) — no rebuild needed.

#include <kvann/core.h>
#include <kvann/index.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 5000;
    auto PATH_BUF = (std::filesystem::temp_directory_path()
                     / "kvann_persistence_demo.idx").string();
    const char* PATH = PATH_BUF.c_str();

    kvann::IndexConfig cfg;
    cfg.dim          = DIM;
    cfg.max_elements = N * 2;

    {
        kvann::Index index(cfg);
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0, 1);
        for (int i = 0; i < N; ++i) {
            std::vector<float> v(DIM);
            for (auto& x : v) x = nd(rng);
            kvann::normalize_vector(v.data(), DIM);
            index.put(i, v.data());
        }
        index.rebuild();

        auto t0 = std::chrono::steady_clock::now();
        auto st = index.save(PATH);
        if (!st.ok()) {
            std::cerr << "save failed: " << st.code_str() << "\n";
            return 1;
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "saved " << N << " vectors in " << ms << " ms\n";
    }

    {
        auto t0 = std::chrono::steady_clock::now();
        auto loaded = kvann::Index::load(PATH);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "loaded in " << ms << " ms (HNSW graph restored, no rebuild)\n";
        std::cout << "stats: live=" << loaded->stats().live_keys
                  << " base=" << loaded->stats().base_count << "\n";
    }

    std::remove(PATH);
    return 0;
}
