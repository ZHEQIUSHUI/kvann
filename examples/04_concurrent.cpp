// 04_concurrent.cpp
//
// Concurrent put + search. kvann is designed for read-mostly workloads where
// writes flow into the delta layer and search continues unblocked.

#include <kvann/core.h>
#include <kvann/index.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

int main() {
    constexpr std::size_t DIM = 128;
    constexpr int N = 2000;

    kvann::IndexConfig cfg;
    cfg.dim          = DIM;
    cfg.max_elements = N * 4;
    kvann::Index index(cfg);

    // Seed with some data and rebuild base.
    {
        std::mt19937 rng(0);
        std::normal_distribution<float> nd(0, 1);
        for (int i = 0; i < N; ++i) {
            std::vector<float> v(DIM);
            for (auto& x : v) x = nd(rng);
            kvann::normalize_vector(v.data(), DIM);
            index.put(i, v.data());
        }
        index.rebuild();
    }

    std::atomic<bool> stop{false};
    std::atomic<size_t> writes{0}, reads{0};

    std::thread writer([&]() {
        std::mt19937 rng(99);
        std::normal_distribution<float> nd(0, 1);
        int next_key = N;
        while (!stop.load()) {
            std::vector<float> v(DIM);
            for (auto& x : v) x = nd(rng);
            kvann::normalize_vector(v.data(), DIM);
            index.put(next_key++, v.data());
            ++writes;
        }
    });

    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t]() {
            std::mt19937 rng(100 + t);
            std::normal_distribution<float> nd(0, 1);
            kvann::SearchParams sp; sp.topk = 10;
            while (!stop.load()) {
                std::vector<float> q(DIM);
                for (auto& x : q) x = nd(rng);
                kvann::normalize_vector(q.data(), DIM);
                index.search(q.data(), sp);
                ++reads;
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    writer.join();
    for (auto& r : readers) r.join();

    auto s = index.stats();
    std::cout << "writes=" << writes.load()
              << "  reads="  << reads.load()
              << "  live="   << s.live_keys
              << "  base="   << s.base_count
              << "  delta="  << s.delta_count << "\n";
    return 0;
}
