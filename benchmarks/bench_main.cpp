// kvann load benchmark
//
// Generates synthetic uniform-on-sphere vectors, reports throughput / latency /
// recall / persistence times. Output is a CSV-style block to stdout so CI can
// archive it as an artifact.
//
// Usage:
//   ./kvann_bench [--n N] [--dim D] [--queries Q] [--threads T]
//                 [--topk K] [--ef EF] [--save-path PATH]
//
// Defaults aim for "fits in CI runner under ~10s": n=20000, dim=128, q=200.

#include <kvann/core.h>
#include <kvann/index.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

struct Args {
    std::size_t n        = 20000;
    std::size_t dim      = 128;
    std::size_t queries  = 200;
    std::size_t threads  = 4;
    int         topk     = 10;
    int         ef       = 64;
    int         rebuild_threads = 1;  // 1=auto
    std::string save_path;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto match = [&](const char* k) { return std::strcmp(argv[i], k) == 0; };
        auto next  = [&](const char* k) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing arg for %s\n", k);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (match("--n"))         a.n        = std::stoul(next("--n"));
        else if (match("--dim"))       a.dim      = std::stoul(next("--dim"));
        else if (match("--queries"))   a.queries  = std::stoul(next("--queries"));
        else if (match("--threads"))   a.threads  = std::stoul(next("--threads"));
        else if (match("--topk"))      a.topk     = std::stoi(next("--topk"));
        else if (match("--ef"))        a.ef       = std::stoi(next("--ef"));
        else if (match("--rebuild-threads")) a.rebuild_threads = std::stoi(next("--rebuild-threads"));
        else if (match("--save-path")) a.save_path = next("--save-path");
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            std::exit(2);
        }
    }
    return a;
}

double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now() - t0).count();
}

void fill_random(float* v, std::size_t n, std::mt19937& rng) {
    std::normal_distribution<float> d(0, 1);
    for (std::size_t i = 0; i < n; ++i) v[i] = d(rng);
    kvann::normalize_vector(v, n);
}

double compute_recall(kvann::Index& idx, const std::vector<float>& vecs,
                      std::size_t n, std::size_t dim, int topk,
                      std::size_t n_queries) {
    std::size_t hits = 0;
    std::size_t total = 0;
    kvann::SearchParams sp;
    sp.topk = topk;
    for (std::size_t q = 0; q < n_queries && q < n; ++q) {
        const float* query = vecs.data() + q * dim;
        // Ground truth via brute-force.
        std::vector<std::pair<int, float>> all;
        all.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const float* v = vecs.data() + i * dim;
            float s = 0;
            for (std::size_t k = 0; k < dim; ++k) s += query[k] * v[k];
            all.emplace_back((int)i, s);
        }
        std::partial_sort(all.begin(), all.begin() + topk, all.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        auto ann = idx.search(query, sp);
        std::vector<int> truth(topk);
        for (int i = 0; i < topk; ++i) truth[i] = all[i].first;
        for (const auto& r : ann) {
            for (int t : truth) if ((int)r.key == t) { ++hits; break; }
        }
        total += topk;
    }
    return total ? double(hits) / double(total) : 0.0;
}

} // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    std::printf("# kvann benchmark\n");
    std::printf("# simd_backend = %s\n", kvann::simd_backend());
    std::printf("# n=%zu dim=%zu queries=%zu threads=%zu topk=%d ef=%d\n",
                a.n, a.dim, a.queries, a.threads, a.topk, a.ef);

    std::mt19937 rng(0xCAFEBEEF);
    std::vector<float> data(a.n * a.dim);
    {
        auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < a.n; ++i) {
            fill_random(data.data() + i * a.dim, a.dim, rng);
        }
        std::printf("metric,phase,unit,value\n");
        std::printf("data_gen,prep,ms,%.3f\n", elapsed_ms(t0));
    }

    kvann::IndexConfig cfg;
    cfg.dim          = a.dim;
    cfg.max_elements = a.n + 1024;
    cfg.hnsw_ef_search = a.ef;
    cfg.rebuild_threads = a.rebuild_threads;
    kvann::Index index(cfg);

    // -------- put --------
    {
        auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < a.n; ++i) {
            index.put(static_cast<kvann::Key>(i), data.data() + i * a.dim);
        }
        double ms = elapsed_ms(t0);
        std::printf("put,delta,ms,%.3f\n", ms);
        std::printf("put,delta,vec_per_sec,%.0f\n", a.n / (ms / 1000.0));
    }

    // -------- batch put smoke --------
    {
        std::vector<kvann::Key> keys(64);
        std::vector<float> vecs(64 * a.dim);
        for (int i = 0; i < 64; ++i) {
            keys[i] = a.n + 1 + i;
            fill_random(vecs.data() + i * a.dim, a.dim, rng);
        }
        auto t0 = std::chrono::steady_clock::now();
        auto st = index.put_batch(keys.data(), vecs.data(), keys.size());
        if (!st.ok()) std::fprintf(stderr, "put_batch failed\n");
        std::printf("put_batch,64,ms,%.3f\n", elapsed_ms(t0));
    }

    // -------- rebuild --------
    {
        auto t0 = std::chrono::steady_clock::now();
        index.rebuild();
        std::printf("rebuild,base,ms,%.3f\n", elapsed_ms(t0));
    }

    // -------- single-thread search --------
    kvann::SearchParams sp;
    sp.topk = a.topk;
    sp.ef   = a.ef;
    {
        auto t0 = std::chrono::steady_clock::now();
        std::size_t total_hits = 0;
        for (std::size_t q = 0; q < a.queries; ++q) {
            auto r = index.search(data.data() + q * a.dim, sp);
            total_hits += r.size();
        }
        double ms = elapsed_ms(t0);
        std::printf("search,1t,ms_per_query,%.4f\n", ms / a.queries);
        std::printf("search,1t,qps,%.1f\n", a.queries / (ms / 1000.0));
        std::printf("search,1t,avg_results,%.2f\n", double(total_hits) / a.queries);
    }

    // -------- multi-thread search --------
    if (a.threads > 1) {
        std::vector<std::thread> th;
        std::atomic<std::size_t> done{0};
        std::size_t per = a.queries;
        auto t0 = std::chrono::steady_clock::now();
        for (std::size_t t = 0; t < a.threads; ++t) {
            th.emplace_back([&, tid=t]() {
                std::mt19937 r(0xABCD + tid);
                std::uniform_int_distribution<std::size_t> dist(0, a.n - 1);
                for (std::size_t i = 0; i < per; ++i) {
                    auto idx = dist(r);
                    auto out = index.search(data.data() + idx * a.dim, sp);
                    if (!out.empty()) ++done;
                }
            });
        }
        for (auto& x : th) x.join();
        double ms = elapsed_ms(t0);
        double total_q = double(a.threads * per);
        std::printf("search,%zut,ms,%.3f\n", a.threads, ms);
        std::printf("search,%zut,qps,%.1f\n", a.threads, total_q / (ms / 1000.0));
    }

    // -------- recall@k --------
    {
        std::size_t nq = std::min<std::size_t>(50, a.queries);
        auto t0 = std::chrono::steady_clock::now();
        double recall = compute_recall(index, data, a.n, a.dim, a.topk, nq);
        std::printf("recall,@%d_%zuq,frac,%.4f\n", a.topk, nq, recall);
        std::printf("recall,@%d_%zuq,brute_force_ms,%.1f\n", a.topk, nq, elapsed_ms(t0));
    }

    // -------- save / load --------
    if (!a.save_path.empty()) {
        {
            auto t0 = std::chrono::steady_clock::now();
            auto st = index.save(a.save_path);
            if (!st.ok()) {
                std::fprintf(stderr, "save failed: %s\n", st.message().c_str());
                return 1;
            }
            std::printf("save,disk,ms,%.3f\n", elapsed_ms(t0));
        }
        {
            auto t0 = std::chrono::steady_clock::now();
            auto loaded = kvann::Index::load(a.save_path);
            std::printf("load,disk,ms,%.3f\n", elapsed_ms(t0));

            // Sanity: search returns same top-1 key.
            auto r1 = index.search(data.data(), sp);
            auto r2 = loaded->search(data.data(), sp);
            bool same = !r1.empty() && !r2.empty() && r1[0].key == r2[0].key;
            std::printf("load,sanity,top1_match,%d\n", same ? 1 : 0);
        }
        std::remove(a.save_path.c_str());
    }

    std::printf("# done\n");
    return 0;
}
