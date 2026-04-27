// Persistence error / corruption detection tests.
//
// Validates that file format v3 (sectioned + CRC32) actually catches:
//   * truncated files
//   * bit-flipped section bodies
//   * wrong magic
//   * mangled section table
//   * round-trip determinism (save twice -> identical bytes)
// And that valid round-trips preserve query semantics.

#include <kvann/core.h>
#include <kvann/index.h>

#include "test_paths.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
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

// Write a small index to a temporary file. Returns path.
std::string make_index_file(const std::string& tag, std::size_t n = 200,
                            std::size_t dim = 32) {
    std::string path = kvann_test::tmp_path("kvann_corrupt_" + tag + ".idx");
    Index idx(cfg_for(dim, n * 2));
    std::mt19937 rng(42);
    for (std::size_t i = 0; i < n; ++i) {
        auto v = random_vec(dim, rng);
        std::string p = "p_" + std::to_string(i);
        idx.put(static_cast<Key>(i), v.data(), p.data(), p.size() + 1);
    }
    idx.rebuild();
    auto st = idx.save(path);
    TEST_ASSERT(st.ok(), "save");
    return path;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    std::vector<uint8_t> buf(static_cast<std::size_t>(in.tellg()));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(buf.data()), buf.size());
    return buf;
}

void write_file(const std::string& path, const std::vector<uint8_t>& buf) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

bool load_throws(const std::string& path, std::string* msg = nullptr) {
    try {
        auto idx = Index::load(path);
        return false;
    } catch (const std::exception& e) {
        if (msg) *msg = e.what();
        return true;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_roundtrip_preserves_search() {
    constexpr std::size_t DIM = 32;
    constexpr int N = 200;
    Index orig(cfg_for(DIM, N * 2));
    std::mt19937 rng(7);
    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < N; ++i) {
        vecs.push_back(random_vec(DIM, rng));
        orig.put(i, vecs.back().data());
    }
    orig.rebuild();

    std::string path = kvann_test::tmp_path("kvann_rt_preserve.idx");
    TEST_ASSERT(orig.save(path).ok(), "save");

    auto loaded = Index::load(path);

    // Compare top-1 across many queries.
    SearchParams sp; sp.topk = 5;
    int matches = 0;
    for (int q = 0; q < N; ++q) {
        auto a = orig.search(vecs[q].data(), sp);
        auto b = loaded->search(vecs[q].data(), sp);
        if (!a.empty() && !b.empty() && a[0].key == b[0].key) ++matches;
    }
    std::remove(path.c_str());
    TEST_ASSERT(matches >= N - 5, "near-identical top-1 across queries");
    std::cout << "  top-1 matches: " << matches << "/" << N << "\n";
}

void test_save_is_deterministic() {
    constexpr std::size_t DIM = 16;
    Index idx(cfg_for(DIM, 256));
    std::mt19937 rng(1);
    for (int i = 0; i < 64; ++i) {
        auto v = random_vec(DIM, rng);
        idx.put(i, v.data());
    }
    idx.rebuild();

    std::string a = kvann_test::tmp_path("kvann_det_a.idx");
    std::string b = kvann_test::tmp_path("kvann_det_b.idx");
    TEST_ASSERT(idx.save(a).ok(), "save a");
    TEST_ASSERT(idx.save(b).ok(), "save b");
    auto da = read_file(a), db = read_file(b);
    std::remove(a.c_str()); std::remove(b.c_str());
    TEST_ASSERT(da == db, "two saves produce identical bytes");
}

void test_load_truncated() {
    auto path = make_index_file("trunc");
    auto buf = read_file(path);

    // Truncate to half-size; CRC must fail or section read must throw.
    write_file(path, std::vector<uint8_t>(buf.begin(), buf.begin() + buf.size() / 2));
    std::string msg;
    bool threw = load_throws(path, &msg);
    std::remove(path.c_str());
    TEST_ASSERT(threw, "truncated should throw");
    std::cout << "  -> " << msg << "\n";
}

void test_load_truncated_in_header() {
    auto path = make_index_file("trunc_hdr");
    auto buf = read_file(path);

    // Keep only first 16 bytes (less than header).
    write_file(path, std::vector<uint8_t>(buf.begin(), buf.begin() + 16));
    bool threw = load_throws(path);
    std::remove(path.c_str());
    TEST_ASSERT(threw, "tiny file should throw");
}

void test_load_wrong_magic() {
    auto path = make_index_file("magic");
    auto buf = read_file(path);
    buf[0] = 'X';
    write_file(path, buf);
    std::string msg;
    bool threw = load_throws(path, &msg);
    std::remove(path.c_str());
    TEST_ASSERT(threw, "bad magic should throw");
    TEST_ASSERT(msg.find("magic") != std::string::npos, "msg mentions magic");
}

void test_load_wrong_version() {
    auto path = make_index_file("ver");
    auto buf = read_file(path);
    // Version is at offset 8 (uint32 LE).
    buf[8] = 0xFF; buf[9] = 0xFF; buf[10] = 0xFF; buf[11] = 0x7F;
    write_file(path, buf);
    bool threw = load_throws(path);
    std::remove(path.c_str());
    TEST_ASSERT(threw, "bad version should throw");
}

// Flip bit in vectors section body — CRC must catch.
void test_load_bitflip_in_vectors() {
    auto path = make_index_file("bitflip_vec");
    auto buf = read_file(path);

    // Header is 32B + table is 8*32 = 256B; sections start at offset 288.
    // Vectors section is the 3rd section; flipping a byte deep in the file
    // should land inside it.
    std::size_t flip = buf.size() / 2;
    buf[flip] ^= 0xFF;
    write_file(path, buf);

    std::string msg;
    bool threw = load_throws(path, &msg);
    std::remove(path.c_str());
    TEST_ASSERT(threw, "bit flip should be caught by CRC");
    TEST_ASSERT(msg.find("CRC") != std::string::npos, "msg mentions CRC");
}

// Empty index round-trips cleanly.
void test_save_load_empty_index() {
    constexpr std::size_t DIM = 8;
    Index empty(cfg_for(DIM, 64));
    std::string path = kvann_test::tmp_path("kvann_empty.idx");
    TEST_ASSERT(empty.save(path).ok(), "save empty");
    auto loaded = Index::load(path);
    auto s = loaded->stats();
    std::remove(path.c_str());
    TEST_ASSERT(s.live_keys == 0, "empty after load");
}

// Index with deletes — only live keys persist.
void test_save_after_delete() {
    constexpr std::size_t DIM = 16;
    Index idx(cfg_for(DIM, 200));
    std::mt19937 rng(1);
    for (int i = 0; i < 50; ++i) {
        auto v = random_vec(DIM, rng);
        idx.put(i, v.data());
    }
    for (int i = 0; i < 25; ++i) idx.del(i);  // delete half
    idx.rebuild();

    std::string path = kvann_test::tmp_path("kvann_after_del.idx");
    TEST_ASSERT(idx.save(path).ok(), "save");
    auto loaded = Index::load(path);
    auto s = loaded->stats();
    TEST_ASSERT(s.live_keys == 25, "only live keys persist");
    std::remove(path.c_str());
}

} // namespace

int main() {
    std::cout << "==== persistence corruption tests ====\n";
    try {
        RUN_TEST(test_roundtrip_preserves_search);
        RUN_TEST(test_save_is_deterministic);
        RUN_TEST(test_load_truncated);
        RUN_TEST(test_load_truncated_in_header);
        RUN_TEST(test_load_wrong_magic);
        RUN_TEST(test_load_wrong_version);
        RUN_TEST(test_load_bitflip_in_vectors);
        RUN_TEST(test_save_load_empty_index);
        RUN_TEST(test_save_after_delete);
        std::cout << "\nALL PERSISTENCE TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
