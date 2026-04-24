// kvann payload (user_data) tests — v0.2 API

#include <kvann/core.h>
#include <kvann/index.h>

#include <cstdio>
#include <cstring>
#include <iostream>
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

void test_put_with_payload() {
    constexpr std::size_t DIM = 128;
    Index index(cfg_for(DIM, 1000));

    std::vector<float> v(DIM, 1.0f);
    normalize_vector(v.data(), DIM);

    const char* text = "Hello, kvann!";
    auto st = index.put(1, v.data(), text, std::strlen(text) + 1);
    TEST_ASSERT(st.ok(), "put failed");

    std::vector<uint8_t> payload;
    TEST_ASSERT(index.get_payload(1, payload).ok(), "get_payload");
    TEST_ASSERT(payload.size() == std::strlen(text) + 1, "size");
    TEST_ASSERT(std::strcmp(reinterpret_cast<const char*>(payload.data()), text) == 0,
                "content");
    std::cout << "  [PASS] put + get_payload\n";
}

void test_search_includes_payload() {
    constexpr std::size_t DIM = 128;
    Index index(cfg_for(DIM, 1000));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);

    for (int i = 0; i < 10; ++i) {
        std::vector<float> v(DIM);
        for (auto& x : v) x = dist(rng);
        normalize_vector(v.data(), DIM);

        std::string data = "data_for_key_" + std::to_string(i);
        index.put(i, v.data(), data.data(), data.size() + 1);
    }
    index.rebuild();

    std::vector<float> q(DIM);
    for (auto& x : q) x = dist(rng);
    normalize_vector(q.data(), DIM);

    SearchParams sp;
    sp.topk = 5;
    sp.include_payload = true;
    auto r = index.search(q.data(), sp);

    for (const auto& s : r) {
        TEST_ASSERT(!s.payload.empty(), "payload populated");
        std::string actual(reinterpret_cast<const char*>(s.payload.data()));
        TEST_ASSERT(actual.find("data_for_key_") == 0, "prefix");
    }
    std::cout << "  [PASS] search includes payload\n";
}

void test_update_payload() {
    constexpr std::size_t DIM = 128;
    Index index(cfg_for(DIM, 1000));
    std::vector<float> v(DIM, 1.0f);
    normalize_vector(v.data(), DIM);

    const char* a = "original";
    index.put(1, v.data(), a, std::strlen(a) + 1);
    std::vector<uint8_t> p;
    index.get_payload(1, p);
    TEST_ASSERT(std::strcmp(reinterpret_cast<const char*>(p.data()), a) == 0, "v1");

    const char* b = "updated_data";
    index.put(1, v.data(), b, std::strlen(b) + 1);
    index.get_payload(1, p);
    TEST_ASSERT(std::strcmp(reinterpret_cast<const char*>(p.data()), b) == 0, "v2");
    std::cout << "  [PASS] update payload\n";
}

void test_persistence_with_payload() {
    constexpr std::size_t DIM = 128;
    const char* PATH = "/tmp/kvann_v2_userdata.index";
    {
        Index index(cfg_for(DIM, 1000));
        std::vector<float> v(DIM, 0.5f);
        normalize_vector(v.data(), DIM);
        const char* d = "persistent_12345";
        index.put(1, v.data(), d, std::strlen(d) + 1);
        TEST_ASSERT(index.save(PATH).ok(), "save");
    }
    {
        auto idx = Index::load(PATH);
        std::vector<uint8_t> p;
        TEST_ASSERT(idx->get_payload(1, p).ok(), "get after load");
        TEST_ASSERT(std::strcmp(reinterpret_cast<const char*>(p.data()),
                                "persistent_12345") == 0, "content");
    }
    std::remove(PATH);
    std::cout << "  [PASS] persistence with payload\n";
}

void test_no_payload() {
    constexpr std::size_t DIM = 128;
    Index index(cfg_for(DIM, 1000));
    std::vector<float> v(DIM, 1.0f);
    normalize_vector(v.data(), DIM);

    index.put(1, v.data());
    std::vector<uint8_t> p;
    auto st = index.get_payload(1, p);
    TEST_ASSERT(st.ok() && p.empty(), "empty payload for plain put");
    std::cout << "  [PASS] no-payload\n";
}

} // namespace

int main() {
    std::cout << "==== user_data tests ====\n";
    try {
        test_put_with_payload();
        test_search_includes_payload();
        test_update_payload();
        test_persistence_with_payload();
        test_no_payload();
        std::cout << "\nALL USER_DATA TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
