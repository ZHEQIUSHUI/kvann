// CRC32 (IEEE 802.3 polynomial, reflected) — table-based, header-only.
//
// Used for kvann file format v3 section integrity checks. Not cryptographic;
// catches torn writes and bit rot.
#pragma once

#include <cstddef>
#include <cstdint>

namespace kvann::detail {

inline uint32_t crc32_table_value(uint32_t i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) {
        c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    }
    return c;
}

// Compile-time-ish table built on first use.
inline const uint32_t* crc32_table() {
    static uint32_t t[256] = {};
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) t[i] = crc32_table_value(i);
        init = true;
    }
    return t;
}

inline uint32_t crc32_update(uint32_t crc, const void* data, std::size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const uint32_t* t = crc32_table();
    crc = ~crc;
    while (len--) crc = t[(crc ^ *p++) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

inline uint32_t crc32(const void* data, std::size_t len) {
    return crc32_update(0u, data, len);
}

} // namespace kvann::detail
