// Cross-platform aligned allocation.
//
// Linux/macOS: posix_memalign / free
// Windows MSVC/MinGW: _aligned_malloc / _aligned_free
//
// Alignment must be a power of two and a multiple of sizeof(void*).
#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace kvann::detail {

inline void* aligned_alloc_bytes(std::size_t alignment, std::size_t bytes) {
    if (bytes == 0) {
        bytes = alignment;
    }
#if defined(_WIN32)
    void* p = _aligned_malloc(bytes, alignment);
    if (!p) throw std::bad_alloc();
    return p;
#else
    void* p = nullptr;
    if (::posix_memalign(&p, alignment, bytes) != 0 || !p) {
        throw std::bad_alloc();
    }
    return p;
#endif
}

inline void aligned_free(void* p) noexcept {
    if (!p) return;
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

} // namespace kvann::detail
