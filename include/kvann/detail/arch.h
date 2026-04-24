// Compile-time platform/architecture detection for kvann.
#pragma once

// ---- OS ----
#if defined(_WIN32)
#define KVANN_OS_WINDOWS 1
#elif defined(__linux__)
#define KVANN_OS_LINUX 1
#elif defined(__APPLE__)
#define KVANN_OS_MACOS 1
#endif

// ---- Architecture ----
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define KVANN_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define KVANN_ARCH_AARCH64 1
#elif defined(__arm__) || defined(_M_ARM)
#define KVANN_ARCH_ARM 1
#endif

// ---- SIMD availability (compile-time) ----
#if defined(KVANN_ARCH_X86)
  #if defined(__AVX2__)
    #define KVANN_HAVE_AVX2 1
  #endif
  #if defined(__FMA__)
    #define KVANN_HAVE_FMA 1
  #endif
  // MSVC sets _M_IX86_FP / arch flags differently; AVX2 implies FMA on /arch:AVX2
  #if defined(_MSC_VER) && defined(__AVX2__) && !defined(KVANN_HAVE_FMA)
    #define KVANN_HAVE_FMA 1
  #endif
#endif

#if defined(KVANN_ARCH_AARCH64)
  // NEON is mandatory on aarch64.
  #define KVANN_HAVE_NEON 1
#elif defined(KVANN_ARCH_ARM) && defined(__ARM_NEON)
  #define KVANN_HAVE_NEON 1
#endif

// ---- Force-inline helper ----
#if defined(_MSC_VER)
#define KVANN_FORCE_INLINE __forceinline
#else
#define KVANN_FORCE_INLINE inline __attribute__((always_inline))
#endif

// ---- Likely / unlikely ----
#if defined(__GNUC__) || defined(__clang__)
#define KVANN_LIKELY(x)   __builtin_expect(!!(x), 1)
#define KVANN_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define KVANN_LIKELY(x)   (x)
#define KVANN_UNLIKELY(x) (x)
#endif

// ---- Restrict ----
#if defined(_MSC_VER)
#define KVANN_RESTRICT __restrict
#else
#define KVANN_RESTRICT __restrict__
#endif
