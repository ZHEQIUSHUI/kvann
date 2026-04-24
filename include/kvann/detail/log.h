// kvann internal logging stub.
//
// M0: no-op by default. M4 will wire this to an injectable logger
// (IndexConfig::log_sink) without changing call sites.
#pragma once

#ifndef KVANN_LOG_ENABLED
#define KVANN_LOG_ENABLED 0
#endif

#if KVANN_LOG_ENABLED
#include <cstdio>
#define KVANN_LOG(level, ...)                                                  \
    do {                                                                       \
        std::fprintf(stderr, "[kvann][" level "] ");                           \
        std::fprintf(stderr, __VA_ARGS__);                                     \
        std::fprintf(stderr, "\n");                                            \
    } while (0)
#else
#define KVANN_LOG(level, ...) ((void)0)
#endif

#define KVANN_LOG_INFO(...)  KVANN_LOG("info",  __VA_ARGS__)
#define KVANN_LOG_WARN(...)  KVANN_LOG("warn",  __VA_ARGS__)
#define KVANN_LOG_ERROR(...) KVANN_LOG("error", __VA_ARGS__)
