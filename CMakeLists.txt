cmake_minimum_required(VERSION 3.14)
project(kvann VERSION 0.2.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standard / output layout
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(KVANN_BUILD_TESTS    "Build unit tests"            ON)
option(KVANN_BUILD_EXAMPLES "Build example programs"      ON)
option(KVANN_BUILD_BENCH    "Build micro-benchmarks"      OFF)
option(KVANN_BUILD_PYTHON   "Build Python bindings (pybind11)" OFF)
option(KVANN_ENABLE_AVX2    "Enable AVX2 (+FMA) on x86_64"            ON)
option(KVANN_ENABLE_NEON    "Enable NEON on aarch64 (always-on)"      ON)
option(KVANN_ENABLE_LTO     "Enable link-time optimization in Release" OFF)
option(BUILD_SHARED_LIBS    "Build shared libraries"                  OFF)

# ---------------------------------------------------------------------------
# Detect target architecture
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(amd64)|(i.86)")
    set(KVANN_TARGET_X86 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)|(ARM64)")
    set(KVANN_TARGET_AARCH64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm)|(ARM)")
    set(KVANN_TARGET_ARM TRUE)
endif()

# ---------------------------------------------------------------------------
# Compile flags (applied via function so they don't leak into install export)
# ---------------------------------------------------------------------------
function(kvann_apply_compile_flags target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4 /permissive- /Zc:__cplusplus /EHsc /utf-8
            $<$<CONFIG:Release>:/O2>
            /wd4100
        )
        target_compile_definitions(${target} PRIVATE
            _CRT_SECURE_NO_WARNINGS
            NOMINMAX
            WIN32_LEAN_AND_MEAN
        )
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic
            -Wno-unused-parameter
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g>
            $<$<CONFIG:Debug>:-O0 -g>
        )
    endif()

    if(KVANN_TARGET_X86 AND KVANN_ENABLE_AVX2)
        if(MSVC)
            target_compile_options(${target} PRIVATE /arch:AVX2)
        else()
            target_compile_options(${target} PRIVATE -mavx2 -mfma -mf16c)
        endif()
    endif()
endfunction()

if(KVANN_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg)
    if(_ipo_ok)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    else()
        message(WARNING "LTO requested but not supported: ${_ipo_msg}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

add_library(kvann
    src/core.cpp
    src/index.cpp
    src/kvann.cpp
)
add_library(kvann::kvann ALIAS kvann)

target_include_directories(kvann
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(kvann PUBLIC Threads::Threads)
kvann_apply_compile_flags(kvann)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(KVANN_BUILD_TESTS)
    enable_testing()

    foreach(_t test_v1 test_user_data test_opt)
        if(_t STREQUAL "test_v1")
            add_executable(${_t} tests/test_v1.cpp)
        elseif(_t STREQUAL "test_user_data")
            add_executable(${_t} tests/test_user_data.cpp)
        else()
            add_executable(${_t} tests/test_opt.cpp)
        endif()
        target_link_libraries(${_t} PRIVATE kvann)
        kvann_apply_compile_flags(${_t})
        add_test(NAME ${_t} COMMAND ${_t})
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(KVANN_BUILD_EXAMPLES)
    add_executable(example example.cpp)
    target_link_libraries(example PRIVATE kvann)
    kvann_apply_compile_flags(example)

    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(KVANN_BUILD_BENCH)
    add_executable(kvann_bench benchmarks/bench_main.cpp)
    target_link_libraries(kvann_bench PRIVATE kvann)
    kvann_apply_compile_flags(kvann_bench)
endif()

# ---------------------------------------------------------------------------
# Python bindings (opt-in)
# ---------------------------------------------------------------------------
if(KVANN_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS kvann
    EXPORT kvannTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/kvann DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT kvannTargets
    FILE kvannTargets.cmake
    NAMESPACE kvann::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kvann
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kvannConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/kvannConfig.cmake"
"include(CMakeFindDependencyMacro)
find_dependency(Threads)
include(\"\${CMAKE_CURRENT_LIST_DIR}/kvannTargets.cmake\")
")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kvannConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/kvannConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kvann
)
