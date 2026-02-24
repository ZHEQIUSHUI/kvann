cmake_minimum_required(VERSION 3.14)
project(kvann VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_AVX2 "Enable AVX2 optimizations" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# 设置编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

if(ENABLE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c")
    add_definitions(-DENABLE_AVX2)
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(Threads REQUIRED)

# 创建库目标（非 header-only）
add_library(kvann
    src/core.cpp
    src/index.cpp
    src/kvann.cpp
)
target_include_directories(kvann PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(kvann PUBLIC Threads::Threads)

# 测试
if(BUILD_TESTS)
    enable_testing()
    
    # V1 测试
    add_executable(test_v1 tests/test_v1.cpp)
    target_link_libraries(test_v1 kvann)
    add_test(NAME v1_test COMMAND test_v1)
    
    # user_data 测试
    add_executable(test_user_data tests/test_user_data.cpp)
    target_link_libraries(test_user_data kvann)
    add_test(NAME user_data_test COMMAND test_user_data)
endif()

# 示例程序
add_executable(example example.cpp)
target_link_libraries(example kvann)

# 安装
install(TARGETS kvann
    EXPORT kvannTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/kvann DESTINATION include)
install(EXPORT kvannTargets
    FILE kvannTargets.cmake
    NAMESPACE kvann::
    DESTINATION lib/cmake/kvann
)
