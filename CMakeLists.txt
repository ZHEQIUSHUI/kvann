cmake_minimum_required(VERSION 3.14)
project(kvann VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_AVX2 "Enable AVX2 optimizations" OFF)

# 设置编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

if(ENABLE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c")
    add_definitions(-DENABLE_AVX2)
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 创建接口库（header-only）
add_library(kvann INTERFACE)
target_include_directories(kvann INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 测试
if(BUILD_TESTS)
    enable_testing()
    
    # V1 测试
    add_executable(test_v1 tests/test_v1.cpp)
    target_link_libraries(test_v1 kvann pthread)
    add_test(NAME v1_test COMMAND test_v1)
    
    # V2 测试
    add_executable(test_v2 tests/test_v2.cpp)
    target_link_libraries(test_v2 kvann pthread)
    add_test(NAME v2_test COMMAND test_v2)
    
    # user_data 测试
    add_executable(test_user_data tests/test_user_data.cpp)
    target_link_libraries(test_user_data kvann pthread)
    add_test(NAME user_data_test COMMAND test_user_data)
    
    # V3 测试
    add_executable(test_v3 tests/test_v3.cpp)
    target_link_libraries(test_v3 kvann pthread)
    add_test(NAME v3_test COMMAND test_v3)
endif()

# 示例程序
add_executable(example example.cpp)
target_link_libraries(example kvann pthread)

# 安装
install(DIRECTORY include/kvann DESTINATION include)
