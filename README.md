# kvann - 动态向量检索引擎

## 项目简介

kvann 是一个工业级、可动态更新的向量检索引擎，采用 **KV-first, ANN-second** 设计哲学。

## 核心特性

### V1 版本（已完成）
- **KV + Slot 映射**：Key 是外部唯一 ID，Slot 是内部连续编号
- **Base/Delta 双层架构**：
  - Base: 已 build 的只读 HNSW 索引
  - Delta: 新增/更新的可变层，使用 brute-force
- **HNSW (CPU)**：高效的近似最近邻搜索
- **Cosine Similarity**：归一化向量 + 点积
- **Tombstone 删除**：逻辑删除，支持立即生效
- **手动 Rebuild**：将 Delta 层合并到 Base 层
- **多线程查询**：读共享锁，写互斥锁
- **持久化**：支持 save/load 冷启动恢复
- **user_data 支持**：每个向量可附带自定义数据（文档、标签、元数据等）

### V2 版本（已完成）
- **向量量化**：
  - FP16（半精度浮点）
  - INT8（8位整型 + per-vector scale）
- **SIMD 优化**：
  - AVX2 FP32 点积
  - F16C FP16 <-> FP32 转换
  - AVX2 INT8 点积
- **自动 Rebuild 策略**：
  - tombstone_ratio > threshold
  - delta_ratio > threshold
- **Delta 层 HNSW**：当 Delta 层变大时自动启用

### V3 版本（已完成）
- **GPU 支持（CUDA 接口）**：
  - GPU Brute-force 索引
  - GPU IVF (Inverted File) 索引
  - CPU 回退实现（无 CUDA 环境可用）
- **Product Quantization (PQ)**：
  - OPQ + PQ
  - ADC (Asymmetric Distance Computation) 查询
  - 典型压缩比 64x (128维FP32 -> 8字节)
- **多层索引架构**：
  - Delta (热数据): CPU HNSW，可写
  - Base (温数据): CPU HNSW，只读
  - Cold (冷数据): GPU IVF-PQ，近似检索
- **分布式支持**：
  - 按 Key hash 分片
  - Fan-out / Merge 查询
  - 并行重建

## 核心设计不变式

1. **KV 是唯一真相** - 索引永远只是加速结构
2. **删除/更新必须立即生效** - 不允许返回已删除/旧版本
3. **索引允许滞后，语义不允许错误**
4. **最终排序必须使用统一、精确的相似度**
5. **任何 ANN 结构都允许被整体丢弃并重建**

## 快速开始

### 基础用法

```cpp
#include <kvann/index.h>

// 创建索引
kvann::Index index(128, 100000);  // 128维，最大10万向量

// 插入向量
std::vector<float> vec(128);
// ... 填充向量数据 ...
index.put(1, vec.data());

// 搜索
auto results = index.search(query_vec.data(), 10);
for (const auto& r : results) {
    std::cout << "key=" << r.key << " score=" << r.score << std::endl;
}

// 重建索引（将 Delta 层合并到 Base 层）
index.rebuild();
index.wait_rebuild();
```

### 带 user_data 的向量

```cpp
// 插入向量时附带自定义数据
std::string metadata = "document_123:这是一篇文档";
index.put_with_data(1, vec.data(), metadata.c_str(), metadata.size() + 1);

// 搜索结果包含 user_data
auto results = index.search(query_vec.data(), 10);
for (const auto& r : results) {
    std::string data(reinterpret_cast<const char*>(r.user_data.data()));
    std::cout << "key=" << r.key << " data=" << data << std::endl;
}
```

### V3 多层索引

```cpp
#include <kvann/index_v3.h>

// 配置分层策略
kvann::TieringPolicy tiering;
tiering.hot_threshold = 1000;    // Delta层最多1000个
tiering.warm_threshold = 10000;  // Base层最多10000个
tiering.enable_gpu = false;
tiering.enable_pq = true;

kvann::IndexV3 index(128, 100000, kvann::QuantizeType::FP32, 
                     kvann::AutoRebuildPolicy(), tiering);

// 分层重建
index.rebuild_with_tiering();

// 查看分层统计
auto stats = index.tier_stats();
std::cout << "Delta: " << stats.delta_count << std::endl;
std::cout << "Base: " << stats.base_count << std::endl;
std::cout << "Cold: " << stats.cold_count << std::endl;
```

### 分布式索引

```cpp
#include <kvann/distributed.h>

// 创建4分片的分布式索引
kvann::DistributedIndex dist_index(128, 4);

// 插入数据（自动分片）
dist_index.put(1, vec.data());

// 重建所有分片
dist_index.rebuild_all();
dist_index.wait_all_rebuilds();

// 搜索（Fan-out/Merge）
auto results = dist_index.search(query_vec.data(), 10);
```

## API 参考

### 核心类

| 类 | 说明 | 适用场景 |
|-----|------|---------|
| `Index` | V1 基础索引 | 单机中等规模数据 |
| `IndexV2` | V2 量化索引 | 需要压缩/量化 |
| `IndexV3` | V3 多层索引 | 大规模分层数据 |
| `DistributedIndex` | 分布式索引 | 超大规模数据 |

### 核心方法

```cpp
// 插入/更新
bool put(Key key, const float* vector);
bool put_with_data(Key key, const float* vector, const void* user_data, size_t len);

// 删除
bool del(Key key);

// 查询
bool exists(Key key);
std::vector<uint8_t> get_user_data(Key key);
std::vector<SearchResult> search(const float* query, int topk);

// 重建
void rebuild();           // V1/V2 标准重建
void rebuild_with_tiering();  // V3 分层重建
void wait_rebuild();

// 持久化
void save(const std::string& path);
static std::unique_ptr<Index> load(const std::string& path);

// 统计
IndexStats stats();
```

### SearchResult 结构

```cpp
struct SearchResult {
    Key key;                        // 向量唯一键
    float score;                    // 相似度分数（越高越相似）
    std::vector<uint8_t> user_data; // 用户自定义数据
};
```

## 构建和测试

```bash
mkdir build && cd build

# 基础构建
cmake .. -DBUILD_TESTS=ON
make -j4

# 启用 AVX2 优化
cmake .. -DBUILD_TESTS=ON -DENABLE_AVX2=ON
make -j4

# 运行测试
ctest --output-on-failure
# 或单独运行
./test_v1        # V1 功能测试
./test_user_data # user_data 功能测试
./test_v2        # V2 量化/SIMD 测试
./test_v3        # V3 GPU/PQ/分布式测试

# 运行示例
./example
```

## 项目结构

```
kvann/
├── include/kvann/
│   ├── core.h           # 核心数据结构（Key, Slot, VectorStorage, user_data）
│   ├── hnsw.h           # HNSW 索引实现
│   ├── index.h          # V1 主索引类
│   ├── quantization.h   # V2 量化支持（FP16, INT8, SIMD）
│   ├── index_v2.h       # V2 扩展类
│   ├── gpu.h            # V3 GPU/CUDA 接口
│   ├── pq.h             # V3 Product Quantization
│   ├── index_v3.h       # V3 多层索引
│   └── distributed.h    # V3 分布式支持
├── tests/
│   ├── test_v1.cpp      # V1 完整测试（12个用例）
│   ├── test_user_data.cpp # user_data 测试（5个用例）
│   ├── test_v2.cpp      # V2 功能测试（6个用例）
│   └── test_v3.cpp      # V3 功能测试（8个用例）
├── example.cpp          # 使用示例
└── CMakeLists.txt
```

## 性能数据

### V1 基准
- 插入 10,000 向量：~33ms
- 重建 10,000 向量：~8s
- 搜索（100次）：~130ms
- Recall@10：> 0.95（通常 1.0）

### V2 量化
- FP16 精度损失：< 0.001
- INT8 精度损失：< scale（典型值 0.001-0.01）
- AVX2 点积：与标量一致（误差 < 1e-7）

### V3 PQ
- 压缩比：64x（128维FP32 -> 8字节）
- 重建 RMSE：~0.07（典型值）

## 设计文档

详见项目根目录的 `doc.md`，包含完整的需求规格和版本规划。

## 扩展开发

### 添加新的量化类型

```cpp
// 在 quantization.h 中添加
enum class QuantizeType {
    FP32,
    FP16,
    INT8,
    YOUR_NEW_TYPE  // 添加新类型
};
```

### 实现 CUDA 内核

```cpp
// 在 gpu.h 中定义接口，在 .cu 文件中实现
#ifdef KVANN_ENABLE_CUDA
// CUDA 实现
#else
// CPU 回退实现
#endif
```
