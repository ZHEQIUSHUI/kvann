# kvann - 动态向量检索引擎

## 项目简介

kvann 是一个工业级、可动态更新的向量检索引擎，采用 **KV-first, ANN-second** 设计哲学。

## 核心特性

### 单一版本（已实现）
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

// 可选配置
kvann::IndexConfig cfg;
cfg.hnsw_ef_search = 128;
cfg.delta_hnsw_threshold = 5000;
kvann::Index tuned(128, cfg);

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

## API 参考

### 核心类

| 类 | 说明 | 适用场景 |
|-----|------|---------|
| `Index` | 基础索引 | 单机中等规模数据 |

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
void rebuild();
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
./test_v1        # 功能测试
./test_user_data # user_data 功能测试

# 运行示例
./example
```

## 项目结构

```
kvann/
├── include/kvann/
│   ├── core.h           # 公共类型与工具函数
│   ├── index.h          # 主索引类
│   └── version.h        # 版本接口
├── tests/
│   ├── test_v1.cpp      # 功能测试（12个用例）
│   └── test_user_data.cpp # user_data 测试（5个用例）
├── example.cpp          # 使用示例
└── CMakeLists.txt
```

## 性能数据

### 基准
- 插入 10,000 向量：~33ms
- 重建 10,000 向量：~8s
- 搜索（100次）：~130ms
- Recall@10：> 0.95（通常 1.0）

## 设计文档

详见项目根目录的 `doc.md`，包含完整的需求规格与设计约束。
### 本地序列化与冷启动

支持本地序列化（`save`）与加载（`load`），用于冷启动直接恢复索引，无需重新构建。

```cpp
// 保存到本地文件
index.save("/tmp/kvann.index");

// 冷启动加载
auto loaded = kvann::Index::load("/tmp/kvann.index");
```

注意：当前持久化面向冷启动恢复，不保证 crash-safe（无 WAL）。
