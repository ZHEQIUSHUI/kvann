# kvann

> 工业级动态向量检索引擎 — KV-first, ANN-second.

C++17，零外部依赖，跨 Linux / Windows × x86_64 / aarch64。

## 特性

- **KV 是真相**：HNSW 仅作召回加速，最终排序用精确余弦 rerank
- **Base / Delta 双层**：base 是已 build 的只读图，delta 是可写层（自动在 brute-force 与 HNSW 之间切换）
- **Tombstone 删除**：立即生效，搜索永不返回已删 key
- **手动 + 异步 rebuild**：snapshot 抓 (key, slot, vec) 后台构建新图，原子 swap，主路径不阻塞
- **Seqlock 写入**：并发 put 不会让搜索看到撕裂的向量
- **SIMD**：x86 AVX2+FMA / aarch64 NEON / 标量 fallback，编译期 dispatch
- **批量 API**：`put_batch` / `search_batch`
- **Payload**：每个 key 可挂任意二进制 user_data，搜索时按需返回
- **Status 错误模型**：可读错误码 + 消息，调用方不用 try-catch
- **可插拔 logger**：`IndexConfig::log_sink`

## 快速开始

```cpp
#include <kvann/core.h>
#include <kvann/index.h>

kvann::IndexConfig cfg;
cfg.dim          = 128;
cfg.max_elements = 1'000'000;
kvann::Index index(cfg);

// 插入
index.put(/*key=*/42, vec.data());

// 带 payload 插入
index.put(43, vec.data(), payload.data(), payload.size());

// 搜索
kvann::SearchParams sp;
sp.topk            = 10;
sp.include_payload = true;
sp.filter          = [](kvann::Key k) { return k % 2 == 0; };

auto results = index.search(query.data(), sp);
for (const auto& r : results) {
    std::cout << "key=" << r.key << " score=" << r.score
              << " payload_len=" << r.payload.size() << "\n";
}

// rebuild
index.rebuild();          // 同步
index.rebuild_async();    // 异步触发
index.wait_rebuild();

// 持久化
index.save("/tmp/idx.bin");
auto loaded = kvann::Index::load("/tmp/idx.bin");
```

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure
```

CMake options:

| Option              | Default | 说明 |
|---------------------|---------|------|
| `KVANN_BUILD_TESTS`    | ON      | 单元测试 |
| `KVANN_BUILD_EXAMPLES` | ON      | 示例程序 |
| `KVANN_ENABLE_AVX2`    | ON      | x86_64 上启用 AVX2+FMA |
| `KVANN_ENABLE_LTO`     | OFF     | Release 启用 LTO |
| `BUILD_SHARED_LIBS`    | OFF     | 编共享库 |

跨平台说明：

- **Linux / macOS / Windows**：CMake 探测平台、对齐分配自动切换 `posix_memalign` ↔ `_aligned_malloc`
- **x86_64**：AVX2+FMA 默认开（运行时由 SIMD 后端自动选择）
- **aarch64**：NEON（mandatory on aarch64，无需额外配置）
- **其他**：标量 fallback

`kvann::simd_backend()` 返回 `"avx2" / "neon" / "scalar"` 用于诊断。

## 集成

```cmake
find_package(kvann CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE kvann::kvann)
```

## API 概览

| 方法 | 说明 |
|------|------|
| `put(key, vec)` | 插入/更新 |
| `put(key, vec, payload, len)` | 带 payload 插入 |
| `put_batch(keys, vecs, n)` | 批量插入 |
| `del(key)` | 逻辑删除 |
| `exists(key)` | 是否存在 |
| `get_payload(key, out)` | 按 key 取 payload |
| `search(query, params)` | 单 query 搜索 |
| `search_batch(queries, n, params)` | 批量搜索 |
| `rebuild()` / `rebuild_async()` | 重建 base 图 |
| `wait_rebuild()` | 等待异步重建结束 |
| `save(path)` / `load(path)` | 持久化 |
| `stats()` | 统计快照 |
| `config()` | 当前配置 |

## 性能（参考，128-d 单机 Linux x86_64 + AVX2）

| 操作 | 量 | 时间 |
|------|----|------|
| Insert | 1k | ~4 ms |
| Rebuild | 1k | ~190 ms |
| Concurrent search (4 threads × 100q) | 400 q | ~17 ms |
| Search | 100 q | ~15 ms |
| Recall@10 | 1k base | 0.97+ |

## 设计原则

1. **KV 是唯一真相** — 索引永远只是加速结构
2. **删除 / 更新立即生效** — 不允许返回已删除 / 旧版本
3. **索引允许滞后，语义不允许错误**
4. **最终排序用统一精确相似度**
5. **任何 ANN 结构都允许整体丢弃并重建**

## 当前限制（Roadmap）

- 邻居选择仅 top-M（HNSW heuristic 多样性裁剪 — 跟 base recall 略有关）
- HNSW 图持久化为重建（load 后会重新 build base）
- 跨进程 mmap 加载未做
- WAL / crash-safe 未做
- 暂不支持量化 / GPU
