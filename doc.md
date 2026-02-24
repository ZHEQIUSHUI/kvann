动态向量检索引擎完整需求文档（Full Spec）

> 项目名称：kvann
语言：C++
设计哲学：KV-first，ANN-second

说明：V1/V2/V3 为演进里程碑，不代表三套独立实现；代码层面仅保留单一版本 API，按里程碑逐步增强能力。




---

1. 项目终极目标（Ultimate Goal）

构建一个 工业级、可动态更新的向量检索引擎，满足：

像 KV 存储一样支持 高频 put / del / update

支持 百万 → 亿级向量

支持 CPU / GPU

支持 量化、分层索引

支持 在线查询 + 后台重建

支持 低延迟 + 高吞吐

可作为 基础库 / SDK 嵌入其他系统



---

2. 核心设计不变式（所有版本都必须满足）

这些是绝对不允许破坏的原则：

1. KV 是唯一真相

索引永远只是加速结构

搜索结果必须通过 KV 校验



2. 删除 / 更新必须立即生效

不允许返回已删除 / 旧版本



3. 索引允许滞后，语义不允许错误


4. 最终排序必须使用统一、精确的相似度


5. 任何 ANN 结构都允许被整体丢弃并重建




---

3. 统一术语定义（供模型理解）

术语	含义

Key	外部唯一 ID（用户可见）
Slot	内部连续编号（索引/存储用）
Base	已 build、只读索引层
Delta	新增/更新的可写层
Tombstone	逻辑删除标记
Rebuild	从 KV 真相重建索引
Recall	召回阶段（ANN）
Rerank	精排阶段（精确距离）



---

4. API 总体设计（跨所有版本）

class Index {
public:
  bool put(Key key, float* vector);
  bool del(Key key);
  bool exists(Key key) const;

  SearchResult search(float* query, int topk);

  void rebuild();
  IndexStats stats() const;

  void save(const std::string& path);
  static Index load(const std::string& path);
};


---

======================

V1：最小可用版本（MVP）

======================

V1 目标

正确

可维护

易实现

动态语义完整


V1 明确包含

KV + slot 映射

Base / Delta 双层

HNSW（CPU）

brute-force delta

cosine similarity（normalize + dot）

tombstone delete

手动 rebuild

多线程查询


V1 明确不包含

GPU

PQ / IVF

分布式

WAL / 崩溃恢复

lock-free


（⚠️ V1 详细实现略，此处默认你使用上一条给你的 V1 文档）


---

======================

V2：性能与规模增强版

======================

> V2 不改变语义，只改变 性能上限和可扩展性



V2 新增目标

2.1 向量量化（CPU）

支持的量化格式

FP16（必须）

INT8 per-vector scale（必须）

预留 PQ 接口（不实现）


规则

rerank 永远使用 统一精度（fp32 accumulate）

ANN 层允许用近似距离



---

2.2 SIMD 优化

必须支持

AVX2 dot

F16C（fp16 → fp32）

向量维度假定为 128 的倍数（但不能写死）


内存布局

向量连续存储

64B / 128B 对齐

SoA / blocked layout 优先



---

2.3 Delta 升级为可写 ANN

规则

delta_size ≤ N：brute-force

delta_size > N：自动 build delta HNSW

delta HNSW 允许 tombstone



---

2.4 自动 rebuild 策略

rebuild 触发条件

tombstone_ratio > T

delta_ratio > D

查询延迟显著升高（可选）


rebuild 要求

rebuild 期间：

查询不中断

新写入仍进入 delta


rebuild 后：

原子切换 base

清空 delta




---

2.5 持久化（基础）

支持

save/load

mmap-friendly 文件布局

只保证冷启动恢复，不保证 crash-safe



---

======================

V3：GPU & 大规模版本

======================

> V3 是 吞吐和规模 的质变版本




---

3.1 GPU 支持（CUDA）

GPU Phase A：Brute-force

GPU 上存放量化向量（INT8 / FP16）

GPU 计算 dot / L2

topK 返回 CPU rerank


GPU Phase B：IVF

CPU 训练 coarse centroids

GPU inverted lists

nprobe 可调



---

3.2 Product Quantization（PQ）

支持

OPQ + PQ

ADC 查询

GPU / CPU 双实现


规则

PQ 只用于召回

rerank 必须回原向量或 fp16



---

3.3 多层索引（Hierarchical Index）

Query
 ├── Delta (CPU, HNSW)
 ├── Base (CPU, HNSW)
 └── Cold (GPU, IVF-PQ)

合并三层候选

统一 rerank



---

3.4 分布式（可选）

Sharding

按 key hash

每 shard 独立 base/delta


查询

fan-out → merge



---

3.5 强一致写语义（高级）

WAL

put/del 追加日志

崩溃恢复


Snapshot

consistent rebuild snapshot



---

3.6 Lock-free / RCU（高级）

查询路径完全无锁

写使用 epoch / RCU

rebuild 使用版本指针切换



---

======================

全局禁止行为（所有版本）

======================

❌ 直接 merge 不同 ANN 分数
❌ 删除时修改图结构
❌ slot 替代 key 作为外部语义
❌ rebuild 阻塞查询
❌ 用 PQ 结果直接返回给用户


---

======================

给模型的最终执行指令

======================

> 请严格按 V1 → V2 → V3 分阶段实现

不要在 V1 中实现 GPU / PQ

不要破坏 KV 语义

不要假设索引是强一致的

rebuild 是正常维护行为


任何时候，以“搜索语义正确性”优先于性能
