# kvann Architecture

> KV-first, ANN-second.

## Layered overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         Public API (Index)                          │
│  put / put_batch / del / search / search_batch / rebuild / save     │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │          Index::Impl           │
                  └───┬───────┬───────┬───────┬───┘
                      │       │       │       │
            ┌─────────▼┐ ┌────▼─┐ ┌───▼──┐ ┌──▼─────────┐
            │ KeyDir   │ │ Slot │ │ Vec  │ │ HnswGraph  │
            │ (sharded │ │ Key  │ │ Store│ │ (base +    │
            │ map)     │ │ Map  │ │      │ │   delta)   │
            └──────────┘ └──────┘ └──────┘ └────────────┘
```

| Component | Purpose | Threading |
|-----------|---------|-----------|
| `KeyDir`        | sharded `Key → {Slot, version, payload}` | per-stripe `shared_mutex` |
| `SlotKeyMap`    | dense `Slot → Key`, `kInvalidKey = dead`  | `vector<atomic<Key>>` lock-free |
| `VectorStore`   | aligned (64 B) blocks, slot-indexed       | per-slot **seqlock** (writer ↔ reader) |
| `HnswGraph`     | layer-0 flat arena + sparse upper layers  | per-slot mutex pool, lock-free reads |
| `DeltaSet`      | atomic alive bitmap + member list         | bitmap atomic, list under one mutex |
| Visited buffer  | per-thread epoch-tagged `vector<uint32_t>`| `thread_local`, no sync needed |

## Read path

```
search(query)
  │
  ▼
1. normalize query  (SIMD)
2. base_graph.search(ef)            ─── shared_lock(base_swap_mutex)
3. delta path
   ├─ if delta_size > bruteforce_limit and delta_hnsw_active
   │       delta_graph.search(ef)
   └─ else  brute-force scan over delta members
4. dedupe by slot
5. rerank with exact cosine (storage_.dot_with — seqlock-safe)
6. apply user filter
7. partial_sort to top-k
8. (optional) attach payloads
```

The HNSW search itself is lock-free for layer-0 (atomic degrees + arena cells)
and takes a per-slot mutex briefly to copy upper-layer neighbor lists.

## Write path

```
put(key, vec, payload)
  │
  ├─ KeyDir.with_write(key) under stripe write_lock:
  │     allocate slot if new (next_slot_++ atomic)
  │     update payload + version
  │
  ├─ storage_.set_vector(slot, vec)            (seqlock write)
  ├─ slot_key_.set(slot, key)                  (atomic store)
  ├─ delta_.mark_alive(slot)                   (bitmap + member set)
  └─ if delta_size > delta_hnsw_threshold:
        promote: backfill all delta members to delta_graph_  (one-time)
     else if delta_hnsw_active:
        delta_graph_.add(slot, vec)
```

Deletes are tombstones: `slot_key_.clear(slot)` + erase from `KeyDir`.
Search rerank skips slots whose `slot_key_.get(slot) == kInvalidKey`.

## Rebuild

```
rebuild()
  │
  ├─ snapshot KeyDir under stripe read_locks:
  │     copy (key, slot) list AND copy each vector via seqlock
  ├─ build new HnswGraph from snapshot
  │     (single-threaded today — parallel add is a follow-up)
  ├─ unique_lock(base_swap_mutex):
  │     base_graph_ = std::move(new_base)
  └─ drain delta members that landed in base; reset delta_graph_
```

Concurrent puts/deletes during rebuild keep flowing into the live storage and
delta. The new base is built against frozen vector data, so search continues
to return correct results throughout — at worst slightly stale candidates,
which the exact-cosine rerank corrects.

## Persistence (file format v3)

```
+----------------------------------+
|  Header (32 B)                   |
|    magic "KVANN03\0"             |
|    fmt_version u32 (=3)          |
|    flags u32 (bit 0 = has_hnsw)  |
|    num_sections u32              |
|    reserved u32                  |
|    reserved u64                  |
+----------------------------------+
|  Section table (8 × 32 B)        |
|    id u32, _, offset u64,         |
|    length u64, crc32 u32, _      |
+----------------------------------+
|  Section 1: META                 |
|    dim, max_elements,            |
|    storage_block_size,           |
|    next_slot, hnsw_M, M_max0,    |
|    ef_construction               |
+----------------------------------+
|  Section 2: KEYS                 |
|    n + (key, slot, version,      |
|         payload_len, payload)*n  |
+----------------------------------+
|  Section 3: VECTORS              |
|    dim, n, float[n × dim]        |
+----------------------------------+
|  Section 4: HNSW_GRAPH (opt)     |
|    enterpoint, max_layer, size,  |
|    n_nodes + per-node arena dump |
+----------------------------------+
```

Each section is checksummed with CRC32 (IEEE 802.3 reflected). When the HNSW
section is present, `Index::load` restores the graph directly into the arena
— no rebuild needed.

## Cross-platform

| Concern | Linux/macOS | Windows |
|---------|-------------|---------|
| Aligned alloc | `posix_memalign` | `_aligned_malloc` |
| RW lock | `std::shared_mutex` | `std::shared_mutex` |
| Threads | `std::thread` | `std::thread` |
| SIMD x86 | `-mavx2 -mfma` | `/arch:AVX2` |
| SIMD aarch64 | NEON intrinsics | n/a |

Compile-time arch detection lives in `include/kvann/detail/arch.h`. The SIMD
backend (`include/kvann/detail/simd.h`) selects an inline implementation at
compile time and reports it via `kvann::simd_backend()`.

## Concurrency invariants

1. `slot_key_[slot]` is the single source of truth for liveness; once set to
   `kInvalidKey`, the slot is dead until re-`put`.
2. `VectorStore::set_vector` is the only writer; readers either see stable
   data (even seqlock counter) or retry.
3. `base_graph_` may only be replaced under `base_swap_mutex_` unique lock.
4. `delta_graph_.clear()` and `delta_hnsw_active_` flips are protected by
   `delta_hnsw_build_mutex_`.

## Known limitations / roadmap

- HNSW neighbor selection is "first-M closest"; the proper distance-based
  diversity heuristic is a follow-up (~1–2 % recall improvement).
- Slot compaction during rebuild not yet implemented (tombstone slots are not
  reclaimed in storage).
- Parallel HNSW add during rebuild is single-threaded today.
- mmap-based load not yet implemented (always read into RAM).
- WAL / crash-safe writes not yet implemented.
