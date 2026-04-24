"""Smoke test for the kvann Python bindings.

Run after building with -DKVANN_BUILD_PYTHON=ON. Adjust PYTHONPATH:
    PYTHONPATH=build/python python3 bindings/python/test_kvann.py
"""

import sys
import numpy as np

import kvann


def main() -> int:
    print("simd backend:", kvann.simd_backend())

    cfg = kvann.IndexConfig()
    cfg.dim = 128
    cfg.max_elements = 1000

    idx = kvann.Index(cfg)

    rng = np.random.default_rng(42)
    n = 200
    keys = np.arange(n, dtype=np.uint64)
    vecs = rng.standard_normal((n, cfg.dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    idx.put_batch(keys, vecs)
    idx.rebuild()

    q = vecs[0]
    ks, ss, _ = idx.search(q, topk=5)
    print("top-5 for vec[0]:", list(zip(ks.tolist(), ss.tolist())))
    assert ks[0] == 0, "self should be top-1"

    # batch
    qs = vecs[:8]
    bks, bss = idx.search_batch(qs, topk=3)
    assert bks.shape == (8, 3)

    # payload
    cfg2 = kvann.IndexConfig()
    cfg2.dim = 4
    cfg2.max_elements = 10
    idx2 = kvann.Index(cfg2)
    v = np.array([1, 0, 0, 0], dtype=np.float32)
    idx2.put(7, v, payload=b"hello")
    p = idx2.get_payload(7)
    assert p == b"hello", f"got {p!r}"

    # save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".idx") as f:
        path = f.name
    try:
        idx.save(path)
        loaded = kvann.Index.load(path)
        ks2, ss2, _ = loaded.search(q, topk=1)
        assert int(ks2[0]) == int(ks[0])
    finally:
        os.remove(path)

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
