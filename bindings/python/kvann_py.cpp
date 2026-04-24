// Python bindings for kvann. Numpy float32 zero-copy in/out.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <kvann/core.h>
#include <kvann/index.h>

#include <stdexcept>

namespace py = pybind11;

namespace {

void check_status(const kvann::Status& s) {
    if (!s.ok()) {
        throw std::runtime_error(std::string("kvann ") + s.code_str() + ": " + s.message());
    }
}

const float* require_vec(py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                         std::size_t expected_dim, const char* what) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(what) + ": expected 1-D float32 array");
    }
    if (static_cast<std::size_t>(arr.shape(0)) != expected_dim) {
        throw std::invalid_argument(std::string(what) + ": dim mismatch");
    }
    return arr.data();
}

const float* require_2d(py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                        std::size_t expected_dim, std::size_t* n_out, const char* what) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(std::string(what) + ": expected 2-D float32 array");
    }
    if (static_cast<std::size_t>(arr.shape(1)) != expected_dim) {
        throw std::invalid_argument(std::string(what) + ": dim mismatch");
    }
    *n_out = static_cast<std::size_t>(arr.shape(0));
    return arr.data();
}

} // namespace

PYBIND11_MODULE(_kvann, m) {
    m.doc() = "kvann — KV-first dynamic vector index (C++ core)";

    m.def("simd_backend", &kvann::simd_backend,
          "Active SIMD backend: 'avx2', 'neon' or 'scalar'.");

    py::enum_<kvann::StatusCode>(m, "StatusCode")
        .value("Ok",              kvann::StatusCode::kOk)
        .value("NotFound",        kvann::StatusCode::kNotFound)
        .value("AlreadyExists",   kvann::StatusCode::kAlreadyExists)
        .value("DimMismatch",     kvann::StatusCode::kDimMismatch)
        .value("Full",            kvann::StatusCode::kFull)
        .value("Io",              kvann::StatusCode::kIo)
        .value("InvalidArgument", kvann::StatusCode::kInvalidArgument)
        .value("Unsupported",     kvann::StatusCode::kUnsupported)
        .value("Internal",        kvann::StatusCode::kInternal);

    py::class_<kvann::IndexConfig>(m, "IndexConfig")
        .def(py::init<>())
        .def_readwrite("dim",                       &kvann::IndexConfig::dim)
        .def_readwrite("max_elements",              &kvann::IndexConfig::max_elements)
        .def_readwrite("hnsw_M",                    &kvann::IndexConfig::hnsw_M)
        .def_readwrite("hnsw_M_max0",               &kvann::IndexConfig::hnsw_M_max0)
        .def_readwrite("hnsw_ef_construction",      &kvann::IndexConfig::hnsw_ef_construction)
        .def_readwrite("hnsw_ef_search",            &kvann::IndexConfig::hnsw_ef_search)
        .def_readwrite("delta_bruteforce_limit",    &kvann::IndexConfig::delta_bruteforce_limit)
        .def_readwrite("delta_hnsw_threshold",      &kvann::IndexConfig::delta_hnsw_threshold)
        .def_readwrite("storage_block_size",        &kvann::IndexConfig::storage_block_size)
        .def_readwrite("lock_stripes",              &kvann::IndexConfig::lock_stripes);

    py::class_<kvann::IndexStats>(m, "IndexStats")
        .def_readonly("dim",             &kvann::IndexStats::dim)
        .def_readonly("total_keys",      &kvann::IndexStats::total_keys)
        .def_readonly("live_keys",       &kvann::IndexStats::live_keys)
        .def_readonly("tombstone_count", &kvann::IndexStats::tombstone_count)
        .def_readonly("base_count",      &kvann::IndexStats::base_count)
        .def_readonly("delta_count",     &kvann::IndexStats::delta_count)
        .def_readonly("tombstone_ratio", &kvann::IndexStats::tombstone_ratio)
        .def_readonly("delta_ratio",     &kvann::IndexStats::delta_ratio)
        .def_property_readonly("simd_backend",
            [](const kvann::IndexStats& s) { return std::string(s.simd_backend); });

    py::class_<kvann::Index>(m, "Index")
        .def(py::init<const kvann::IndexConfig&>(), py::arg("config"))

        .def("put",
             [](kvann::Index& self, kvann::Key key,
                py::array_t<float, py::array::c_style | py::array::forcecast> vec,
                py::object payload) {
                 const float* p = require_vec(vec, self.config().dim, "put.vec");
                 if (payload.is_none()) {
                     check_status(self.put(key, p));
                 } else {
                     std::string b = py::cast<std::string>(payload);
                     check_status(self.put(key, p, b.data(), b.size()));
                 }
             },
             py::arg("key"), py::arg("vec"), py::arg("payload") = py::none())

        .def("put_batch",
             [](kvann::Index& self,
                py::array_t<kvann::Key, py::array::c_style | py::array::forcecast> keys,
                py::array_t<float, py::array::c_style | py::array::forcecast> vecs) {
                 if (keys.ndim() != 1) throw std::invalid_argument("keys must be 1-D");
                 std::size_t n = 0;
                 const float* vp = require_2d(vecs, self.config().dim, &n, "put_batch.vecs");
                 if (static_cast<std::size_t>(keys.shape(0)) != n) {
                     throw std::invalid_argument("len(keys) != len(vecs)");
                 }
                 std::size_t err_idx = 0;
                 auto st = self.put_batch(keys.data(), vp, n, &err_idx);
                 if (!st.ok()) {
                     throw std::runtime_error("put_batch failed at index " +
                                              std::to_string(err_idx) + ": " +
                                              st.code_str());
                 }
             },
             py::arg("keys"), py::arg("vecs"))

        .def("delete",  [](kvann::Index& self, kvann::Key k) { check_status(self.del(k)); },
             py::arg("key"))
        .def("__contains__",
             [](const kvann::Index& self, kvann::Key k) { return self.exists(k); })
        .def("exists",
             [](const kvann::Index& self, kvann::Key k) { return self.exists(k); },
             py::arg("key"))

        .def("get_payload",
             [](const kvann::Index& self, kvann::Key k) -> py::object {
                 std::vector<uint8_t> out;
                 auto st = self.get_payload(k, out);
                 if (!st.ok()) return py::none();
                 return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
             },
             py::arg("key"))

        .def("search",
             [](const kvann::Index& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> query,
                int topk, int ef, bool include_payload) {
                 const float* p = require_vec(query, self.config().dim, "search.query");
                 kvann::SearchParams sp;
                 sp.topk            = topk;
                 sp.ef              = ef;
                 sp.include_payload = include_payload;
                 auto results = self.search(p, sp);

                 // Build numpy arrays for keys + scores.
                 py::array_t<kvann::Key> ks(results.size());
                 py::array_t<float>     ss(results.size());
                 auto* k = ks.mutable_data();
                 auto* s = ss.mutable_data();
                 py::list payloads;
                 for (std::size_t i = 0; i < results.size(); ++i) {
                     k[i] = results[i].key;
                     s[i] = results[i].score;
                     if (include_payload) {
                         payloads.append(py::bytes(
                             reinterpret_cast<const char*>(results[i].payload.data()),
                             results[i].payload.size()));
                     }
                 }
                 if (include_payload) {
                     return py::make_tuple(ks, ss, payloads);
                 }
                 return py::make_tuple(ks, ss, py::none());
             },
             py::arg("query"), py::arg("topk") = 10, py::arg("ef") = 0,
             py::arg("include_payload") = false)

        .def("search_batch",
             [](const kvann::Index& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                int topk, int ef) {
                 std::size_t n = 0;
                 const float* qp = require_2d(queries, self.config().dim, &n, "search_batch.queries");
                 kvann::SearchParams sp;
                 sp.topk = topk;
                 sp.ef   = ef;
                 auto out = self.search_batch(qp, n, sp);

                 // Pad to fixed (n, topk) — fill missing with sentinel.
                 py::array_t<kvann::Key> ks({(py::ssize_t)n, (py::ssize_t)topk});
                 py::array_t<float>     ss({(py::ssize_t)n, (py::ssize_t)topk});
                 auto* k = ks.mutable_data();
                 auto* s = ss.mutable_data();
                 for (std::size_t i = 0; i < n; ++i) {
                     for (int j = 0; j < topk; ++j) {
                         std::size_t off = i * topk + j;
                         if (j < (int)out[i].size()) {
                             k[off] = out[i][j].key;
                             s[off] = out[i][j].score;
                         } else {
                             k[off] = kvann::kInvalidKey;
                             s[off] = -1.0f;
                         }
                     }
                 }
                 return py::make_tuple(ks, ss);
             },
             py::arg("queries"), py::arg("topk") = 10, py::arg("ef") = 0)

        .def("rebuild",  [](kvann::Index& self) { check_status(self.rebuild()); })
        .def("rebuild_async",
             [](kvann::Index& self) { check_status(self.rebuild_async()); })
        .def("wait_rebuild", &kvann::Index::wait_rebuild)

        .def("save", [](const kvann::Index& self, const std::string& path) {
            check_status(self.save(path));
        }, py::arg("path"))

        .def_static("load", &kvann::Index::load, py::arg("path"))

        .def("stats",  &kvann::Index::stats)
        .def_property_readonly("dim",
            [](const kvann::Index& self) { return self.config().dim; });
}
