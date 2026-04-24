# Python bindings — pulled in via FetchContent so the core library stays
# zero-dep. Enable with -DKVANN_BUILD_PYTHON=ON.

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_kvann kvann_py.cpp)
target_link_libraries(_kvann PRIVATE kvann)
kvann_apply_compile_flags(_kvann)

set_target_properties(_kvann PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/kvann"
    OUTPUT_NAME "_kvann"
)

# Install __init__.py that re-exports the C extension.
file(WRITE "${CMAKE_BINARY_DIR}/python/kvann/__init__.py"
"\"kvann — Python wrapper around the C++ core (_kvann).\"
from ._kvann import (
    Index, IndexConfig, IndexStats, StatusCode, simd_backend,
)
__all__ = [\"Index\", \"IndexConfig\", \"IndexStats\", \"StatusCode\", \"simd_backend\"]
")
