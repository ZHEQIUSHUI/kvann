add_executable(ex_quickstart       01_quickstart.cpp)
add_executable(ex_payload_filter   02_payload_filter.cpp)
add_executable(ex_persistence      03_persistence.cpp)
add_executable(ex_concurrent       04_concurrent.cpp)

foreach(_t ex_quickstart ex_payload_filter ex_persistence ex_concurrent)
    target_link_libraries(${_t} PRIVATE kvann)
    kvann_apply_compile_flags(${_t})
endforeach()
