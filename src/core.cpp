#include <kvann/core.h>
#include <kvann/detail/simd.h>

#include <cmath>

namespace kvann {

float cosine_similarity(const float* a, const float* b, std::size_t dim) {
    return simd::dot_f32(a, b, dim);
}

void normalize_vector(float* vec, std::size_t dim) {
    simd::normalize_f32(vec, dim);
}

bool is_normalized(const float* vec, std::size_t dim, float eps) {
    float nrm2 = simd::dot_f32(vec, vec, dim);
    return std::fabs(nrm2 - 1.0f) < eps;
}

const char* simd_backend() {
    return simd::backend_name();
}

} // namespace kvann
