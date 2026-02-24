#include <kvann/core.h>
#include <cmath>

namespace kvann {

float cosine_similarity(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

void normalize_vector(float* vec, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (size_t i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
    }
}

bool is_normalized(const float* vec, size_t dim, float eps) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    return std::abs(norm - 1.0f) < eps;
}

} // namespace kvann
