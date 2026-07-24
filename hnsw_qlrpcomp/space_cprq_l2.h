#pragma once

#include "cprq_package_cpu.h"
#include "hnswlib/hnswlib.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <immintrin.h>

namespace cprq_hnsw {

constexpr uint32_t kCprqNodeMagic = 0x4350524eu;   // "CPRN"
constexpr uint32_t kCprqQueryMagic = 0x43505251u;  // "CPRQ"

struct CprqQueryView {
    uint32_t magic = kCprqQueryMagic;
    uint32_t reserved = 0;
    const float* transformed_alpha = nullptr;
    const float* residual = nullptr;
    float residual_norm = 0.0f;
};

struct CprqDistanceContext {
    uint32_t dimensions = 0;
    uint32_t rank = 0;
    float residual_step = 0.0f;
    float residual_step_squared = 0.0f;
    size_t alpha_offset = 0;
    size_t beta_offset = 0;
    size_t norm_offset = 0;
    size_t bins_offset = 0;
    size_t node_size = 0;
};

#if defined(__AVX512F__)
inline float horizontal_sum_512(__m512 value) {
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, value);
    float result = 0.0f;
    for (float lane : lanes) {
        result += lane;
    }
    return result;
}

inline int64_t horizontal_sum_i32_512(__m512i value) {
    alignas(64) int32_t lanes[16];
    _mm512_store_si512(reinterpret_cast<__m512i*>(lanes), value);
    int64_t result = 0;
    for (int32_t lane : lanes) {
        result += lane;
    }
    return result;
}
#endif

#if defined(__AVX2__)
inline float horizontal_sum_256(__m256 value) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, value);
    float result = 0.0f;
    for (float lane : lanes) {
        result += lane;
    }
    return result;
}

inline int64_t horizontal_sum_i32_256(__m256i value) {
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), value);
    int64_t result = 0;
    for (int32_t lane : lanes) {
        result += lane;
    }
    return result;
}
#endif

inline float query_residual_dot_bins(
    const float* residual,
    const int8_t* bins,
    uint32_t dimensions) {
    uint32_t dim = 0;
    float result = 0.0f;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (; dim + 16 <= dimensions; dim += 16) {
        const __m128i packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(bins + dim));
        const __m512i widened = _mm512_cvtepi8_epi32(packed);
        const __m512 bin_values = _mm512_cvtepi32_ps(widened);
        const __m512 query_values = _mm512_loadu_ps(residual + dim);
        sum = _mm512_fmadd_ps(query_values, bin_values, sum);
    }
    result += horizontal_sum_512(sum);
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    for (; dim + 8 <= dimensions; dim += 8) {
        const __m128i packed =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bins + dim));
        const __m256i widened = _mm256_cvtepi8_epi32(packed);
        const __m256 bin_values = _mm256_cvtepi32_ps(widened);
        const __m256 query_values = _mm256_loadu_ps(residual + dim);
        sum = _mm256_fmadd_ps(query_values, bin_values, sum);
    }
    result += horizontal_sum_256(sum);
#endif
    for (; dim < dimensions; ++dim) {
        result += residual[dim] * static_cast<float>(bins[dim]);
    }
    return result;
}

inline uint64_t squared_bin_difference(
    const int8_t* first,
    const int8_t* second,
    uint32_t dimensions) {
    uint32_t dim = 0;
    int64_t result = 0;
#if defined(__AVX512F__)
    __m512i sum = _mm512_setzero_si512();
    for (; dim + 16 <= dimensions; dim += 16) {
        const __m128i first_packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(first + dim));
        const __m128i second_packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(second + dim));
        const __m512i first_values = _mm512_cvtepi8_epi32(first_packed);
        const __m512i second_values = _mm512_cvtepi8_epi32(second_packed);
        const __m512i difference = _mm512_sub_epi32(first_values, second_values);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(difference, difference));
    }
    result += horizontal_sum_i32_512(sum);
#elif defined(__AVX2__)
    __m256i sum = _mm256_setzero_si256();
    for (; dim + 8 <= dimensions; dim += 8) {
        const __m128i first_packed =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(first + dim));
        const __m128i second_packed =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(second + dim));
        const __m256i first_values = _mm256_cvtepi8_epi32(first_packed);
        const __m256i second_values = _mm256_cvtepi8_epi32(second_packed);
        const __m256i difference = _mm256_sub_epi32(first_values, second_values);
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(difference, difference));
    }
    result += horizontal_sum_i32_256(sum);
#endif
    for (; dim < dimensions; ++dim) {
        const int32_t difference =
            static_cast<int32_t>(first[dim]) - static_cast<int32_t>(second[dim]);
        result += difference * difference;
    }
    return static_cast<uint64_t>(result);
}

inline float transformed_query_terms(
    const float* query_alpha,
    const float* node_alpha,
    const float* node_beta,
    uint32_t rank) {
    uint32_t k = 0;
    float result = 0.0f;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (; k + 16 <= rank; k += 16) {
        const __m512 difference =
            _mm512_sub_ps(_mm512_loadu_ps(query_alpha + k), _mm512_loadu_ps(node_alpha + k));
        const __m512 beta = _mm512_loadu_ps(node_beta + k);
        sum = _mm512_fmadd_ps(difference, difference, sum);
        sum = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), _mm512_mul_ps(difference, beta), sum);
    }
    result += horizontal_sum_512(sum);
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    for (; k + 8 <= rank; k += 8) {
        const __m256 difference =
            _mm256_sub_ps(_mm256_loadu_ps(query_alpha + k), _mm256_loadu_ps(node_alpha + k));
        const __m256 beta = _mm256_loadu_ps(node_beta + k);
        sum = _mm256_fmadd_ps(difference, difference, sum);
        sum = _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(difference, beta), sum);
    }
    result += horizontal_sum_256(sum);
#endif
    for (; k < rank; ++k) {
        const float difference = query_alpha[k] - node_alpha[k];
        result += difference * difference - 2.0f * difference * node_beta[k];
    }
    return result;
}

inline float transformed_node_terms(
    const float* first_alpha,
    const float* first_beta,
    const float* second_alpha,
    const float* second_beta,
    uint32_t rank) {
    uint32_t k = 0;
    float result = 0.0f;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (; k + 16 <= rank; k += 16) {
        const __m512 alpha_difference =
            _mm512_sub_ps(_mm512_loadu_ps(first_alpha + k), _mm512_loadu_ps(second_alpha + k));
        const __m512 beta_difference =
            _mm512_sub_ps(_mm512_loadu_ps(first_beta + k), _mm512_loadu_ps(second_beta + k));
        sum = _mm512_fmadd_ps(alpha_difference, alpha_difference, sum);
        sum = _mm512_fmadd_ps(
            _mm512_set1_ps(2.0f),
            _mm512_mul_ps(alpha_difference, beta_difference),
            sum);
    }
    result += horizontal_sum_512(sum);
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    for (; k + 8 <= rank; k += 8) {
        const __m256 alpha_difference =
            _mm256_sub_ps(_mm256_loadu_ps(first_alpha + k), _mm256_loadu_ps(second_alpha + k));
        const __m256 beta_difference =
            _mm256_sub_ps(_mm256_loadu_ps(first_beta + k), _mm256_loadu_ps(second_beta + k));
        sum = _mm256_fmadd_ps(alpha_difference, alpha_difference, sum);
        sum = _mm256_fmadd_ps(
            _mm256_set1_ps(2.0f),
            _mm256_mul_ps(alpha_difference, beta_difference),
            sum);
    }
    result += horizontal_sum_256(sum);
#endif
    for (; k < rank; ++k) {
        const float alpha_difference = first_alpha[k] - second_alpha[k];
        const float beta_difference = first_beta[k] - second_beta[k];
        result +=
            alpha_difference * alpha_difference +
            2.0f * alpha_difference * beta_difference;
    }
    return result;
}

inline uint32_t payload_magic(const void* payload) {
    uint32_t value = 0;
    std::memcpy(&value, payload, sizeof(value));
    return value;
}

inline float cprq_l2_distance(
    const void* first_payload,
    const void* second_payload,
    const void* context_pointer) {
    const CprqDistanceContext& context =
        *static_cast<const CprqDistanceContext*>(context_pointer);
    const uint32_t first_magic = payload_magic(first_payload);
    const uint32_t second_magic = payload_magic(second_payload);

    if (first_magic == kCprqNodeMagic && second_magic == kCprqNodeMagic) {
        const uint8_t* first = static_cast<const uint8_t*>(first_payload);
        const uint8_t* second = static_cast<const uint8_t*>(second_payload);
        const float* first_alpha =
            reinterpret_cast<const float*>(first + context.alpha_offset);
        const float* first_beta =
            reinterpret_cast<const float*>(first + context.beta_offset);
        const float* second_alpha =
            reinterpret_cast<const float*>(second + context.alpha_offset);
        const float* second_beta =
            reinterpret_cast<const float*>(second + context.beta_offset);
        const int8_t* first_bins =
            reinterpret_cast<const int8_t*>(first + context.bins_offset);
        const int8_t* second_bins =
            reinterpret_cast<const int8_t*>(second + context.bins_offset);

        const float transformed_distance =
            transformed_node_terms(
                first_alpha,
                first_beta,
                second_alpha,
                second_beta,
                context.rank);
        const float residual_distance =
            context.residual_step_squared *
            static_cast<float>(
                squared_bin_difference(first_bins, second_bins, context.dimensions));
        return std::max(0.0f, transformed_distance + residual_distance);
    }

    const CprqQueryView* query = nullptr;
    const uint8_t* node = nullptr;
    if (first_magic == kCprqQueryMagic && second_magic == kCprqNodeMagic) {
        query = static_cast<const CprqQueryView*>(first_payload);
        node = static_cast<const uint8_t*>(second_payload);
    } else if (first_magic == kCprqNodeMagic && second_magic == kCprqQueryMagic) {
        query = static_cast<const CprqQueryView*>(second_payload);
        node = static_cast<const uint8_t*>(first_payload);
    } else {
        throw std::runtime_error("CPRQ distance received an unknown payload type");
    }

    const float* node_alpha =
        reinterpret_cast<const float*>(node + context.alpha_offset);
    const float* node_beta =
        reinterpret_cast<const float*>(node + context.beta_offset);
    const int8_t* node_bins =
        reinterpret_cast<const int8_t*>(node + context.bins_offset);
    uint32_t node_bin_norm = 0;
    std::memcpy(&node_bin_norm, node + context.norm_offset, sizeof(node_bin_norm));

    const float transformed_distance =
        transformed_query_terms(
            query->transformed_alpha,
            node_alpha,
            node_beta,
            context.rank);
    const float residual_dot =
        query_residual_dot_bins(query->residual, node_bins, context.dimensions);
    const float residual_distance =
        query->residual_norm +
        context.residual_step_squared * static_cast<float>(node_bin_norm) -
        2.0f * context.residual_step * residual_dot;
    return std::max(0.0f, transformed_distance + residual_distance);
}

class CprqQuery {
 public:
    explicit CprqQuery(uint32_t dimensions, uint32_t rank)
        : transformed_alpha(rank), residual(dimensions) {
        view.transformed_alpha = transformed_alpha.data();
        view.residual = residual.data();
    }

    CprqQueryView view;
    std::vector<float> transformed_alpha;
    std::vector<float> residual;
};

class CprqL2Space : public hnswlib::SpaceInterface<float> {
 public:
    explicit CprqL2Space(const CprqDatabase& database)
        : mean_(database.mean), basis_q_(database.basis_q) {
        context_.dimensions = database.header.cols;
        context_.rank = database.header.rank;
        context_.residual_step = database.residual_step;
        context_.residual_step_squared =
            database.residual_step * database.residual_step;
        context_.alpha_offset = 8;
        context_.beta_offset =
            context_.alpha_offset + sizeof(float) * context_.rank;
        context_.norm_offset =
            context_.beta_offset + sizeof(float) * context_.rank;
        context_.bins_offset = context_.norm_offset + sizeof(uint32_t);
        context_.node_size = context_.bins_offset + context_.dimensions;

        if (mean_.size() != context_.dimensions ||
            basis_q_.size() !=
                static_cast<size_t>(context_.dimensions) * context_.rank) {
            throw std::runtime_error("CPRQ space received inconsistent global model data");
        }
    }

    size_t get_data_size() override {
        return context_.node_size;
    }

    hnswlib::DISTFUNC<float> get_dist_func() override {
        return cprq_l2_distance;
    }

    void* get_dist_func_param() override {
        return &context_;
    }

    const CprqDistanceContext& context() const {
        return context_;
    }

    std::string simd_name() const {
#if defined(__AVX512F__)
        return "AVX-512";
#elif defined(__AVX2__)
        return "AVX2";
#elif defined(__SSE2__)
        return "SSE2/scalar fallback";
#else
        return "scalar";
#endif
    }

    std::vector<uint8_t> make_node_payload(
        const CprqDatabase& database,
        uint32_t row) const {
        if (row >= database.header.rows) {
            throw std::runtime_error("CPRQ node row is out of range");
        }
        std::vector<uint8_t> payload(context_.node_size, 0);
        std::memcpy(payload.data(), &kCprqNodeMagic, sizeof(kCprqNodeMagic));
        const size_t row_rank = static_cast<size_t>(row) * context_.rank;
        const size_t row_dimensions = static_cast<size_t>(row) * context_.dimensions;
        std::memcpy(
            payload.data() + context_.alpha_offset,
            database.transformed_alpha.data() + row_rank,
            sizeof(float) * context_.rank);
        std::memcpy(
            payload.data() + context_.beta_offset,
            database.transformed_beta.data() + row_rank,
            sizeof(float) * context_.rank);
        std::memcpy(
            payload.data() + context_.norm_offset,
            database.residual_bin_norm.data() + row,
            sizeof(uint32_t));
        std::memcpy(
            payload.data() + context_.bins_offset,
            database.residual_bins.data() + row_dimensions,
            context_.dimensions);
        return payload;
    }

    void prepare_query(const float* full_precision_query, CprqQuery& query) const {
        if (query.transformed_alpha.size() != context_.rank ||
            query.residual.size() != context_.dimensions) {
            throw std::runtime_error("CPRQ query scratch has the wrong shape");
        }
        std::fill(
            query.transformed_alpha.begin(),
            query.transformed_alpha.end(),
            0.0f);

        for (uint32_t dim = 0; dim < context_.dimensions; ++dim) {
            const float centered = full_precision_query[dim] - mean_[dim];
            const float* q_row =
                basis_q_.data() + static_cast<size_t>(dim) * context_.rank;
            for (uint32_t k = 0; k < context_.rank; ++k) {
                query.transformed_alpha[k] += q_row[k] * centered;
            }
        }

        double residual_norm = 0.0;
        for (uint32_t dim = 0; dim < context_.dimensions; ++dim) {
            const float centered = full_precision_query[dim] - mean_[dim];
            const float* q_row =
                basis_q_.data() + static_cast<size_t>(dim) * context_.rank;
            float approximation = 0.0f;
            for (uint32_t k = 0; k < context_.rank; ++k) {
                approximation += q_row[k] * query.transformed_alpha[k];
            }
            const float residual = centered - approximation;
            query.residual[dim] = residual;
            residual_norm += static_cast<double>(residual) * residual;
        }
        query.view.residual_norm = static_cast<float>(residual_norm);
    }

 private:
    CprqDistanceContext context_;
    std::vector<float> mean_;
    std::vector<float> basis_q_;
};

}  // namespace cprq_hnsw
