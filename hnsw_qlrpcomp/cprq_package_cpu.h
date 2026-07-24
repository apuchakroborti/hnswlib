#pragma once

#include <zstd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <immintrin.h>

namespace cprq_hnsw {

struct QuantizedBlock {
    float min_value = 0.0f;
    float max_value = 0.0f;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint8_t bits = 0;
    std::vector<uint8_t> values;
};

struct PackageHeader {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t manual_block_size = 0;
    float absolute_error_bound = 0.0f;
    uint32_t rank = 0;
    uint32_t oversampling = 0;
    uint32_t maximum_magnitude = 0;
    uint32_t blocks_per_dimension = 0;
    uint32_t block_size = 0;
    uint64_t residual_stream_size = 0;
};

struct DecodeStatistics {
    uint64_t zero_blocks = 0;
    uint64_t nonzero_blocks = 0;
    uint64_t clipped_low_values = 0;
    uint64_t clipped_high_values = 0;
    uint32_t largest_decoded_magnitude = 0;
};

struct CprqDatabase {
    PackageHeader header;
    DecodeStatistics decode_statistics;
    float residual_step = 0.0f;

    // Global model data. basis_q is column-orthonormal and basis_r satisfies B = Q R.
    std::vector<float> mean;
    std::vector<float> basis_q;
    std::vector<float> basis_r;

    // Row-major node data prepared from the quantized model and clipped residual bins.
    std::vector<float> transformed_alpha;
    std::vector<float> transformed_beta;
    std::vector<uint32_t> residual_bin_norm;
    std::vector<int8_t> residual_bins;
};

class ByteReader {
 public:
    explicit ByteReader(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}

    template <typename T>
    T scalar() {
        static_assert(std::is_trivially_copyable<T>::value, "scalar must be POD");
        require(sizeof(T), "scalar");
        T value{};
        std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    template <typename T>
    std::vector<T> vector() {
        const uint64_t count64 = scalar<uint64_t>();
        if (count64 > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CPRQ vector length overflows size_t");
        }
        const size_t count = static_cast<size_t>(count64);
        const size_t bytes = count * sizeof(T);
        require(bytes, "vector");
        std::vector<T> result(count);
        if (bytes != 0) {
            std::memcpy(result.data(), bytes_.data() + offset_, bytes);
            offset_ += bytes;
        }
        return result;
    }

    QuantizedBlock quantized_block() {
        QuantizedBlock block;
        block.min_value = scalar<float>();
        block.max_value = scalar<float>();
        block.rows = scalar<uint32_t>();
        block.cols = scalar<uint32_t>();
        block.bits = scalar<uint8_t>();
        block.values = vector<uint8_t>();
        return block;
    }

    size_t offset() const {
        return offset_;
    }

 private:
    void require(size_t count, const char* description) const {
        if (offset_ > bytes_.size() || bytes_.size() - offset_ < count) {
            throw std::runtime_error(
                std::string("CPRQ package is truncated while reading ") + description);
        }
    }

    const std::vector<uint8_t>& bytes_;
    size_t offset_ = 0;
};

inline std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open CPRQ package: " + path);
    }
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("Cannot determine CPRQ package size: " + path);
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    if (!input) {
        throw std::runtime_error("Cannot read CPRQ package: " + path);
    }
    return bytes;
}

inline uint32_t first_u32(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < sizeof(uint32_t)) {
        return 0;
    }
    uint32_t value = 0;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
}

inline uint64_t u64_at(const std::vector<uint8_t>& bytes, size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(uint64_t)) {
        throw std::runtime_error("nvCOMP-native header is truncated");
    }
    uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

inline std::vector<uint8_t> decompress_nvcomp_native_zstd_cpu(
    const std::vector<uint8_t>& saved) {
    constexpr uint32_t kNvcompNativeMagic = 0x52911326u;
    constexpr size_t kFixedHeaderSize = 72u;
    if (first_u32(saved) != kNvcompNativeMagic || saved.size() < kFixedHeaderSize) {
        return {};
    }

    const uint64_t compressed_payload_size = u64_at(saved, 8);
    const uint64_t raw_size = u64_at(saved, 16);
    const uint64_t chunk_count = u64_at(saved, 24);
    const uint64_t chunk_size = u64_at(saved, 48);
    const uint64_t payload_offset = u64_at(saved, 56);
    if (chunk_count == 0 || chunk_size == 0 ||
        chunk_count > std::numeric_limits<size_t>::max() / 16u) {
        throw std::runtime_error("Invalid nvCOMP-native Zstd chunk metadata");
    }
    const uint64_t expected_payload_offset =
        kFixedHeaderSize + 2u * chunk_count * sizeof(uint64_t);
    if (payload_offset != expected_payload_offset ||
        payload_offset > saved.size() ||
        compressed_payload_size != saved.size() - payload_offset) {
        throw std::runtime_error("Unsupported nvCOMP-native Zstd header layout");
    }
    if (raw_size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("nvCOMP-native raw size exceeds size_t");
    }

    std::vector<uint8_t> raw(static_cast<size_t>(raw_size));
    const size_t offsets_base = kFixedHeaderSize;
    const size_t sizes_base =
        offsets_base + static_cast<size_t>(chunk_count) * sizeof(uint64_t);
    for (uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const uint64_t relative_offset =
            u64_at(saved, offsets_base + static_cast<size_t>(chunk) * sizeof(uint64_t));
        const uint64_t compressed_size =
            u64_at(saved, sizes_base + static_cast<size_t>(chunk) * sizeof(uint64_t));
        if (relative_offset > compressed_payload_size ||
            compressed_size > compressed_payload_size - relative_offset) {
            throw std::runtime_error("nvCOMP-native Zstd chunk exceeds payload");
        }

        const uint64_t raw_offset = chunk * chunk_size;
        if (raw_offset >= raw_size) {
            throw std::runtime_error("nvCOMP-native Zstd chunk exceeds raw output");
        }
        const size_t expected_chunk_size = static_cast<size_t>(
            std::min<uint64_t>(chunk_size, raw_size - raw_offset));
        const size_t decoded_size = ZSTD_decompress(
            raw.data() + static_cast<size_t>(raw_offset),
            expected_chunk_size,
            saved.data() + static_cast<size_t>(payload_offset + relative_offset),
            static_cast<size_t>(compressed_size));
        if (ZSTD_isError(decoded_size) || decoded_size != expected_chunk_size) {
            throw std::runtime_error(
                std::string("CPU Zstd failed for nvCOMP-native chunk ") +
                std::to_string(chunk) + ": " + ZSTD_getErrorName(decoded_size));
        }
    }
    return raw;
}

inline std::vector<uint8_t> load_raw_or_zstd_package(const std::string& path) {
    std::vector<uint8_t> saved = read_file_bytes(path);
    constexpr uint32_t kPackageMagic = 0x51505745u;
    if (first_u32(saved) == kPackageMagic) {
        return saved;
    }

    std::vector<uint8_t> nvcomp_raw = decompress_nvcomp_native_zstd_cpu(saved);
    if (!nvcomp_raw.empty()) {
        if (first_u32(nvcomp_raw) != kPackageMagic) {
            throw std::runtime_error("nvCOMP-native output is not a CPRQ package");
        }
        return nvcomp_raw;
    }

    // A standard single-frame Zstd package is accepted as a second lossless format.
    const unsigned long long content_size =
        ZSTD_getFrameContentSize(saved.data(), saved.size());
    if (content_size == ZSTD_CONTENTSIZE_ERROR) {
        throw std::runtime_error(
            "Package is neither raw CPRQ nor a usable compressed package: " +
            path);
    }
    if (content_size == ZSTD_CONTENTSIZE_UNKNOWN ||
        content_size > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("Zstandard package does not provide a usable content size");
    }

    std::vector<uint8_t> raw(static_cast<size_t>(content_size));
    const size_t decoded_size =
        ZSTD_decompress(raw.data(), raw.size(), saved.data(), saved.size());
    if (ZSTD_isError(decoded_size)) {
        throw std::runtime_error(
            std::string("Zstandard package decompression failed: ") +
            ZSTD_getErrorName(decoded_size));
    }
    if (decoded_size != raw.size() || first_u32(raw) != kPackageMagic) {
        throw std::runtime_error("Decoded Zstandard bytes are not a complete CPRQ package");
    }
    return raw;
}

inline std::vector<float> dequantize_block(
    const QuantizedBlock& block,
    size_t expected_count,
    const char* name) {
    if (block.bits == 0 || block.bits > 8) {
        throw std::runtime_error(std::string(name) + " uses unsupported quantization bits");
    }
    if (static_cast<size_t>(block.rows) * block.cols != expected_count ||
        block.values.size() != expected_count) {
        throw std::runtime_error(std::string(name) + " shape/value count mismatch");
    }
    const uint32_t levels = (1u << block.bits) - 1u;
    const float scale =
        (block.max_value - block.min_value) / static_cast<float>(levels);
    std::vector<float> output(expected_count);
    for (size_t i = 0; i < expected_count; ++i) {
        output[i] = block.min_value + scale * static_cast<float>(block.values[i]);
    }
    return output;
}

inline void decode_residual_stream(
    const std::vector<uint8_t>& stream,
    const PackageHeader& header,
    std::vector<int8_t>& row_major_bins,
    std::vector<uint32_t>& bin_norms,
    DecodeStatistics& statistics) {
    constexpr uint8_t kWidthMask = 0x3fu;

    if (header.rows == 0 || header.cols == 0 || header.block_size == 0 ||
        header.blocks_per_dimension == 0) {
        throw std::runtime_error("Invalid residual stream dimensions");
    }
    const uint32_t expected_blocks_per_dimension =
        (header.rows + header.block_size - 1u) / header.block_size;
    if (header.blocks_per_dimension != expected_blocks_per_dimension) {
        throw std::runtime_error("Residual blocks-per-dimension metadata mismatch");
    }

    const size_t value_count = static_cast<size_t>(header.rows) * header.cols;
    row_major_bins.assign(value_count, 0);
    bin_norms.assign(header.rows, 0);

    const uint64_t block_count =
        static_cast<uint64_t>(header.cols) * header.blocks_per_dimension;
    const uint64_t mask_word_count = (block_count + 31u) / 32u;
    const size_t mask_bytes = static_cast<size_t>(mask_word_count) * sizeof(uint32_t);
    if (stream.size() < mask_bytes) {
        throw std::runtime_error("Residual block-mask stream is truncated");
    }

    std::vector<uint32_t> mask_words(static_cast<size_t>(mask_word_count));
    std::memcpy(mask_words.data(), stream.data(), mask_bytes);
    uint64_t nonzero_count = 0;
    for (uint64_t block = 0; block < block_count; ++block) {
        if ((mask_words[block >> 5u] & (1u << (block & 31u))) != 0u) {
            ++nonzero_count;
        }
    }
    if (nonzero_count > stream.size() - mask_bytes) {
        throw std::runtime_error("Residual nonzero-header table is truncated");
    }

    const uint8_t* nonzero_headers = stream.data() + mask_bytes;
    const uint8_t* payload = nonzero_headers + nonzero_count;
    const size_t payload_size = stream.size() - mask_bytes - nonzero_count;
    size_t payload_offset = 0;
    uint64_t nonzero_index = 0;

    for (uint32_t dim = 0; dim < header.cols; ++dim) {
        for (uint32_t block = 0; block < header.blocks_per_dimension; ++block) {
            const uint64_t logical_block =
                static_cast<uint64_t>(dim) * header.blocks_per_dimension + block;
            const uint32_t row_start = block * header.block_size;
            const uint32_t rows_in_block =
                std::min(header.block_size, header.rows - row_start);

            const bool nonzero =
                (mask_words[logical_block >> 5u] &
                 (1u << (logical_block & 31u))) != 0u;
            if (!nonzero) {
                ++statistics.zero_blocks;
                continue;
            }
            ++statistics.nonzero_blocks;

            const uint8_t block_header = nonzero_headers[nonzero_index++];
            const uint32_t bit_width = block_header & kWidthMask;
            if ((block_header & 0x80u) != 0u || bit_width == 0 || bit_width > 32) {
                throw std::runtime_error(
                    "Residual block has invalid magnitude bit width at dimension " +
                    std::to_string(dim) + ", block " + std::to_string(block) +
                    ", header " + std::to_string(block_header));
            }
            const size_t bytes_per_plane = (rows_in_block + 7u) / 8u;
            const size_t sign_bytes = bytes_per_plane;
            const size_t magnitude_bytes = static_cast<size_t>(bit_width) * bytes_per_plane;
            if (payload_offset > payload_size ||
                payload_size - payload_offset < sign_bytes + magnitude_bytes) {
                throw std::runtime_error("Residual block payload is truncated");
            }

            const uint8_t* signs = payload + payload_offset;
            const uint8_t* magnitude_planes = signs + sign_bytes;
            payload_offset += sign_bytes + magnitude_bytes;

            for (uint32_t lane = 0; lane < rows_in_block; ++lane) {
                const size_t byte_index = lane / 8u;
                const uint8_t sign_mask =
                    static_cast<uint8_t>(1u << (lane % 8u));
                uint32_t magnitude = 0;
                for (uint32_t bit = 0; bit < bit_width; ++bit) {
                    const uint64_t packed_bit =
                        static_cast<uint64_t>(lane) * bit_width + bit;
                    const uint8_t byte = magnitude_planes[packed_bit / 8u];
                    if ((byte & (1u << (packed_bit % 8u))) != 0u) {
                        magnitude |= (1u << bit);
                    }
                }
                statistics.largest_decoded_magnitude =
                    std::max(statistics.largest_decoded_magnitude, magnitude);

                int64_t signed_value = 0;
                if (magnitude != 0) {
                    const bool positive = (signs[byte_index] & sign_mask) != 0;
                    signed_value = positive
                        ? static_cast<int64_t>(magnitude)
                        : -static_cast<int64_t>(magnitude);
                }
                if (signed_value < -128) {
                    signed_value = -128;
                    ++statistics.clipped_low_values;
                } else if (signed_value > 127) {
                    signed_value = 127;
                    ++statistics.clipped_high_values;
                }

                const uint32_t row = row_start + lane;
                const int8_t bin = static_cast<int8_t>(signed_value);
                row_major_bins[static_cast<size_t>(row) * header.cols + dim] = bin;
                const int32_t widened = static_cast<int32_t>(bin);
                bin_norms[row] += static_cast<uint32_t>(widened * widened);
            }
        }
    }
    if (nonzero_index != nonzero_count || payload_offset != payload_size) {
        throw std::runtime_error("Residual block-mask stream has trailing bytes");
    }
}

inline void thin_qr_mgs(
    const std::vector<float>& basis,
    uint32_t dimensions,
    uint32_t rank,
    std::vector<float>& q,
    std::vector<float>& r) {
    if (basis.size() != static_cast<size_t>(dimensions) * rank) {
        throw std::runtime_error("Basis shape mismatch before QR factorization");
    }
    q.assign(basis.size(), 0.0f);
    r.assign(static_cast<size_t>(rank) * rank, 0.0f);
    std::vector<double> column(dimensions);

    for (uint32_t k = 0; k < rank; ++k) {
        for (uint32_t d = 0; d < dimensions; ++d) {
            column[d] = basis[static_cast<size_t>(d) * rank + k];
        }

        // Two MGS passes make the small-rank factorization stable enough for quantized B.
        for (int pass = 0; pass < 2; ++pass) {
            for (uint32_t j = 0; j < k; ++j) {
                double projection = 0.0;
                for (uint32_t d = 0; d < dimensions; ++d) {
                    projection +=
                        static_cast<double>(q[static_cast<size_t>(d) * rank + j]) *
                        column[d];
                }
                r[static_cast<size_t>(j) * rank + k] +=
                    static_cast<float>(projection);
                for (uint32_t d = 0; d < dimensions; ++d) {
                    column[d] -=
                        projection * q[static_cast<size_t>(d) * rank + j];
                }
            }
        }

        double squared_norm = 0.0;
        for (double value : column) {
            squared_norm += value * value;
        }
        const double norm = std::sqrt(squared_norm);
        if (!(norm > 1.0e-10)) {
            throw std::runtime_error("Quantized basis is numerically rank deficient");
        }
        r[static_cast<size_t>(k) * rank + k] = static_cast<float>(norm);
        for (uint32_t d = 0; d < dimensions; ++d) {
            q[static_cast<size_t>(d) * rank + k] =
                static_cast<float>(column[d] / norm);
        }
    }
}

inline void transform_alpha(
    const std::vector<float>& alpha,
    const std::vector<float>& r,
    uint32_t rows,
    uint32_t rank,
    std::vector<float>& transformed) {
    transformed.assign(static_cast<size_t>(rows) * rank, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        const float* source = alpha.data() + static_cast<size_t>(row) * rank;
        float* target = transformed.data() + static_cast<size_t>(row) * rank;
        for (uint32_t j = 0; j < rank; ++j) {
            float sum = 0.0f;
            for (uint32_t k = j; k < rank; ++k) {
                sum += r[static_cast<size_t>(j) * rank + k] * source[k];
            }
            target[j] = sum;
        }
    }
}

inline void prepare_residual_beta(
    const std::vector<int8_t>& bins,
    const std::vector<float>& q,
    uint32_t rows,
    uint32_t dimensions,
    uint32_t rank,
    float residual_step,
    std::vector<float>& beta) {
    beta.assign(static_cast<size_t>(rows) * rank, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t row64 = 0; row64 < static_cast<int64_t>(rows); ++row64) {
        const uint32_t row = static_cast<uint32_t>(row64);
        const int8_t* row_bins = bins.data() + static_cast<size_t>(row) * dimensions;
        float* row_beta = beta.data() + static_cast<size_t>(row) * rank;
        for (uint32_t dim = 0; dim < dimensions; ++dim) {
            const float residual = residual_step * static_cast<float>(row_bins[dim]);
            const float* q_row = q.data() + static_cast<size_t>(dim) * rank;
            uint32_t k = 0;
#if defined(__AVX512F__)
            const __m512 residual_vector = _mm512_set1_ps(residual);
            for (; k + 16 <= rank; k += 16) {
                const __m512 old_value = _mm512_loadu_ps(row_beta + k);
                const __m512 basis_value = _mm512_loadu_ps(q_row + k);
                _mm512_storeu_ps(
                    row_beta + k,
                    _mm512_fmadd_ps(residual_vector, basis_value, old_value));
            }
#elif defined(__AVX2__)
            const __m256 residual_vector = _mm256_set1_ps(residual);
            for (; k + 8 <= rank; k += 8) {
                const __m256 old_value = _mm256_loadu_ps(row_beta + k);
                const __m256 basis_value = _mm256_loadu_ps(q_row + k);
                _mm256_storeu_ps(
                    row_beta + k,
                    _mm256_fmadd_ps(residual_vector, basis_value, old_value));
            }
#endif
            for (; k < rank; ++k) {
                row_beta[k] += residual * q_row[k];
            }
        }
    }
}

inline CprqDatabase load_cprq_database(const std::string& path) {
    constexpr uint32_t kPackageMagic = 0x51505745u;
    constexpr uint32_t kPackageVersion = 1u;

    std::vector<uint8_t> raw = load_raw_or_zstd_package(path);
    ByteReader reader(raw);
    const uint32_t magic = reader.scalar<uint32_t>();
    const uint32_t version = reader.scalar<uint32_t>();
    if (magic != kPackageMagic || version != kPackageVersion) {
        throw std::runtime_error("Unsupported CPRQ W-encoding package header");
    }

    CprqDatabase database;
    PackageHeader& header = database.header;
    header.rows = reader.scalar<uint32_t>();
    header.cols = reader.scalar<uint32_t>();
    header.manual_block_size = reader.scalar<uint32_t>();
    header.absolute_error_bound = reader.scalar<float>();
    header.rank = reader.scalar<uint32_t>();
    header.oversampling = reader.scalar<uint32_t>();
    header.maximum_magnitude = reader.scalar<uint32_t>();
    header.blocks_per_dimension = reader.scalar<uint32_t>();
    header.block_size = reader.scalar<uint32_t>();
    header.residual_stream_size = reader.scalar<uint64_t>();

    const QuantizedBlock mean_block = reader.quantized_block();
    const QuantizedBlock basis_block = reader.quantized_block();
    const QuantizedBlock alpha_block = reader.quantized_block();
    const std::vector<uint8_t> residual_stream = reader.vector<uint8_t>();
    if (reader.offset() != raw.size()) {
        throw std::runtime_error("CPRQ package contains trailing bytes");
    }
    if (residual_stream.size() != header.residual_stream_size) {
        throw std::runtime_error("CPRQ residual stream size does not match its header");
    }

    database.mean =
        dequantize_block(mean_block, header.cols, "quantized mean");
    const std::vector<float> basis =
        dequantize_block(
            basis_block,
            static_cast<size_t>(header.cols) * header.rank,
            "quantized basis");
    const std::vector<float> alpha =
        dequantize_block(
            alpha_block,
            static_cast<size_t>(header.rows) * header.rank,
            "quantized alpha");

    database.residual_step = 2.0f * header.absolute_error_bound;
    decode_residual_stream(
        residual_stream,
        header,
        database.residual_bins,
        database.residual_bin_norm,
        database.decode_statistics);

    thin_qr_mgs(
        basis,
        header.cols,
        header.rank,
        database.basis_q,
        database.basis_r);
    transform_alpha(
        alpha,
        database.basis_r,
        header.rows,
        header.rank,
        database.transformed_alpha);
    prepare_residual_beta(
        database.residual_bins,
        database.basis_q,
        header.rows,
        header.cols,
        header.rank,
        database.residual_step,
        database.transformed_beta);
    return database;
}

}  // namespace cprq_hnsw
