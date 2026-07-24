#include "space_cprq_l2.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FvecsData {
    uint32_t dimensions = 0;
    uint32_t rows = 0;
    std::vector<float> values;
};

FvecsData load_fvecs(const std::string& path, uint32_t expected_dimensions) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open query fvecs file: " + path);
    }
    const std::streamoff file_size = input.tellg();
    if (file_size < static_cast<std::streamoff>(sizeof(int32_t))) {
        throw std::runtime_error("Query fvecs file is too small: " + path);
    }

    input.seekg(0, std::ios::beg);
    int32_t first_dimension = 0;
    input.read(reinterpret_cast<char*>(&first_dimension), sizeof(first_dimension));
    if (!input || first_dimension <= 0 ||
        static_cast<uint32_t>(first_dimension) != expected_dimensions) {
        throw std::runtime_error("Query fvecs dimension does not match CPRQ package");
    }

    const size_t record_size =
        sizeof(int32_t) + static_cast<size_t>(expected_dimensions) * sizeof(float);
    if (file_size % static_cast<std::streamoff>(record_size) != 0) {
        throw std::runtime_error("Query fvecs file has a partial record");
    }
    const size_t row_count = static_cast<size_t>(file_size) / record_size;
    if (row_count > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Query fvecs row count exceeds uint32_t");
    }

    FvecsData result;
    result.dimensions = expected_dimensions;
    result.rows = static_cast<uint32_t>(row_count);
    result.values.resize(row_count * expected_dimensions);

    input.seekg(0, std::ios::beg);
    for (uint32_t row = 0; row < result.rows; ++row) {
        int32_t row_dimension = 0;
        input.read(reinterpret_cast<char*>(&row_dimension), sizeof(row_dimension));
        if (!input || row_dimension != first_dimension) {
            throw std::runtime_error(
                "Inconsistent query fvecs dimension at row " + std::to_string(row));
        }
        input.read(
            reinterpret_cast<char*>(
                result.values.data() + static_cast<size_t>(row) * expected_dimensions),
            static_cast<std::streamsize>(
                static_cast<size_t>(expected_dimensions) * sizeof(float)));
        if (!input) {
            throw std::runtime_error(
                "Unexpected EOF in query fvecs row " + std::to_string(row));
        }
    }
    return result;
}

double seconds_between(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

uint64_t file_size_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open saved HNSW index: " + path);
    }
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("Cannot determine saved HNSW index size: " + path);
    }
    return static_cast<uint64_t>(size);
}

void print_index_file_size(const std::string& path) {
    const uint64_t bytes = file_size_bytes(path);
    std::cout
        << "  index_size_bytes: " << bytes << "\n"
        << "  index_size_megabytes: "
        << static_cast<double>(bytes) / 1000000.0 << "\n"
        << "  index_size_gigabytes: "
        << static_cast<double>(bytes) / 1000000000.0 << "\n";
}

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " [M] [ef_construction] [ef_search] [query_count]"
        << " [compressed_package] [query_fvecs] [optional_index_output]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 8) {
            print_usage(argv[0]);
            return 2;
        }

        const size_t m = argc > 1 ? std::stoul(argv[1]) : 16;
        const size_t ef_construction = argc > 2 ? std::stoul(argv[2]) : 200;
        const size_t ef_search = argc > 3 ? std::stoul(argv[3]) : 200;
        const uint32_t requested_query_count =
            argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 1000;
        const std::string package_path = argc > 5
            ? argv[5]
            : "/tmp/qlrpcomp_gpu_pool_mean_rSVD_model_components_r64_qb_8_abs0905/"
              "embeds_0[(16,20,36,3584)]_w_encoding_full_package/"
              "w_encoding_full_package_zstd.bin";
        const std::string query_path = argc > 6
            ? argv[6]
            : "/tmp/qlrpcomp_gpu_pool_kmeans_model_components_r64_qb_8_abs0905_decompressed/"
              "embeds_0[(16,20,36,3584)].fvecs";
#ifdef CPRQ_HNSW_DEFAULT_INDEX_OUTPUT
        const std::string default_index_output = CPRQ_HNSW_DEFAULT_INDEX_OUTPUT;
#else
        const std::string default_index_output = "hnsw_qlrpcomp/hnsw.bin";
#endif
        const std::string index_output =
            argc > 7 ? argv[7] : default_index_output;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "[CPRQ HNSW] Loading and preparing compressed package\n"
                  << "  package: " << package_path << "\n";
        const auto package_start = std::chrono::steady_clock::now();
        cprq_hnsw::CprqDatabase database =
            cprq_hnsw::load_cprq_database(package_path);
        const auto package_end = std::chrono::steady_clock::now();

        const cprq_hnsw::PackageHeader& header = database.header;
        const cprq_hnsw::DecodeStatistics& stats = database.decode_statistics;
        const uint64_t clipped_values =
            stats.clipped_low_values + stats.clipped_high_values;
        const uint64_t total_values =
            static_cast<uint64_t>(header.rows) * header.cols;

        cprq_hnsw::CprqL2Space space(database);
        std::cout
            << "[CPRQ HNSW] Package/model preparation\n"
            << "  rows: " << header.rows << "\n"
            << "  dimensions: " << header.cols << "\n"
            << "  rank: " << header.rank << "\n"
            << "  block_size: " << header.block_size << "\n"
            << "  blocks_per_dimension: " << header.blocks_per_dimension << "\n"
            << "  absolute_error_bound: " << header.absolute_error_bound << "\n"
            << "  residual_step: " << database.residual_step << "\n"
            << "  package_maximum_magnitude: " << header.maximum_magnitude << "\n"
            << "  decoded_maximum_magnitude: " << stats.largest_decoded_magnitude << "\n"
            << "  zero_blocks: " << stats.zero_blocks << "\n"
            << "  nonzero_blocks: " << stats.nonzero_blocks << "\n"
            << "  clipped_low_values: " << stats.clipped_low_values << "\n"
            << "  clipped_high_values: " << stats.clipped_high_values << "\n"
            << "  clipped_fraction: "
            << (total_values == 0
                    ? 0.0
                    : static_cast<double>(clipped_values) / total_values)
            << "\n"
            << "  node_payload_bytes: " << space.get_data_size() << "\n"
            << "  SIMD_path: " << space.simd_name() << "\n"
            << "  package_decode_QR_alpha_beta_seconds: "
            << seconds_between(package_start, package_end) << "\n";

        std::cout << "[CPRQ HNSW] Constructing graph\n"
                  << "  M: " << m << "\n"
                  << "  ef_construction: " << ef_construction << "\n"
                  << "  ef_search: " << ef_search << "\n";
        hnswlib::HierarchicalNSW<float> index(
            &space,
            header.rows,
            m,
            ef_construction);
        index.setEf(ef_search);

        const auto build_start = std::chrono::steady_clock::now();
        for (uint32_t row = 0; row < header.rows; ++row) {
            const std::vector<uint8_t> node =
                space.make_node_payload(database, row);
            index.addPoint(node.data(), static_cast<hnswlib::labeltype>(row));
        }
        const auto build_end = std::chrono::steady_clock::now();
        const double build_seconds = seconds_between(build_start, build_end);
        std::cout
            << "[CPRQ HNSW] Graph construction results\n"
            << "  build_seconds: " << build_seconds << "\n"
            << "  build_vectors_per_second: "
            << static_cast<double>(header.rows) / build_seconds << "\n";

        const auto save_start = std::chrono::steady_clock::now();
        index.saveIndex(index_output);
        const auto save_end = std::chrono::steady_clock::now();
        std::cout
            << "[CPRQ HNSW] Saved graph index\n"
            << "  index_output: " << index_output << "\n"
            << "  index_save_seconds: "
            << seconds_between(save_start, save_end) << "\n";
        print_index_file_size(index_output);

        std::cout << "[CPRQ HNSW] Loading full-precision queries\n"
                  << "  query_fvecs: " << query_path << "\n";
        FvecsData queries = load_fvecs(query_path, header.cols);
        const uint32_t query_count =
            std::min(requested_query_count, std::min(queries.rows, header.rows));
        if (query_count == 0) {
            throw std::runtime_error("Query count is zero");
        }

        cprq_hnsw::CprqQuery query(header.cols, header.rank);
        space.prepare_query(queries.values.data(), query);
        const std::vector<uint8_t> validation_node_0 =
            space.make_node_payload(database, 0);
        const std::vector<uint8_t> validation_node_1 =
            space.make_node_payload(database, 1);
        const float formula_query_distance =
            cprq_hnsw::cprq_l2_distance(
                &query.view,
                validation_node_0.data(),
                &space.context());
        const float formula_node_distance =
            cprq_hnsw::cprq_l2_distance(
                validation_node_0.data(),
                validation_node_1.data(),
                &space.context());

        double direct_query_distance = 0.0;
        double direct_node_distance = 0.0;
        for (uint32_t dim = 0; dim < header.cols; ++dim) {
            const float* basis_row =
                database.basis_q.data() + static_cast<size_t>(dim) * header.rank;
            double low_rank_0 = 0.0;
            double low_rank_1 = 0.0;
            for (uint32_t k = 0; k < header.rank; ++k) {
                low_rank_0 +=
                    static_cast<double>(basis_row[k]) *
                    database.transformed_alpha[k];
                low_rank_1 +=
                    static_cast<double>(basis_row[k]) *
                    database.transformed_alpha[header.rank + k];
            }
            const double residual_0 =
                database.residual_step *
                database.residual_bins[dim];
            const double residual_1 =
                database.residual_step *
                database.residual_bins[header.cols + dim];
            const double centered_query =
                static_cast<double>(queries.values[dim]) - database.mean[dim];
            const double query_difference =
                centered_query - low_rank_0 - residual_0;
            const double node_difference =
                low_rank_0 + residual_0 - low_rank_1 - residual_1;
            direct_query_distance += query_difference * query_difference;
            direct_node_distance += node_difference * node_difference;
        }
        std::cout
            << "[CPRQ HNSW] Distance validation against explicit reconstruction\n"
            << "  query_node_formula: " << formula_query_distance << "\n"
            << "  query_node_direct: " << direct_query_distance << "\n"
            << "  query_node_absolute_error: "
            << std::abs(formula_query_distance - direct_query_distance) << "\n"
            << "  node_node_formula: " << formula_node_distance << "\n"
            << "  node_node_direct: " << direct_node_distance << "\n"
            << "  node_node_absolute_error: "
            << std::abs(formula_node_distance - direct_node_distance) << "\n";

        uint32_t self_matches = 0;
        double preparation_seconds = 0.0;
        double hnsw_search_seconds = 0.0;
        const auto end_to_end_start = std::chrono::steady_clock::now();
        for (uint32_t row = 0; row < query_count; ++row) {
            const float* full_precision_query =
                queries.values.data() + static_cast<size_t>(row) * header.cols;

            const auto preparation_start = std::chrono::steady_clock::now();
            space.prepare_query(full_precision_query, query);
            const auto preparation_end = std::chrono::steady_clock::now();
            preparation_seconds +=
                seconds_between(preparation_start, preparation_end);

            const auto search_start = std::chrono::steady_clock::now();
            const std::priority_queue<
                std::pair<float, hnswlib::labeltype>> result =
                index.searchKnn(&query.view, 1);
            const auto search_end = std::chrono::steady_clock::now();
            hnsw_search_seconds += seconds_between(search_start, search_end);

            if (!result.empty() && result.top().second == row) {
                ++self_matches;
            }
        }
        const auto end_to_end_end = std::chrono::steady_clock::now();
        const double end_to_end_seconds =
            seconds_between(end_to_end_start, end_to_end_end);

        std::cout
            << "[CPRQ HNSW] Full-precision query results\n"
            << "  query_count: " << query_count << "\n"
            << "  self_top1_matches: " << self_matches << "\n"
            << "  self_top1_recall: "
            << static_cast<double>(self_matches) / query_count << "\n"
            << "  query_preparation_seconds: " << preparation_seconds << "\n"
            << "  hnsw_search_seconds: " << hnsw_search_seconds << "\n"
            << "  hnsw_search_qps_excluding_query_preparation: "
            << static_cast<double>(query_count) / hnsw_search_seconds << "\n"
            << "  end_to_end_query_seconds: " << end_to_end_seconds << "\n"
            << "  end_to_end_qps_including_query_preparation: "
            << static_cast<double>(query_count) / end_to_end_seconds << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[CPRQ HNSW] ERROR: " << error.what() << "\n";
        return 1;
    }
}
