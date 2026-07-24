#include "../../hnswlib/hnswlib.h"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct FvecsData {
    int dim = 0;
    int max_elements = 0;
    std::vector<float> data;
};

FvecsData load_fvecs(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open fvecs file: " + path);
    }

    const std::streamoff file_size = input.tellg();
    if (file_size < static_cast<std::streamoff>(sizeof(int32_t))) {
        throw std::runtime_error("fvecs file is too small: " + path);
    }

    input.seekg(0, std::ios::beg);
    int32_t dim_i32 = 0;
    input.read(reinterpret_cast<char*>(&dim_i32), sizeof(dim_i32));
    if (!input || dim_i32 <= 0) {
        throw std::runtime_error("Invalid first-vector dimension in fvecs file");
    }

    const size_t dim = static_cast<size_t>(dim_i32);
    const size_t record_size = sizeof(int32_t) + dim * sizeof(float);
    if (file_size % static_cast<std::streamoff>(record_size) != 0) {
        throw std::runtime_error("fvecs file size is not divisible by record size");
    }

    const size_t num_vectors = static_cast<size_t>(file_size) / record_size;
    FvecsData loaded;
    loaded.dim = static_cast<int>(dim);
    loaded.max_elements = static_cast<int>(num_vectors);
    loaded.data.resize(num_vectors * dim);

    input.seekg(0, std::ios::beg);
    for (size_t row = 0; row < num_vectors; row++) {
        int32_t row_dim = 0;
        input.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
        if (!input || row_dim != dim_i32) {
            throw std::runtime_error("Inconsistent fvecs row dimension at row " + std::to_string(row));
        }

        input.read(reinterpret_cast<char*>(loaded.data.data() + row * dim), dim * sizeof(float));
        if (!input) {
            throw std::runtime_error("Unexpected EOF while reading fvecs row " + std::to_string(row));
        }
    }

    return loaded;
}

double elapsed_seconds(
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
    std::cout << std::fixed << std::setprecision(6)
              << "[EXAMPLE SEARCH] HNSW index output: " << path << "\n"
              << "[EXAMPLE SEARCH] HNSW index size bytes: " << bytes << "\n"
              << "[EXAMPLE SEARCH] HNSW index size megabytes: "
              << static_cast<double>(bytes) / 1000000.0 << "\n"
              << "[EXAMPLE SEARCH] HNSW index size gigabytes: "
              << static_cast<double>(bytes) / 1000000000.0 << "\n";
}

int main(int argc, char** argv) {
    if (argc > 5) {
        std::cerr
            << "Usage: " << argv[0]
            << " [M] [ef_construction] [data_fvecs] [index_output]\n";
        return 2;
    }

    int M = argc > 1 ? std::stoi(argv[1]) : 16;
    int ef_construction = argc > 2 ? std::stoi(argv[2]) : 200;
    const std::string data_path = argc > 3
        ? argv[3]
        : "/tmp/qlrpcomp_gpu_pool_kmeans_model_components_r64_qb_8_abs0905_decompressed/embeds_0[(16,20,36,3584)].fvecs";
    const std::string hnsw_path = argc > 4 ? argv[4] : "hnsw.bin";

    std::cout << "[EXAMPLE SEARCH] Loading fvecs data: " << data_path << std::endl;
    FvecsData loaded = load_fvecs(data_path);
    int dim = loaded.dim;
    int max_elements = loaded.max_elements;
    float* data = loaded.data.data();

    std::cout << "[EXAMPLE SEARCH] Initiating L2 Space: " << std::endl;
    // Initing index
    hnswlib::L2Space space(dim);
    std::cout << "[EXAMPLE SEARCH] Initiating HNSW Graph: " << std::endl;
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    alg_hnsw->setEf(ef_construction);

    std::cout << "[EXAMPLE SEARCH] Loaded fvecs data: \n"
        << "Path: "<< data_path << "\n"
        << "Dimension: "<< dim << "\n"
        << "max_elements: "<< max_elements << "\n"
        << "M: "<< M << "\n"
        << "ef_construction: "<< ef_construction << "\n"
        << std::endl;

    std::cout << "[EXAMPLE SEARCH] Adding data to Index: " << std::endl;
    // Add data to index
    const auto build_start = std::chrono::steady_clock::now();
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }
    const auto build_end = std::chrono::steady_clock::now();
    const double build_time_seconds = elapsed_seconds(build_start, build_end);
    std::cout << "[EXAMPLE SEARCH] HNSW graph build time seconds: "
              << build_time_seconds << "\n";
    std::cout << "[EXAMPLE SEARCH] HNSW graph build vectors per second: "
              << (max_elements / build_time_seconds) << "\n";

    std::cout << "[EXAMPLE SEARCH] Query the elements for themselves and measure recall: " << std::endl;
    // Query the elements for themselves and measure recall
    float correct = 0;
    const auto search_start = std::chrono::steady_clock::now();
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    const auto search_end = std::chrono::steady_clock::now();
    const double search_time_seconds = elapsed_seconds(search_start, search_end);
    const double qps = max_elements / search_time_seconds;
    
    std::cout << "[EXAMPLE SEARCH] Correct: " << correct << std::endl;
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    std::cout << "[EXAMPLE SEARCH] Total search time seconds: "
              << search_time_seconds << "\n";
    std::cout << "[EXAMPLE SEARCH] Query per second QPS: "
              << qps << "\n";

    // Serialize index
    alg_hnsw->saveIndex(hnsw_path);
    print_index_file_size(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    alg_hnsw->setEf(ef_construction);
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = (float)correct / max_elements;
    std::cout << "Recall of deserialized index: " << recall << "\n";

    delete alg_hnsw;
    return 0;
}
