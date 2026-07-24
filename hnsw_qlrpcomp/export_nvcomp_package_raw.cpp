#include <gpu_pooling/QLRPCompPool.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open nvCOMP package: " + path);
    }
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("Cannot determine nvCOMP package size");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (!input) {
        throw std::runtime_error("Cannot read nvCOMP package: " + path);
    }
    return bytes;
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot open raw package output: " + path);
    }
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("Cannot write raw package output: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "Usage: " << argv[0]
            << " input_nvcomp_zstd.bin expected_raw_size output_raw.bin\n";
        return 2;
    }

    cudaStream_t stream = nullptr;
    try {
        const std::vector<uint8_t> saved = read_bytes(argv[1]);
        const size_t expected_raw_size = std::stoull(argv[2]);
        if (expected_raw_size == 0) {
            throw std::runtime_error("Expected raw size must be positive");
        }

        const cudaError_t stream_status = cudaStreamCreate(&stream);
        if (stream_status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(stream_status));
        }

        float total_ms = 0.0f;
        float codec_ms = 0.0f;
        const std::vector<uint8_t> raw =
            cprq_gpu_pool_flow::cprq_nvcomp_zstd_decompress_host_saved_bytes(
                saved,
                expected_raw_size,
                stream,
                &total_ms,
                &codec_ms);
        write_bytes(argv[3], raw);

        cudaStreamDestroy(stream);
        stream = nullptr;
        std::cout
            << "raw_package_bytes=" << raw.size() << "\n"
            << "nvcomp_total_ms=" << total_ms << "\n"
            << "nvcomp_codec_ms=" << codec_ms << "\n"
            << "raw_output=" << argv[3] << "\n";
        return 0;
    } catch (const std::exception& error) {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
