# Compressed CPRQ HNSW CPU Example

This directory is independent of the files under `hnswlib/hnswlib`.
It reuses the public `SpaceInterface` and `HierarchicalNSW` APIs without
changing the existing HNSW implementation.

## Build the CPU Example

From `cuda_cprq_imprv/hnswlib`:

```bash
cmake \
  -S hnsw_qlrpcomp \
  -B build_cprq_compressed \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_cprq_compressed --parallel
```

The target is:

```text
build_cprq_compressed/example_search_cprq_compressed
```

The CMake target uses `-march=native`, OpenMP, and `libzstd`. On the current
machine, the custom distance functions compile to AVX-512.

## Run

Arguments are:

```text
M ef_construction ef_search query_count compressed_package query_fvecs [index_output]
```

If `index_output` is omitted, the CMake-built executable saves the graph to
`cuda_cprq_imprv/hnswlib/hnsw_qlrpcomp/hnsw.bin`. The program reports the
saved size in bytes, decimal megabytes, and decimal gigabytes. Pass the final
argument to use a different path.

For the tested 11,520-vector, 3,584-dimensional package with `M=16`, the
compressed-node index occupied `49,035,428` bytes (`49.035428` MB,
`0.049035` GB). The comparable FP32-node example occupied `166,861,988`
bytes (`166.861988` MB, `0.166862` GB).

Example using the currently available nvCOMP-native Zstd package:

```bash
mkdir -p logs

OMP_NUM_THREADS=16 \
./build_cprq_compressed/example_search_cprq_compressed \
  16 \
  200 \
  200 \
  1000 \
  '/tmp/qlrpcomp_gpu_pool_mean_rSVD_model_components_r64_qb_8_abs0905/embeds_0[(16,20,36,3584)]_w_encoding_full_package/w_encoding_full_package_zstd.bin' \
  '/tmp/qlrpcomp_gpu_pool_kmeans_model_components_r64_qb_8_abs0905_decompressed/embeds_0[(16,20,36,3584)].fvecs' \
  > logs/example_search_cprq_compressed.log 2>&1

# Single threaded abs:0.0905
OMP_NUM_THREADS=1 \
./build_cprq_compressed/example_search_cprq_compressed \
  16 \
  200 \
  200 \
  1000 \
  '/tmp/qlrpcomp_gpu_pool_mean_rSVD_model_components_r64_qb_8_abs0905/embeds_0[(16,20,36,3584)]_w_encoding_full_package/w_encoding_full_package_zstd.bin' \
  '/tmp/qlrpcomp_gpu_pool_mean_rSVD_model_components_r64_qb_8_abs0905_decompressed/embeds_0[(16,20,36,3584)].fvecs' \
  > logs/example_search_cprq_compressed_OMP_NUM_THREADS_1.log 2>&1

# Single threaded abs:0.905
OMP_NUM_THREADS=1 \
./build_cprq_compressed/example_search_cprq_compressed \
  16 \
  200 \
  200 \
  1000 \
  '/tmp/qlrpcomp_gpu_pool_mean_rSVD_model_components_r64_qb_8_abs905/embeds_0[(16,20,36,3584)]_w_encoding_full_package/w_encoding_full_package_zstd.bin' \
  '/tmp/qlrpcomp_gpu_pool_mean_rSVD_model_components_r64_qb_8_abs905_decompressed/embeds_0[(16,20,36,3584)].fvecs' \
  > logs/example_search_cprq_compressed_OMP_NUM_THREADS_1.log 2>&1

# Single threaded abs:0.905 & from the real data
OMP_NUM_THREADS=1 \
./build_cprq_compressed/example_search_cprq_compressed \
  16 \
  200 \
  200 \
  1000 \
  '/tmp/qlrpcomp_gpu_pool_mean_rSVD_model_components_r64_qb_8_abs905/embeds_0[(16,20,36,3584)]_w_encoding_full_package/w_encoding_full_package_zstd.bin' \
  '../../data/extra3_real_bandwidth_case_study/cprq/Qwen_Qwen2.5-VL-7B-Instruct/work/cprq_gpu/input_embedding_input_fvecs/embeds_0[(16,20,36,3584)].fvecs' \
  > logs/example_search_cprq_compressed_OMP_NUM_THREADS_1.log 2>&1


```

The CPU loader accepts:

- A raw W-encoding full package.
- The current nvCOMP-native chunked Zstd package.
- A standard single-frame Zstd package.

## Export the Raw Package with the Existing QLRPComp Helper

`export_nvcomp_package_raw.cpp` calls:

```cpp
cprq_gpu_pool_flow::cprq_nvcomp_zstd_decompress_host_saved_bytes(...)
```

Build the converter:

```bash
chmod +x hnsw_qlrpcomp/build_nvcomp_exporter.sh
./hnsw_qlrpcomp/build_nvcomp_exporter.sh
```

Export the package. The expected raw size comes from
`w_encoding_full_package.meta`.

```bash
./hnsw_qlrpcomp/export_nvcomp_package_raw \
  '/tmp/qlrpcomp_gpu_pool_mean_rSVD_model_components_r64_qb_8_abs0905/embeds_0[(16,20,36,3584)]_w_encoding_full_package/w_encoding_full_package_zstd.bin' \
  22492399 \
  '/tmp/qlrpcomp_gpu_pool_mean_rSVD_model_components_r64_qb_8_abs0905/embeds_0[(16,20,36,3584)]_w_encoding_full_package/w_encoding_full_package_raw.bin'
```

This conversion requires a visible CUDA device because the existing helper
uses nvCOMP on the GPU. The HNSW example itself does not require a GPU.

## Reported Measurements

The program reports:

- Package decoding, QR, transformed-alpha, and beta preparation time.
- Clipping counts and the largest decoded residual magnitude.
- HNSW construction time and insertion throughput.
- Query preparation time.
- HNSW-only search time and QPS.
- End-to-end query time and QPS.
- Direct-reconstruction checks for query-to-node and node-to-node distances.
