cmake_minimum_required(VERSION 3.12)

project(cprq_compressed_hnsw_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
find_library(ZSTD_LIBRARY zstd REQUIRED)

add_executable(
    example_search_cprq_compressed
    example_search_cprq_compressed.cpp)

target_include_directories(
    example_search_cprq_compressed
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_compile_options(
    example_search_cprq_compressed
    PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic)

target_compile_definitions(
    example_search_cprq_compressed
    PRIVATE
    CPRQ_HNSW_DEFAULT_INDEX_OUTPUT="${CMAKE_CURRENT_SOURCE_DIR}/hnsw.bin")

target_link_libraries(
    example_search_cprq_compressed
    PRIVATE
    OpenMP::OpenMP_CXX
    ${ZSTD_LIBRARY})
