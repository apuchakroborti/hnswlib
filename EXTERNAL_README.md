# External Build and Run Notes for hnswlib C++ Examples

This repository is a header-only C++ HNSW implementation with Python bindings.
The C++ implementation lives under `hnswlib/`, and the runnable C++ examples
live under `examples/cpp/`.

The commands below assume you are at the repository root:

```bash
cd /home/achakroborti1/ann-benchmark/hnswlib/hnswlib
```

## 1. Project Layout

Important files and directories:

```text
hnswlib/                  Header-only HNSW implementation
examples/cpp/             C++ example programs
tests/cpp/                C++ test and benchmark-style programs
CMakeLists.txt            CMake build definition for examples and tests
ALGO_PARAMS.md            Short upstream notes about HNSW parameters
logs/                     Recommended directory for run logs
EXTERNAL_README.md        This file
```

The C++ examples are built by CMake when `HNSWLIB_EXAMPLES=ON`.

## 2. Build Requirements

You need:

```text
CMake
A C++11 compiler such as g++
OpenMP support for the multithreaded examples
```

On Ubuntu-like systems, the usual packages are:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
```

If you do not have `sudo` access, load or install a compiler and CMake through
your cluster or environment module system.

## 3. Clean CMake Build

Use a separate build directory so generated files do not mix with source files:

```bash
cmake -S . -B build_cpp -DHNSWLIB_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j8
```

The example binaries will be created in `build_cpp/`, for example:

```text
build_cpp/example_search
build_cpp/example_filter
build_cpp/example_replace_deleted
build_cpp/example_mt_search
build_cpp/example_mt_filter
build_cpp/example_mt_replace_deleted
build_cpp/example_epsilon_search
build_cpp/example_multivector_search
```

If you want a smaller build with only the interface library and no examples:

```bash
cmake -S . -B build_lib_only -DHNSWLIB_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build_lib_only -j8
```

## 4. Run the C++ Examples

Create the logging directory first:

```bash
mkdir -p logs
```

Run the basic single-threaded search example:

```bash
./build_cpp/example_search
```

Run it and save output to a log:

```bash
./build_cpp/example_search > logs/example_search.log 2>&1
```

Run all common examples one by one:

```bash
./build_cpp/example_search
./build_cpp/example_filter
./build_cpp/example_replace_deleted
./build_cpp/example_mt_search
./build_cpp/example_mt_filter
./build_cpp/example_mt_replace_deleted
./build_cpp/example_epsilon_search
./build_cpp/example_multivector_search

# Regular build
cmake --build build_cpp --target example_search -j8
# Run with time
timeout 30s ./build_cpp/example_search > /tmp/example_search_check.log 2>&1

# When need to choose AVX, AVX512, and SSE
# AVX
g++ -O3 -std=c++11 -mavx -I. examples/cpp/example_search.cpp -o examples/cpp/example_search_avx
./examples/cpp/example_search_avx > logs/example_search_avx.log 2>&1

# AVX512
g++ -O3 -std=c++11 -mavx512f -I. examples/cpp/example_search.cpp -o examples/cpp/example_search_avx512

# SSE
g++ -O3 -std=c++11 -msse4.2 -mno-avx -DNO_MANUAL_VECTORIZATION=0 -I. examples/cpp/example_search.cpp -o examples/cpp/example_search_sse
# or
g++ -O3 -std=c++11 -msse4.2 -mno-avx -I. examples/cpp/example_search.cpp -o examples/cpp/example_search_sse

# for GNU, so it automatically enables whatever your current CPU supports. To force specific behavior, edit CMakeLists.txt CXX flags or configure with custom flags, for example:
cmake -S . -B build_avx -DHNSWLIB_EXAMPLES=ON -DCMAKE_CXX_FLAGS="-O3 -std=c++11 -mavx"
cmake --build build_avx -j8

cmake -S . -B build_scalar -DHNSWLIB_EXAMPLES=ON -DCMAKE_CXX_FLAGS="-O3 -std=c++11 -DNO_MANUAL_VECTORIZATION"
cmake --build build_scalar -j8

# SSE
cmake -S . -B build_sse \
  -DHNSWLIB_EXAMPLES=ON \
  -DCMAKE_CXX_FLAGS="-O3 -std=c++11 -msse4.2 -mno-avx"

cmake --build build_sse -j8

# Run
./build_sse/example_search

```

Run each example with a separate log file:

```bash
./build_cpp/example_search > logs/example_search.log 2>&1
./build_cpp/example_filter > logs/example_filter.log 2>&1
./build_cpp/example_replace_deleted > logs/example_replace_deleted.log 2>&1
./build_cpp/example_mt_search > logs/example_mt_search.log 2>&1
./build_cpp/example_mt_filter > logs/example_mt_filter.log 2>&1
./build_cpp/example_mt_replace_deleted > logs/example_mt_replace_deleted.log 2>&1
./build_cpp/example_epsilon_search > logs/example_epsilon_search.log 2>&1
./build_cpp/example_multivector_search > logs/example_multivector_search.log 2>&1
```

## 5. Record Runtime and Memory in Logs

Use `/usr/bin/time -v` to capture wall-clock time, CPU usage, and peak resident
memory.  This is useful when comparing different HNSW parameters.

Example:

```bash
/usr/bin/time -v ./build_cpp/example_search > logs/example_search_run.log 2>&1
```

The log will contain both the program output, such as recall, and the timing
summary from `/usr/bin/time`.

Recommended naming pattern:

```bash
/usr/bin/time -v ./build_cpp/example_search > logs/example_search_M16_efc200_ef10.log 2>&1
```

Useful fields to inspect in the log:

```text
Recall
Recall of deserialized index
Elapsed (wall clock) time
User time
System time
Maximum resident set size
Percent of CPU this job got
```

## 6. Main HNSW Parameters in the C++ Examples

Most examples define parameters near the top of `main()`.

For example, `examples/cpp/example_search.cpp` contains:

```cpp
int dim = 16;
int max_elements = 10000;
int M = 16;
int ef_construction = 200;
```

The index is then constructed with:

```cpp
hnswlib::L2Space space(dim);
hnswlib::HierarchicalNSW<float>* alg_hnsw =
    new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
```

Meaning of the most important parameters:

```text
dim               Vector dimensionality.
max_elements      Maximum number of vectors the index can store.
M                 Graph connectivity parameter. Higher M usually improves
                  recall but increases memory and build time.
ef_construction   Candidate list width during graph construction. Higher values
                  usually improve index quality but increase build time.
ef                Candidate list width during query. Higher values usually
                  improve recall but slow query.
k                 Number of nearest neighbors requested by searchKnn.
num_threads       Number of CPU threads used by multithreaded examples.
random_seed       Seed for reproducible level assignment and random data.
allow_replace_deleted
                  Enables reusing deleted internal slots.
```

By default, the C++ HNSW class sets query `ef` to 10.  To test a different
query-time `ef`, call `setEf()` after creating or loading the index:

```cpp
alg_hnsw->setEf(50);
```

Place this before the query loop:

```cpp
hnswlib::HierarchicalNSW<float>* alg_hnsw =
    new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

alg_hnsw->setEf(50);
```

After loading a saved index, set `ef` again because the C++ load path resets
query `ef` to its default:

```cpp
alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
alg_hnsw->setEf(50);
```

## 7. Example Parameter Changes

### 7.1. Larger Graph, Better Recall, More Memory

Edit `examples/cpp/example_search.cpp`:

```cpp
int M = 32;
int ef_construction = 400;
```

Add:

```cpp
alg_hnsw->setEf(100);
```

Rebuild and run:

```bash
cmake --build build_cpp -j8
/usr/bin/time -v ./build_cpp/example_search > logs/example_search_M32_efc400_ef100.log 2>&1
```

### 7.2. Smaller Graph, Faster Build, Lower Memory

Edit:

```cpp
int M = 8;
int ef_construction = 100;
```

Add:

```cpp
alg_hnsw->setEf(20);
```

Rebuild and run:

```bash
cmake --build build_cpp -j8
/usr/bin/time -v ./build_cpp/example_search > logs/example_search_M8_efc100_ef20.log 2>&1
```

### 7.3. Larger Dataset

Edit:

```cpp
int max_elements = 100000;
```

Then rebuild and log:

```bash
cmake --build build_cpp -j8
/usr/bin/time -v ./build_cpp/example_search > logs/example_search_N100000.log 2>&1
```

### 7.4. Higher Dimensional Vectors

Edit:

```cpp
int dim = 128;
```

Higher-dimensional data often benefits from larger `M` and
`ef_construction`, for example:

```cpp
int M = 32;
int ef_construction = 400;
alg_hnsw->setEf(100);
```

Rebuild and log:

```bash
cmake --build build_cpp -j8
/usr/bin/time -v ./build_cpp/example_search > logs/example_search_dim128_M32_efc400_ef100.log 2>&1
```

### 7.5. Multithreaded Example

Edit `examples/cpp/example_mt_search.cpp`:

```cpp
int num_threads = 8;
int M = 16;
int ef_construction = 200;
```

Optionally add:

```cpp
alg_hnsw->setEf(50);
```

Rebuild and log:

```bash
cmake --build build_cpp -j8
/usr/bin/time -v ./build_cpp/example_mt_search > logs/example_mt_search_threads8_M16_efc200_ef50.log 2>&1
```

## 8. Fast Parameter Sweep by Copying an Example

Because the examples hardcode parameters, the cleanest way to compare settings
is to copy one example and edit the copy.

```bash
cp examples/cpp/example_search.cpp examples/cpp/example_search_tuned.cpp
```

Then add this target to `CMakeLists.txt` near the other examples:

```cmake
add_executable(example_search_tuned examples/cpp/example_search_tuned.cpp)
target_link_libraries(example_search_tuned hnswlib)
```

Now edit only `example_search_tuned.cpp`, rebuild, and run:

```bash
cmake --build build_cpp -j8
/usr/bin/time -v ./build_cpp/example_search_tuned > logs/example_search_tuned.log 2>&1
```

This keeps the original example unchanged.

## 9. Manual One-File Build Without CMake

Since hnswlib is header-only, a single example can also be compiled directly.

Basic example:

```bash
g++ -O3 -std=c++11 -march=native -I. examples/cpp/example_search.cpp -o /tmp/example_search
/usr/bin/time -v /tmp/example_search > logs/example_search_manual_gpp.log 2>&1
```

Multithreaded example:

```bash
g++ -O3 -std=c++11 -march=native -fopenmp -pthread -I. examples/cpp/example_mt_search.cpp -o /tmp/example_mt_search
/usr/bin/time -v /tmp/example_mt_search > logs/example_mt_search_manual_gpp.log 2>&1
```

Prefer CMake for regular experiments because it already knows all example and
test targets.

## 10. Suggested Experiment Log Table

For every run, record the parameter values in the log filename and optionally
in a separate notes file.

Suggested filename format:

```text
logs/<program>_N<max_elements>_dim<dim>_M<M>_efc<ef_construction>_ef<ef>_k<k>.log
```

Example:

```bash
/usr/bin/time -v ./build_cpp/example_search > logs/example_search_N10000_dim16_M16_efc200_ef10_k1.log 2>&1
```

Suggested metrics to compare:

```text
Build time
Query time
Recall
Maximum resident set size
Index file size, if saveIndex is used
Number of elements
Vector dimension
M
ef_construction
ef
k
num_threads
```

## 11. Running C++ Tests with Performance Output

Some files under `tests/cpp/` print more benchmark-style output.  CMake builds
several of them as targets.

Examples:

```bash
./build_cpp/test_updates
./build_cpp/epsilon_search_test
./build_cpp/multivector_search_test
```

Log them with:

```bash
/usr/bin/time -v ./build_cpp/test_updates > logs/test_updates.log 2>&1
/usr/bin/time -v ./build_cpp/epsilon_search_test > logs/epsilon_search_test.log 2>&1
/usr/bin/time -v ./build_cpp/multivector_search_test > logs/multivector_search_test.log 2>&1
```

`tests/cpp/updates_test.cpp` prints columns like:

```text
ef    recall    time    hops    distcomp
```

That test is useful when you want query-time recall and search-effort metrics
for multiple `ef` values.

## 12. Quick Recommended Workflow

For a clean run:

```bash
mkdir -p logs
cmake -S . -B build_cpp -DHNSWLIB_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j8
/usr/bin/time -v ./build_cpp/example_search > logs/example_search_baseline.log 2>&1
```

Then change `M`, `ef_construction`, `ef`, `dim`, or `max_elements`, rebuild,
and save the next run under a new descriptive log filename.
