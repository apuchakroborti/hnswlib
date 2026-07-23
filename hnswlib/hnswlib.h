// Ensures this header is included only once in each translation unit.
#pragma once

// https://github.com/nmslib/hnswlib/pull/508
// This allows others to provide their own error stream (e.g. RcppHNSW)
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifndef HNSWLIB_ERR_OVERRIDE
  // Defines a preprocessor macro used by later compile-time checks or code.
  #define HNSWERR std::cerr
// Selects the fallback branch of the compile-time condition.
#else
  // Defines a preprocessor macro used by later compile-time checks or code.
  #define HNSWERR HNSWLIB_ERR_OVERRIDE
// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#ifndef NO_MANUAL_VECTORIZATION
// Starts or continues a conditional compilation branch selected by compiler macros.
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
// Defines a preprocessor macro used by later compile-time checks or code.
#define USE_SSE
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef __AVX__
// Defines a preprocessor macro used by later compile-time checks or code.
#define USE_AVX
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef __AVX512F__
// Defines a preprocessor macro used by later compile-time checks or code.
#define USE_AVX512
// Ends the current conditional compilation block.
#endif
// Ends the current conditional compilation block.
#endif
// Ends the current conditional compilation block.
#endif
// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_AVX) || defined(USE_SSE)
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef _MSC_VER

#include <intrin.h>

#include <stdexcept>
// Executes this C++ statement as part of the surrounding HNSW implementation.
static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    
    __cpuidex(out, eax, ecx);
// Closes the current C++ scope.
}
// Executes this C++ statement as part of the surrounding HNSW implementation.
static __int64 xgetbv(unsigned int x) {
    // Returns the computed value or object to the caller.
    return _xgetbv(x);
// Closes the current C++ scope.
}
// Selects the fallback branch of the compile-time condition.
#else

#include <x86intrin.h>

#include <cpuid.h>

#include <stdint.h>
// Executes this C++ statement as part of the surrounding HNSW implementation.
static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
// Closes the current C++ scope.
}
// Executes this C++ statement as part of the surrounding HNSW implementation.
static uint64_t xgetbv(unsigned int index) {
    
    uint32_t eax, edx;
    // Assigns or initializes a value used by the algorithm state.
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    // Returns the computed value or object to the caller.
    return ((uint64_t)edx << 32) | eax;
// Closes the current C++ scope.
}
// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_AVX512)

#include <immintrin.h>
// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(__GNUC__)
// Defines a preprocessor macro used by later compile-time checks or code.
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
// Defines a preprocessor macro used by later compile-time checks or code.
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
// Selects the fallback branch of the compile-time condition.
#else
// Defines a preprocessor macro used by later compile-time checks or code.
#define PORTABLE_ALIGN32 __declspec(align(32))
// Defines a preprocessor macro used by later compile-time checks or code.
#define PORTABLE_ALIGN64 __declspec(align(64))
// Ends the current conditional compilation block.
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
// Defines a preprocessor macro used by later compile-time checks or code.
#define _XCR_XFEATURE_ENABLED_MASK  0

// Executes this C++ statement as part of the surrounding HNSW implementation.
static bool AVXCapable() {
    
    int cpuInfo[4];

    // CPU support
    
    cpuid(cpuInfo, 0, 0);
    // Assigns or initializes a value used by the algorithm state.
    int nIds = cpuInfo[0];

    // Assigns or initializes a value used by the algorithm state.
    bool HW_AVX = false;
    // Branches only when the condition is true.
    if (nIds >= 0x00000001) {
        
        cpuid(cpuInfo, 0x00000001, 0);
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
    // Closes the current C++ scope.
    }

    // OS support
    
    cpuid(cpuInfo, 1, 0);

    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    // Assigns or initializes a value used by the algorithm state.
    bool avxSupported = false;
    // Branches only when the condition is true.
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        // Assigns or initializes a value used by the algorithm state.
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    // Closes the current C++ scope.
    }
    // Returns the computed value or object to the caller.
    return HW_AVX && avxSupported;
// Closes the current C++ scope.
}

// Executes this C++ statement as part of the surrounding HNSW implementation.
static bool AVX512Capable() {
    // Branches only when the condition is true.
    if (!AVXCapable()) return false;

    
    int cpuInfo[4];

    // CPU support
    
    cpuid(cpuInfo, 0, 0);
    // Assigns or initializes a value used by the algorithm state.
    int nIds = cpuInfo[0];

    // Assigns or initializes a value used by the algorithm state.
    bool HW_AVX512F = false;
    // Branches only when the condition is true.
    if (nIds >= 0x00000007) {  //  AVX512 Foundation
        
        cpuid(cpuInfo, 0x00000007, 0);
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
    // Closes the current C++ scope.
    }

    // OS support
    
    cpuid(cpuInfo, 1, 0);

    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    // Assigns or initializes a value used by the algorithm state.
    bool avx512Supported = false;
    // Branches only when the condition is true.
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        // Assigns or initializes a value used by the algorithm state.
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    // Closes the current C++ scope.
    }
    // Returns the computed value or object to the caller.
    return HW_AVX512F && avx512Supported;
// Closes the current C++ scope.
}
// Ends the current conditional compilation block.
#endif


#include <queue>

#include <vector>

#include <iostream>

#include <string.h>

// Places the following declarations inside the hnswlib namespace.
namespace hnswlib {
// Creates a shorter type alias used by the implementation.
typedef size_t labeltype;

// This can be extended to store state for filtering (e.g. from a std::set)
// Declares a C++ type that groups related data and behavior.
class BaseFilterFunctor {
 // Sets the access level for the following class members.
 public:
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    virtual bool operator()(hnswlib::labeltype id) { return true; }
    
    virtual ~BaseFilterFunctor() {};
// Closes the current class, struct, namespace, or block scope.
};

// Declares that the following class or function is templated over one or more types.
template<typename dist_t>
// Declares a C++ type that groups related data and behavior.
class BaseSearchStopCondition {
 // Sets the access level for the following class members.
 public:
    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    virtual void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    virtual void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    // Assigns or initializes a value used by the algorithm state.
    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

    // Assigns or initializes a value used by the algorithm state.
    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

    // Assigns or initializes a value used by the algorithm state.
    virtual bool should_remove_extra() = 0;

    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    virtual void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) = 0;

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    virtual ~BaseSearchStopCondition() {}
// Closes the current class, struct, namespace, or block scope.
};

// Declares that the following class or function is templated over one or more types.
template <typename T>
// Declares a C++ type that groups related data and behavior.
class pairGreater {
 // Sets the access level for the following class members.
 public:
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    bool operator()(const T& p1, const T& p2) {
        // Returns the computed value or object to the caller.
        return p1.first > p2.first;
    // Closes the current C++ scope.
    }
// Closes the current class, struct, namespace, or block scope.
};

// Declares that the following class or function is templated over one or more types.
template<typename T>
// Uses a reference or address operation so the code can avoid copying or can pass an address onward.
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    out.write((char *) &podRef, sizeof(T));
// Closes the current C++ scope.
}

// Declares that the following class or function is templated over one or more types.
template<typename T>
// Uses a reference or address operation so the code can avoid copying or can pass an address onward.
static void readBinaryPOD(std::istream &in, T &podRef) {
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    in.read((char *) &podRef, sizeof(T));
// Closes the current C++ scope.
}

// Declares that the following class or function is templated over one or more types.
template<typename MTYPE>
// Creates a shorter type alias used by the implementation.
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

// Declares that the following class or function is templated over one or more types.
template<typename MTYPE>
// Declares a C++ type that groups related data and behavior.
class SpaceInterface {
 // Sets the access level for the following class members.
 public:
    // virtual void search(void *);
    // Assigns or initializes a value used by the algorithm state.
    virtual size_t get_data_size() = 0;

    // Assigns or initializes a value used by the algorithm state.
    virtual DISTFUNC<MTYPE> get_dist_func() = 0;

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    virtual void *get_dist_func_param() = 0;

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    virtual ~SpaceInterface() {}
// Closes the current class, struct, namespace, or block scope.
};

// Declares that the following class or function is templated over one or more types.
template<typename dist_t>
// Declares a C++ type that groups related data and behavior.
class AlgorithmInterface {
 // Sets the access level for the following class members.
 public:
    // Manages heap memory; pointers here own or reference dynamically allocated storage.
    virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;

    // Uses a heap-backed priority queue to keep best candidates ordered by distance.
    virtual std::priority_queue<std::pair<dist_t, labeltype>>
        // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
        searchKnn(const void*, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closer fist
    // Uses a dynamic array to store a resizable sequence of values.
    virtual std::vector<std::pair<dist_t, labeltype>>
        // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
        searchKnnCloserFirst(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;

    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    virtual void saveIndex(const std::string &location) = 0;
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    virtual ~AlgorithmInterface(){
    // Closes the current C++ scope.
    }
// Closes the current class, struct, namespace, or block scope.
};

// Declares that the following class or function is templated over one or more types.
template<typename dist_t>
// Uses a dynamic array to store a resizable sequence of values.
std::vector<std::pair<dist_t, labeltype>>
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,
                                                 // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                                                 BaseFilterFunctor* isIdAllowed) const {
    // Uses a dynamic array to store a resizable sequence of values.
    std::vector<std::pair<dist_t, labeltype>> result;

    // here searchKnn returns the result in the order of further first
    // Assigns or initializes a value used by the algorithm state.
    auto ret = searchKnn(query_data, k, isIdAllowed);
    // Opens a new C++ scope for the preceding declaration or control statement.
    {
        // Assigns or initializes a value used by the algorithm state.
        size_t sz = ret.size();
        
        result.resize(sz);
        // Repeats the block while the condition remains true.
        while (!ret.empty()) {
            // Assigns or initializes a value used by the algorithm state.
            result[--sz] = ret.top();
            
            ret.pop();
        // Closes the current C++ scope.
        }
    // Closes the current C++ scope.
    }

    // Returns the computed value or object to the caller.
    return result;
// Closes the current C++ scope.
}
// Executes this C++ statement as part of the surrounding HNSW implementation.
}  // namespace hnswlib


#include "space_l2.h"

#include "space_ip.h"

#include "stop_condition.h"

#include "bruteforce.h"

#include "hnswalg.h"
