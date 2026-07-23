// Ensures this header is included only once in each translation unit.
#pragma once

#include "hnswlib.h"

// Places the following declarations inside the hnswlib namespace.
namespace hnswlib {

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);
    // Assigns or initializes a value used by the algorithm state.
    float res = 0;
    // Iterates over a bounded range of values or elements.
    for (unsigned i = 0; i < qty; i++) {
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    // Closes the current C++ scope.
    }
    // Returns the computed value or object to the caller.
    return res;
// Closes the current C++ scope.
}

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    // Returns the computed value or object to the caller.
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
// Closes the current C++ scope.
}

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_AVX)

// Favor using AVX if available.
// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    
    float PORTABLE_ALIGN32 TmpRes[8];
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);

    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty / 16;
    // Assigns or initializes a value used by the algorithm state.
    size_t qty4 = qty / 4;

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd1 = pVect1 + 16 * qty16;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd2 = pVect1 + 4 * qty4;

    // Assigns or initializes a value used by the algorithm state.
    __m256 sum256 = _mm256_set1_ps(0);

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        // Assigns or initializes a value used by the algorithm state.
        __m256 v1 = _mm256_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 8;
        // Assigns or initializes a value used by the algorithm state.
        __m256 v2 = _mm256_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 8;
        // Assigns or initializes a value used by the algorithm state.
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm256_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 8;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm256_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 8;
        // Assigns or initializes a value used by the algorithm state.
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    // Closes the current C++ scope.
    }

    
    __m128 v1, v2;
    // Assigns or initializes a value used by the algorithm state.
    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd2) {
        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    // Closes the current C++ scope.
    }

    
    _mm_store_ps(TmpRes, sum_prod);
    // Assigns or initializes a value used by the algorithm state.
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    // Returns the computed value or object to the caller.
    return sum;
// Closes the current C++ scope.
}

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Returns the computed value or object to the caller.
    return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
// Closes the current C++ scope.
}

// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_SSE)

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    
    float PORTABLE_ALIGN32 TmpRes[8];
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);

    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty / 16;
    // Assigns or initializes a value used by the algorithm state.
    size_t qty4 = qty / 4;

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd1 = pVect1 + 16 * qty16;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd2 = pVect1 + 4 * qty4;

    
    __m128 v1, v2;
    // Assigns or initializes a value used by the algorithm state.
    __m128 sum_prod = _mm_set1_ps(0);

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd1) {
        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    // Closes the current C++ scope.
    }

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd2) {
        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    // Closes the current C++ scope.
    }

    
    _mm_store_ps(TmpRes, sum_prod);
    // Assigns or initializes a value used by the algorithm state.
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    // Returns the computed value or object to the caller.
    return sum;
// Closes the current C++ scope.
}

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Returns the computed value or object to the caller.
    return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
// Closes the current C++ scope.
}

// Ends the current conditional compilation block.
#endif


// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_AVX512)

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    
    float PORTABLE_ALIGN64 TmpRes[16];
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);

    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty / 16;


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd1 = pVect1 + 16 * qty16;

    // Assigns or initializes a value used by the algorithm state.
    __m512 sum512 = _mm512_set1_ps(0);

    // Assigns or initializes a value used by the algorithm state.
    size_t loop = qty16 / 4;
    
    // Repeats the block while the condition remains true.
    while (loop--) {
        // Assigns or initializes a value used by the algorithm state.
        __m512 v1 = _mm512_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        __m512 v2 = _mm512_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 16;
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 16;

        // Assigns or initializes a value used by the algorithm state.
        __m512 v3 = _mm512_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        __m512 v4 = _mm512_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 16;
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 16;

        // Assigns or initializes a value used by the algorithm state.
        __m512 v5 = _mm512_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        __m512 v6 = _mm512_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 16;
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 16;

        // Assigns or initializes a value used by the algorithm state.
        __m512 v7 = _mm512_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        __m512 v8 = _mm512_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 16;
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 16;

        // Assigns or initializes a value used by the algorithm state.
        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        // Assigns or initializes a value used by the algorithm state.
        sum512 = _mm512_fmadd_ps(v3, v4, sum512);
        // Assigns or initializes a value used by the algorithm state.
        sum512 = _mm512_fmadd_ps(v5, v6, sum512);
        // Assigns or initializes a value used by the algorithm state.
        sum512 = _mm512_fmadd_ps(v7, v8, sum512);
    // Closes the current C++ scope.
    }

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd1) {
        // Assigns or initializes a value used by the algorithm state.
        __m512 v1 = _mm512_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        __m512 v2 = _mm512_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 16;
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 16;
        // Assigns or initializes a value used by the algorithm state.
        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
    // Closes the current C++ scope.
    }

    // Assigns or initializes a value used by the algorithm state.
    float sum = _mm512_reduce_add_ps(sum512);
    // Returns the computed value or object to the caller.
    return sum;
// Closes the current C++ scope.
}

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Returns the computed value or object to the caller.
    return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
// Closes the current C++ scope.
}

// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_AVX)

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    
    float PORTABLE_ALIGN32 TmpRes[8];
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);

    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty / 16;


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd1 = pVect1 + 16 * qty16;

    // Assigns or initializes a value used by the algorithm state.
    __m256 sum256 = _mm256_set1_ps(0);

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        // Assigns or initializes a value used by the algorithm state.
        __m256 v1 = _mm256_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 8;
        // Assigns or initializes a value used by the algorithm state.
        __m256 v2 = _mm256_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 8;
        // Assigns or initializes a value used by the algorithm state.
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm256_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 8;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm256_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 8;
        // Assigns or initializes a value used by the algorithm state.
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    // Closes the current C++ scope.
    }

    
    _mm256_store_ps(TmpRes, sum256);
    // Assigns or initializes a value used by the algorithm state.
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    // Returns the computed value or object to the caller.
    return sum;
// Closes the current C++ scope.
}

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Returns the computed value or object to the caller.
    return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
// Closes the current C++ scope.
}

// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_SSE)

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    
    float PORTABLE_ALIGN32 TmpRes[8];
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);

    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty / 16;

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd1 = pVect1 + 16 * qty16;

    
    __m128 v1, v2;
    // Assigns or initializes a value used by the algorithm state.
    __m128 sum_prod = _mm_set1_ps(0);

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd1) {
        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    // Closes the current C++ scope.
    }
    
    _mm_store_ps(TmpRes, sum_prod);
    // Assigns or initializes a value used by the algorithm state.
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    // Returns the computed value or object to the caller.
    return sum;
// Closes the current C++ scope.
}

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Returns the computed value or object to the caller.
    return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
// Closes the current C++ scope.
}

// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
// Assigns or initializes a value used by the algorithm state.
static DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
// Assigns or initializes a value used by the algorithm state.
static DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
// Assigns or initializes a value used by the algorithm state.
static DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
// Assigns or initializes a value used by the algorithm state.
static DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);
    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty >> 4 << 4;
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v + qty16;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v + qty16;

    // Assigns or initializes a value used by the algorithm state.
    size_t qty_left = qty - qty16;
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    // Returns the computed value or object to the caller.
    return 1.0f - (res + res_tail);
// Closes the current C++ scope.
}

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);
    // Assigns or initializes a value used by the algorithm state.
    size_t qty4 = qty >> 2 << 2;

    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    // Assigns or initializes a value used by the algorithm state.
    size_t qty_left = qty - qty4;

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v + qty4;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v + qty4;
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    // Returns the computed value or object to the caller.
    return 1.0f - (res + res_tail);
// Closes the current C++ scope.
}
// Ends the current conditional compilation block.
#endif

// Declares a C++ type that groups related data and behavior.
class InnerProductSpace : public SpaceInterface<float> {
    
    DISTFUNC<float> fstdistfunc_;
    
    size_t data_size_;
    
    size_t dim_;

 // Sets the access level for the following class members.
 public:
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    InnerProductSpace(size_t dim) {
        // Assigns or initializes a value used by the algorithm state.
        fstdistfunc_ = InnerProductDistance;
// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    // Starts or continues a conditional compilation branch selected by compiler macros.
    #if defined(USE_AVX512)
        // Branches only when the condition is true.
        if (AVX512Capable()) {
            // Assigns or initializes a value used by the algorithm state.
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            // Assigns or initializes a value used by the algorithm state.
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else if (AVXCapable()) {
            // Assigns or initializes a value used by the algorithm state.
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            // Assigns or initializes a value used by the algorithm state.
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        // Closes the current C++ scope.
        }
    // Starts or continues a conditional compilation branch selected by compiler macros.
    #elif defined(USE_AVX)
        // Branches only when the condition is true.
        if (AVXCapable()) {
            // Assigns or initializes a value used by the algorithm state.
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            // Assigns or initializes a value used by the algorithm state.
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        // Closes the current C++ scope.
        }
    // Ends the current conditional compilation block.
    #endif
    // Starts or continues a conditional compilation branch selected by compiler macros.
    #if defined(USE_AVX)
        // Branches only when the condition is true.
        if (AVXCapable()) {
            // Assigns or initializes a value used by the algorithm state.
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
            // Assigns or initializes a value used by the algorithm state.
            InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
        // Closes the current C++ scope.
        }
    // Ends the current conditional compilation block.
    #endif

        // Branches only when the condition is true.
        if (dim % 16 == 0)
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        // Handles the alternative path when the previous condition was false.
        else if (dim % 4 == 0)
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        // Handles the alternative path when the previous condition was false.
        else if (dim > 16)
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        // Handles the alternative path when the previous condition was false.
        else if (dim > 4)
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
// Ends the current conditional compilation block.
#endif
        // Assigns or initializes a value used by the algorithm state.
        dim_ = dim;
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        data_size_ = dim * sizeof(float);
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    size_t get_data_size() {
        // Returns the computed value or object to the caller.
        return data_size_;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    DISTFUNC<float> get_dist_func() {
        // Returns the computed value or object to the caller.
        return fstdistfunc_;
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void *get_dist_func_param() {
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        return &dim_;
    // Closes the current C++ scope.
    }

// Executes this C++ statement as part of the surrounding HNSW implementation.
~InnerProductSpace() {}
// Closes the current class, struct, namespace, or block scope.
};

// Executes this C++ statement as part of the surrounding HNSW implementation.
}  // namespace hnswlib
