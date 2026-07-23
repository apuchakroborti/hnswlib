// Ensures this header is included only once in each translation unit.
#pragma once

#include "hnswlib.h"
// #include <iostream>

// Places the following declarations inside the hnswlib namespace.
namespace hnswlib {

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);

    // Assigns or initializes a value used by the algorithm state.
    float res = 0;
    // Iterates over a bounded range of values or elements.
    for (size_t i = 0; i < qty; i++) {
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        float t = *pVect1 - *pVect2;
        
        pVect1++;
        
        pVect2++;
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        res += t * t;
    // Closes the current C++ scope.
    }
    // Returns the computed value or object to the caller.
    return (res);
// Closes the current C++ scope.
}

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_AVX512)

// Favor using AVX512 if available.
// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);
    
    float PORTABLE_ALIGN64 TmpRes[16];
    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty >> 4;

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd1 = pVect1 + (qty16 << 4);

    
    __m512 diff, v1, v2;
    // Assigns or initializes a value used by the algorithm state.
    __m512 sum = _mm512_set1_ps(0);

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd1) {
        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm512_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 16;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm512_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 16;
        // Assigns or initializes a value used by the algorithm state.
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        // Assigns or initializes a value used by the algorithm state.
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    // Closes the current C++ scope.
    }

    
    _mm512_store_ps(TmpRes, sum);
    // Assigns or initializes a value used by the algorithm state.
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    // Returns the computed value or object to the caller.
    return (res);
// Closes the current C++ scope.
}
// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_AVX)

// Favor using AVX if available.
// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);
    
    float PORTABLE_ALIGN32 TmpRes[8];
    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty >> 4;

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd1 = pVect1 + (qty16 << 4);

    
    __m256 diff, v1, v2;
    // Assigns or initializes a value used by the algorithm state.
    __m256 sum = _mm256_set1_ps(0);

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd1) {
        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm256_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 8;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm256_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 8;
        // Assigns or initializes a value used by the algorithm state.
        diff = _mm256_sub_ps(v1, v2);
        // Assigns or initializes a value used by the algorithm state.
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm256_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 8;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm256_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 8;
        // Assigns or initializes a value used by the algorithm state.
        diff = _mm256_sub_ps(v1, v2);
        // Assigns or initializes a value used by the algorithm state.
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    // Closes the current C++ scope.
    }

    
    _mm256_store_ps(TmpRes, sum);
    // Returns the computed value or object to the caller.
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
// Closes the current C++ scope.
}

// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_SSE)

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);
    
    float PORTABLE_ALIGN32 TmpRes[8];
    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty >> 4;

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd1 = pVect1 + (qty16 << 4);

    
    __m128 diff, v1, v2;
    // Assigns or initializes a value used by the algorithm state.
    __m128 sum = _mm_set1_ps(0);

    // Repeats the block while the condition remains true.
    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        diff = _mm_sub_ps(v1, v2);
        // Assigns or initializes a value used by the algorithm state.
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        diff = _mm_sub_ps(v1, v2);
        // Assigns or initializes a value used by the algorithm state.
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        diff = _mm_sub_ps(v1, v2);
        // Assigns or initializes a value used by the algorithm state.
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        // Assigns or initializes a value used by the algorithm state.
        v1 = _mm_loadu_ps(pVect1);
        // Assigns or initializes a value used by the algorithm state.
        pVect1 += 4;
        // Assigns or initializes a value used by the algorithm state.
        v2 = _mm_loadu_ps(pVect2);
        // Assigns or initializes a value used by the algorithm state.
        pVect2 += 4;
        // Assigns or initializes a value used by the algorithm state.
        diff = _mm_sub_ps(v1, v2);
        // Assigns or initializes a value used by the algorithm state.
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    // Closes the current C++ scope.
    }

    
    _mm_store_ps(TmpRes, sum);
    // Returns the computed value or object to the caller.
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
// Closes the current C++ scope.
}
// Ends the current conditional compilation block.
#endif

// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
// Assigns or initializes a value used by the algorithm state.
static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);
    // Assigns or initializes a value used by the algorithm state.
    size_t qty16 = qty >> 4 << 4;
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v + qty16;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v + qty16;

    // Assigns or initializes a value used by the algorithm state.
    size_t qty_left = qty - qty16;
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    // Returns the computed value or object to the caller.
    return (res + res_tail);
// Closes the current C++ scope.
}
// Ends the current conditional compilation block.
#endif


// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_SSE)
// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v;
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);


    // Assigns or initializes a value used by the algorithm state.
    size_t qty4 = qty >> 2;

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    const float *pEnd1 = pVect1 + (qty4 << 2);

    
    __m128 diff, v1, v2;
    // Assigns or initializes a value used by the algorithm state.
    __m128 sum = _mm_set1_ps(0);

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
        diff = _mm_sub_ps(v1, v2);
        // Assigns or initializes a value used by the algorithm state.
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    // Closes the current C++ scope.
    }
    
    _mm_store_ps(TmpRes, sum);
    // Returns the computed value or object to the caller.
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
// Closes the current C++ scope.
}

// Executes this C++ statement as part of the surrounding HNSW implementation.
static float
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);
    // Assigns or initializes a value used by the algorithm state.
    size_t qty4 = qty >> 2 << 2;

    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    // Assigns or initializes a value used by the algorithm state.
    size_t qty_left = qty - qty4;

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect1 = (float *) pVect1v + qty4;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    float *pVect2 = (float *) pVect2v + qty4;
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    // Returns the computed value or object to the caller.
    return (res + res_tail);
// Closes the current C++ scope.
}
// Ends the current conditional compilation block.
#endif

// Declares a C++ type that groups related data and behavior.
class L2Space : public SpaceInterface<float> {
    
    DISTFUNC<float> fstdistfunc_;
    
    size_t data_size_;
    
    size_t dim_;

 // Sets the access level for the following class members.
 public:
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    L2Space(size_t dim) {
        // Assigns or initializes a value used by the algorithm state.
        fstdistfunc_ = L2Sqr;
// Starts or continues a conditional compilation branch selected by compiler macros.
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    // Starts or continues a conditional compilation branch selected by compiler macros.
    #if defined(USE_AVX512)
        // Branches only when the condition is true.
        if (AVX512Capable())
            // Assigns or initializes a value used by the algorithm state.
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        // Handles the alternative path when the previous condition was false.
        else if (AVXCapable())
            // Assigns or initializes a value used by the algorithm state.
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    // Starts or continues a conditional compilation branch selected by compiler macros.
    #elif defined(USE_AVX)
        // Branches only when the condition is true.
        if (AVXCapable())
            // Assigns or initializes a value used by the algorithm state.
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    // Ends the current conditional compilation block.
    #endif

        // Branches only when the condition is true.
        if (dim % 16 == 0)
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = L2SqrSIMD16Ext;
        // Handles the alternative path when the previous condition was false.
        else if (dim % 4 == 0)
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = L2SqrSIMD4Ext;
        // Handles the alternative path when the previous condition was false.
        else if (dim > 16)
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        // Handles the alternative path when the previous condition was false.
        else if (dim > 4)
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
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
    ~L2Space() {}
// Closes the current class, struct, namespace, or block scope.
};

// Executes this C++ statement as part of the surrounding HNSW implementation.
static int
// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t *) qty_ptr);
    // Assigns or initializes a value used by the algorithm state.
    int res = 0;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    unsigned char *a = (unsigned char *) pVect1;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    unsigned char *b = (unsigned char *) pVect2;

    // Assigns or initializes a value used by the algorithm state.
    qty = qty >> 2;
    // Iterates over a bounded range of values or elements.
    for (size_t i = 0; i < qty; i++) {
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        res += ((*a) - (*b)) * ((*a) - (*b));
        
        a++;
        
        b++;
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        res += ((*a) - (*b)) * ((*a) - (*b));
        
        a++;
        
        b++;
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        res += ((*a) - (*b)) * ((*a) - (*b));
        
        a++;
        
        b++;
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        res += ((*a) - (*b)) * ((*a) - (*b));
        
        a++;
        
        b++;
    // Closes the current C++ scope.
    }
    // Returns the computed value or object to the caller.
    return (res);
// Closes the current C++ scope.
}

// Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    size_t qty = *((size_t*)qty_ptr);
    // Assigns or initializes a value used by the algorithm state.
    int res = 0;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    unsigned char* a = (unsigned char*)pVect1;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    unsigned char* b = (unsigned char*)pVect2;

    // Iterates over a bounded range of values or elements.
    for (size_t i = 0; i < qty; i++) {
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        res += ((*a) - (*b)) * ((*a) - (*b));
        
        a++;
        
        b++;
    // Closes the current C++ scope.
    }
    // Returns the computed value or object to the caller.
    return (res);
// Closes the current C++ scope.
}

// Declares a C++ type that groups related data and behavior.
class L2SpaceI : public SpaceInterface<int> {
    
    DISTFUNC<int> fstdistfunc_;
    
    size_t data_size_;
    
    size_t dim_;

 // Sets the access level for the following class members.
 public:
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    L2SpaceI(size_t dim) {
        // Branches only when the condition is true.
        if (dim % 4 == 0) {
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = L2SqrI4x;
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else {
            // Assigns or initializes a value used by the algorithm state.
            fstdistfunc_ = L2SqrI;
        // Closes the current C++ scope.
        }
        // Assigns or initializes a value used by the algorithm state.
        dim_ = dim;
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        data_size_ = dim * sizeof(unsigned char);
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    size_t get_data_size() {
        // Returns the computed value or object to the caller.
        return data_size_;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    DISTFUNC<int> get_dist_func() {
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
    ~L2SpaceI() {}
// Closes the current class, struct, namespace, or block scope.
};
// Executes this C++ statement as part of the surrounding HNSW implementation.
}  // namespace hnswlib
