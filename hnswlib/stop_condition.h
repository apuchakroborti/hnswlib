// Ensures this header is included only once in each translation unit.
#pragma once

#include "space_l2.h"

#include "space_ip.h"

#include <assert.h>

#include <unordered_map>

// Places the following declarations inside the hnswlib namespace.
namespace hnswlib {

// Declares that the following class or function is templated over one or more types.
template<typename DOCIDTYPE>
// Declares a C++ type that groups related data and behavior.
class BaseMultiVectorSpace : public SpaceInterface<float> {
 // Sets the access level for the following class members.
 public:
    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    virtual DOCIDTYPE get_doc_id(const void *datapoint) = 0;

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    virtual void set_doc_id(void *datapoint, DOCIDTYPE doc_id) = 0;
// Closes the current class, struct, namespace, or block scope.
};


// Declares that the following class or function is templated over one or more types.
template<typename DOCIDTYPE>
// Declares a C++ type that groups related data and behavior.
class MultiVectorL2Space : public BaseMultiVectorSpace<DOCIDTYPE> {
    
    DISTFUNC<float> fstdistfunc_;
    
    size_t data_size_;
    
    size_t vector_size_;
    
    size_t dim_;

 // Sets the access level for the following class members.
 public:
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    MultiVectorL2Space(size_t dim) {
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
        vector_size_ = dim * sizeof(float);
        // Assigns or initializes a value used by the algorithm state.
        data_size_ = vector_size_ + sizeof(DOCIDTYPE);
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    size_t get_data_size() override {
        // Returns the computed value or object to the caller.
        return data_size_;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    DISTFUNC<float> get_dist_func() override {
        // Returns the computed value or object to the caller.
        return fstdistfunc_;
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void *get_dist_func_param() override {
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        return &dim_;
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    DOCIDTYPE get_doc_id(const void *datapoint) override {
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        return *(DOCIDTYPE *)((char *)datapoint + vector_size_);
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void set_doc_id(void *datapoint, DOCIDTYPE doc_id) override {
        *(DOCIDTYPE*)((char *)datapoint + vector_size_) = doc_id;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    ~MultiVectorL2Space() {}
// Closes the current class, struct, namespace, or block scope.
};


// Declares that the following class or function is templated over one or more types.
template<typename DOCIDTYPE>
// Declares a C++ type that groups related data and behavior.
class MultiVectorInnerProductSpace : public BaseMultiVectorSpace<DOCIDTYPE> {
    
    DISTFUNC<float> fstdistfunc_;
    
    size_t data_size_;
    
    size_t vector_size_;
    
    size_t dim_;

 // Sets the access level for the following class members.
 public:
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    MultiVectorInnerProductSpace(size_t dim) {
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
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        vector_size_ = dim * sizeof(float);
        // Assigns or initializes a value used by the algorithm state.
        data_size_ = vector_size_ + sizeof(DOCIDTYPE);
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    size_t get_data_size() override {
        // Returns the computed value or object to the caller.
        return data_size_;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    DISTFUNC<float> get_dist_func() override {
        // Returns the computed value or object to the caller.
        return fstdistfunc_;
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void *get_dist_func_param() override {
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        return &dim_;
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    DOCIDTYPE get_doc_id(const void *datapoint) override {
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        return *(DOCIDTYPE *)((char *)datapoint + vector_size_);
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void set_doc_id(void *datapoint, DOCIDTYPE doc_id) override {
        *(DOCIDTYPE*)((char *)datapoint + vector_size_) = doc_id;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    ~MultiVectorInnerProductSpace() {}
// Closes the current class, struct, namespace, or block scope.
};


// Declares that the following class or function is templated over one or more types.
template<typename DOCIDTYPE, typename dist_t>
// Declares a C++ type that groups related data and behavior.
class MultiVectorSearchStopCondition : public BaseSearchStopCondition<dist_t> {
    
    size_t curr_num_docs_;
    
    size_t num_docs_to_search_;
    
    size_t ef_collection_;
    
    std::unordered_map<DOCIDTYPE, size_t> doc_counter_;
    // Uses a heap-backed priority queue to keep best candidates ordered by distance.
    std::priority_queue<std::pair<dist_t, DOCIDTYPE>> search_results_;
    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    BaseMultiVectorSpace<DOCIDTYPE>& space_;

 // Sets the access level for the following class members.
 public:
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    MultiVectorSearchStopCondition(
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        BaseMultiVectorSpace<DOCIDTYPE>& space,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        size_t num_docs_to_search,
        // Assigns or initializes a value used by the algorithm state.
        size_t ef_collection = 10)
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        : space_(space) {
            // Assigns or initializes a value used by the algorithm state.
            curr_num_docs_ = 0;
            // Assigns or initializes a value used by the algorithm state.
            num_docs_to_search_ = num_docs_to_search;
            // Assigns or initializes a value used by the algorithm state.
            ef_collection_ = std::max(ef_collection, num_docs_to_search);
        // Closes the current C++ scope.
        }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) override {
        // Assigns or initializes a value used by the algorithm state.
        DOCIDTYPE doc_id = space_.get_doc_id(datapoint);
        // Branches only when the condition is true.
        if (doc_counter_[doc_id] == 0) {
            // Assigns or initializes a value used by the algorithm state.
            curr_num_docs_ += 1;
        // Closes the current C++ scope.
        }
        
        search_results_.emplace(dist, doc_id);
        // Assigns or initializes a value used by the algorithm state.
        doc_counter_[doc_id] += 1;
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) override {
        // Assigns or initializes a value used by the algorithm state.
        DOCIDTYPE doc_id = space_.get_doc_id(datapoint);
        // Assigns or initializes a value used by the algorithm state.
        doc_counter_[doc_id] -= 1;
        // Branches only when the condition is true.
        if (doc_counter_[doc_id] == 0) {
            // Assigns or initializes a value used by the algorithm state.
            curr_num_docs_ -= 1;
        // Closes the current C++ scope.
        }
        
        search_results_.pop();
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) override {
        // Assigns or initializes a value used by the algorithm state.
        bool stop_search = candidate_dist > lowerBound && curr_num_docs_ == ef_collection_;
        // Returns the computed value or object to the caller.
        return stop_search;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) override {
        // Assigns or initializes a value used by the algorithm state.
        bool flag_consider_candidate = curr_num_docs_ < ef_collection_ || lowerBound > candidate_dist;
        // Returns the computed value or object to the caller.
        return flag_consider_candidate;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    bool should_remove_extra() override {
        // Assigns or initializes a value used by the algorithm state.
        bool flag_remove_extra = curr_num_docs_ > ef_collection_;
        // Returns the computed value or object to the caller.
        return flag_remove_extra;
    // Closes the current C++ scope.
    }

    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) override {
        // Repeats the block while the condition remains true.
        while (curr_num_docs_ > num_docs_to_search_) {
            // Assigns or initializes a value used by the algorithm state.
            dist_t dist_cand = candidates.back().first;
            // Assigns or initializes a value used by the algorithm state.
            dist_t dist_res = search_results_.top().first;
            // Assigns or initializes a value used by the algorithm state.
            assert(dist_cand == dist_res);
            // Assigns or initializes a value used by the algorithm state.
            DOCIDTYPE doc_id = search_results_.top().second;
            // Assigns or initializes a value used by the algorithm state.
            doc_counter_[doc_id] -= 1;
            // Branches only when the condition is true.
            if (doc_counter_[doc_id] == 0) {
                // Assigns or initializes a value used by the algorithm state.
                curr_num_docs_ -= 1;
            // Closes the current C++ scope.
            }
            
            search_results_.pop();
            
            candidates.pop_back();
        // Closes the current C++ scope.
        }
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    ~MultiVectorSearchStopCondition() {}
// Closes the current class, struct, namespace, or block scope.
};


// Declares that the following class or function is templated over one or more types.
template<typename dist_t>
// Declares a C++ type that groups related data and behavior.
class EpsilonSearchStopCondition : public BaseSearchStopCondition<dist_t> {
    
    float epsilon_;
    
    size_t min_num_candidates_;
    
    size_t max_num_candidates_;
    
    size_t curr_num_items_;

 // Sets the access level for the following class members.
 public:
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    EpsilonSearchStopCondition(float epsilon, size_t min_num_candidates, size_t max_num_candidates) {
        // Assigns or initializes a value used by the algorithm state.
        assert(min_num_candidates <= max_num_candidates);
        // Assigns or initializes a value used by the algorithm state.
        epsilon_ = epsilon;
        // Assigns or initializes a value used by the algorithm state.
        min_num_candidates_ = min_num_candidates;
        // Assigns or initializes a value used by the algorithm state.
        max_num_candidates_ = max_num_candidates;
        // Assigns or initializes a value used by the algorithm state.
        curr_num_items_ = 0;
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) override {
        // Assigns or initializes a value used by the algorithm state.
        curr_num_items_ += 1;
    // Closes the current C++ scope.
    }

    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) override {
        // Assigns or initializes a value used by the algorithm state.
        curr_num_items_ -= 1;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) override {
        // Branches only when the condition is true.
        if (candidate_dist > lowerBound && curr_num_items_ == max_num_candidates_) {
            // new candidate can't improve found results
            // Returns the computed value or object to the caller.
            return true;
        // Closes the current C++ scope.
        }
        // Branches only when the condition is true.
        if (candidate_dist > epsilon_ && curr_num_items_ >= min_num_candidates_) {
            // new candidate is out of epsilon region and
            // minimum number of candidates is checked
            // Returns the computed value or object to the caller.
            return true;
        // Closes the current C++ scope.
        }
        // Returns the computed value or object to the caller.
        return false;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) override {
        // Assigns or initializes a value used by the algorithm state.
        bool flag_consider_candidate = curr_num_items_ < max_num_candidates_ || lowerBound > candidate_dist;
        // Returns the computed value or object to the caller.
        return flag_consider_candidate;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    bool should_remove_extra() {
        // Assigns or initializes a value used by the algorithm state.
        bool flag_remove_extra = curr_num_items_ > max_num_candidates_;
        // Returns the computed value or object to the caller.
        return flag_remove_extra;
    // Closes the current C++ scope.
    }

    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) override {
        // Repeats the block while the condition remains true.
        while (!candidates.empty() && candidates.back().first > epsilon_) {
            
            candidates.pop_back();
        // Closes the current C++ scope.
        }
        // Repeats the block while the condition remains true.
        while (candidates.size() > max_num_candidates_) {
            
            candidates.pop_back();
        // Closes the current C++ scope.
        }
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    ~EpsilonSearchStopCondition() {}
// Closes the current class, struct, namespace, or block scope.
};
// Executes this C++ statement as part of the surrounding HNSW implementation.
}  // namespace hnswlib
