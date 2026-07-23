// Ensures this header is included only once in each translation unit.
#pragma once


#include "visited_list_pool.h"

#include "hnswlib.h"

#include <atomic>

#include <random>

#include <stdlib.h>

#include <assert.h>

#include <unordered_set>

#include <list>

#include <memory>

// Places the following declarations inside the hnswlib namespace.
namespace hnswlib {
// Creates a shorter type alias used by the implementation.
typedef unsigned int tableint;
// Creates a shorter type alias used by the implementation.
typedef unsigned int linklistsizeint;

// Declares that the following class or function is templated over one or more types.
template<typename dist_t>
// Declares a C++ type that groups related data and behavior.
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 // Sets the access level for the following class members.
 public:
    // Assigns or initializes a value used by the algorithm state.
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    // Assigns or initializes a value used by the algorithm state.
    static const unsigned char DELETE_MARK = 0x01;

    
    size_t max_elements_{0};
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    
    size_t size_data_per_element_{0};
    
    size_t size_links_per_element_{0};
    // Manages heap memory; pointers here own or reference dynamically allocated storage.
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    
    size_t M_{0};
    
    size_t maxM_{0};
    
    size_t maxM0_{0};
    
    size_t ef_construction_{0};
    
    size_t ef_{ 0 };

    
    double mult_{0.0}, revSize_{0.0};
    
    int maxlevel_{0};

    
    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    // Uses a dynamic array to store a resizable sequence of values.
    mutable std::vector<std::mutex> label_op_locks_;

    // Uses a mutex or lock object to protect shared state during threaded operations.
    std::mutex global;
    // Uses a dynamic array to store a resizable sequence of values.
    std::vector<std::mutex> link_list_locks_;

    
    tableint enterpoint_node_{0};

    
    size_t size_links_level0_{0};
    
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    char *data_level0_memory_{nullptr};
    // Dereferences or casts through a pointer to read or write the object stored at an address.
    char **linkLists_{nullptr};
    // Uses a dynamic array to store a resizable sequence of values.
    std::vector<int> element_levels_;  // keeps level of each element

    
    size_t data_size_{0};

    
    DISTFUNC<dist_t> fstdistfunc_;
    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void *dist_func_param_{nullptr};

    // Uses a mutex or lock object to protect shared state during threaded operations.
    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    
    std::unordered_map<labeltype, tableint> label_lookup_;

    
    std::default_random_engine level_generator_;
    
    std::default_random_engine update_probability_generator_;

    
    mutable std::atomic<long> metric_distance_computations{0};
    
    mutable std::atomic<long> metric_hops{0};

    // Manages heap memory; pointers here own or reference dynamically allocated storage.
    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    // Manages heap memory; pointers here own or reference dynamically allocated storage.
    std::mutex deleted_elements_lock;  // lock for deleted_elements
    // Manages heap memory; pointers here own or reference dynamically allocated storage.
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements


    // Dereferences or casts through a pointer to read or write the object stored at an address.
    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    HierarchicalNSW(
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        SpaceInterface<dist_t> *s,
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        const std::string &location,
        // Assigns or initializes a value used by the algorithm state.
        bool nmslib = false,
        // Assigns or initializes a value used by the algorithm state.
        size_t max_elements = 0,
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        bool allow_replace_deleted = false)
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        : allow_replace_deleted_(allow_replace_deleted) {
        
        loadIndex(location, s, max_elements);
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    HierarchicalNSW(
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        SpaceInterface<dist_t> *s,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        size_t max_elements,
        // Assigns or initializes a value used by the algorithm state.
        size_t M = 16,
        // Assigns or initializes a value used by the algorithm state.
        size_t ef_construction = 200,
        // Assigns or initializes a value used by the algorithm state.
        size_t random_seed = 100,
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        bool allow_replace_deleted = false)
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            link_list_locks_(max_elements),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            element_levels_(max_elements),
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            allow_replace_deleted_(allow_replace_deleted) {
        // Assigns or initializes a value used by the algorithm state.
        max_elements_ = max_elements;
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        num_deleted_ = 0;
        // Uses a pointer to access a member of the object it points to.
        data_size_ = s->get_data_size();
        // Uses a pointer to access a member of the object it points to.
        fstdistfunc_ = s->get_dist_func();
        // Uses a pointer to access a member of the object it points to.
        dist_func_param_ = s->get_dist_func_param();
        // Branches only when the condition is true.
        if ( M <= 10000 ) {
            // Assigns or initializes a value used by the algorithm state.
            M_ = M;
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else {
            
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            // Assigns or initializes a value used by the algorithm state.
            M_ = 10000;
        // Closes the current C++ scope.
        }
        // Assigns or initializes a value used by the algorithm state.
        maxM_ = M_;
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        maxM0_ = M_ * 2;
        // Assigns or initializes a value used by the algorithm state.
        ef_construction_ = std::max(ef_construction, M_);
        // Assigns or initializes a value used by the algorithm state.
        ef_ = 10;

        
        level_generator_.seed(random_seed);
        
        update_probability_generator_.seed(random_seed + 1);

        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        // Assigns or initializes a value used by the algorithm state.
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        // Assigns or initializes a value used by the algorithm state.
        offsetData_ = size_links_level0_;
        // Assigns or initializes a value used by the algorithm state.
        label_offset_ = size_links_level0_ + data_size_;
        // Assigns or initializes a value used by the algorithm state.
        offsetLevel0_ = 0;

        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        // Branches only when the condition is true.
        if (data_level0_memory_ == nullptr)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Not enough memory");

        // Assigns or initializes a value used by the algorithm state.
        cur_element_count = 0;

        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        // Assigns or initializes a value used by the algorithm state.
        enterpoint_node_ = -1;
        // Assigns or initializes a value used by the algorithm state.
        maxlevel_ = -1;

        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        // Branches only when the condition is true.
        if (linkLists_ == nullptr)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        mult_ = 1 / log(1.0 * M_);
        // Assigns or initializes a value used by the algorithm state.
        revSize_ = 1.0 / mult_;
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    ~HierarchicalNSW() {
        
        clear();
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void clear() {
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        free(data_level0_memory_);
        // Assigns or initializes a value used by the algorithm state.
        data_level0_memory_ = nullptr;
        // Iterates over a bounded range of values or elements.
        for (tableint i = 0; i < cur_element_count; i++) {
            // Branches only when the condition is true.
            if (element_levels_[i] > 0)
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                free(linkLists_[i]);
        // Closes the current C++ scope.
        }
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        free(linkLists_);
        // Assigns or initializes a value used by the algorithm state.
        linkLists_ = nullptr;
        // Assigns or initializes a value used by the algorithm state.
        cur_element_count = 0;
        
        visited_list_pool_.reset(nullptr);
    // Closes the current C++ scope.
    }


    // Declares a C++ type that groups related data and behavior.
    struct CompareByFirst {
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
            std::pair<dist_t, tableint> const& b) const noexcept {
            // Returns the computed value or object to the caller.
            return a.first < b.first;
        // Closes the current C++ scope.
        }
    // Closes the current class, struct, namespace, or block scope.
    };


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void setEf(size_t ef) {
        // Assigns or initializes a value used by the algorithm state.
        ef_ = ef;
    // Closes the current C++ scope.
    }


    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        // Returns the computed value or object to the caller.
        return label_op_locks_[lock_id];
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    inline labeltype getExternalLabel(tableint internal_id) const {
        
        labeltype return_label;
        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        // Returns the computed value or object to the caller.
        return return_label;
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    // Closes the current C++ scope.
    }


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    // Closes the current C++ scope.
    }


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    inline char *getDataByInternalId(tableint internal_id) const {
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    int getRandomLevel(double reverse_size) {
        
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        double r = -log(distribution(level_generator_)) * reverse_size;
        // Returns the computed value or object to the caller.
        return (int) r;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    size_t getMaxElements() {
        // Returns the computed value or object to the caller.
        return max_elements_;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    size_t getCurrentElementCount() {
        // Returns the computed value or object to the caller.
        return cur_element_count;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    size_t getDeletedCount() {
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        return num_deleted_;
    // Closes the current C++ scope.
    }

    // Uses a heap-backed priority queue to keep best candidates ordered by distance.
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        // Uses a pointer to access a member of the object it points to.
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        // Uses a pointer to access a member of the object it points to.
        vl_type *visited_array = vl->mass;
        // Uses a pointer to access a member of the object it points to.
        vl_type visited_array_tag = vl->curV;

        // Uses a heap-backed priority queue to keep best candidates ordered by distance.
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        // Uses a heap-backed priority queue to keep best candidates ordered by distance.
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        
        dist_t lowerBound;
        // Branches only when the condition is true.
        if (!isMarkedDeleted(ep_id)) {
            // Assigns or initializes a value used by the algorithm state.
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            
            top_candidates.emplace(dist, ep_id);
            // Assigns or initializes a value used by the algorithm state.
            lowerBound = dist;
            
            candidateSet.emplace(-dist, ep_id);
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else {
            // Assigns or initializes a value used by the algorithm state.
            lowerBound = std::numeric_limits<dist_t>::max();
            
            candidateSet.emplace(-lowerBound, ep_id);
        // Closes the current C++ scope.
        }
        // Assigns or initializes a value used by the algorithm state.
        visited_array[ep_id] = visited_array_tag;

        // Repeats the block while the condition remains true.
        while (!candidateSet.empty()) {
            // Assigns or initializes a value used by the algorithm state.
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            // Branches only when the condition is true.
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                
                break;
            // Closes the current C++ scope.
            }
            
            candidateSet.pop();

            // Assigns or initializes a value used by the algorithm state.
            tableint curNodeNum = curr_el_pair.second;

            // Uses a mutex or lock object to protect shared state during threaded operations.
            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            // Branches only when the condition is true.
            if (layer == 0) {
                // Assigns or initializes a value used by the algorithm state.
                data = (int*)get_linklist0(curNodeNum);
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            } else {
                // Assigns or initializes a value used by the algorithm state.
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            // Closes the current C++ scope.
            }
            // Assigns or initializes a value used by the algorithm state.
            size_t size = getListCount((linklistsizeint*)data);
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            tableint *datal = (tableint *) (data + 1);
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef USE_SSE
            // Dereferences or casts through a pointer to read or write the object stored at an address.
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            // Dereferences or casts through a pointer to read or write the object stored at an address.
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            // Dereferences or casts through a pointer to read or write the object stored at an address.
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            // Dereferences or casts through a pointer to read or write the object stored at an address.
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
// Ends the current conditional compilation block.
#endif

            // Iterates over a bounded range of values or elements.
            for (size_t j = 0; j < size; j++) {
                // Dereferences or casts through a pointer to read or write the object stored at an address.
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef USE_SSE
                // Dereferences or casts through a pointer to read or write the object stored at an address.
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                // Dereferences or casts through a pointer to read or write the object stored at an address.
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
// Ends the current conditional compilation block.
#endif
                // Branches only when the condition is true.
                if (visited_array[candidate_id] == visited_array_tag) continue;
                // Assigns or initializes a value used by the algorithm state.
                visited_array[candidate_id] = visited_array_tag;
                // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                char *currObj1 = (getDataByInternalId(candidate_id));

                // Assigns or initializes a value used by the algorithm state.
                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                // Branches only when the condition is true.
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    
                    candidateSet.emplace(-dist1, candidate_id);
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef USE_SSE
                    
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
// Ends the current conditional compilation block.
#endif

                    // Branches only when the condition is true.
                    if (!isMarkedDeleted(candidate_id))
                        
                        top_candidates.emplace(dist1, candidate_id);

                    // Branches only when the condition is true.
                    if (top_candidates.size() > ef_construction_)
                        
                        top_candidates.pop();

                    // Branches only when the condition is true.
                    if (!top_candidates.empty())
                        // Assigns or initializes a value used by the algorithm state.
                        lowerBound = top_candidates.top().first;
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }
        // Uses a pointer to access a member of the object it points to.
        visited_list_pool_->releaseVisitedList(vl);

        // Returns the computed value or object to the caller.
        return top_candidates;
    // Closes the current C++ scope.
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    // Declares that the following class or function is templated over one or more types.
    template <bool bare_bone_search = true, bool collect_metrics = false>
    // Uses a heap-backed priority queue to keep best candidates ordered by distance.
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    searchBaseLayerST(
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        tableint ep_id,
        // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
        const void *data_point,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        size_t ef,
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        BaseFilterFunctor* isIdAllowed = nullptr,
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        // Uses a pointer to access a member of the object it points to.
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        // Uses a pointer to access a member of the object it points to.
        vl_type *visited_array = vl->mass;
        // Uses a pointer to access a member of the object it points to.
        vl_type visited_array_tag = vl->curV;

        // Uses a heap-backed priority queue to keep best candidates ordered by distance.
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        // Uses a heap-backed priority queue to keep best candidates ordered by distance.
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        
        dist_t lowerBound;
        // Branches only when the condition is true.
        if (bare_bone_search || 
            // Dereferences or casts through a pointer to read or write the object stored at an address.
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            char* ep_data = getDataByInternalId(ep_id);
            // Assigns or initializes a value used by the algorithm state.
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            // Assigns or initializes a value used by the algorithm state.
            lowerBound = dist;
            
            top_candidates.emplace(dist, ep_id);
            // Branches only when the condition is true.
            if (!bare_bone_search && stop_condition) {
                // Uses a pointer to access a member of the object it points to.
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            // Closes the current C++ scope.
            }
            
            candidate_set.emplace(-dist, ep_id);
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else {
            // Assigns or initializes a value used by the algorithm state.
            lowerBound = std::numeric_limits<dist_t>::max();
            
            candidate_set.emplace(-lowerBound, ep_id);
        // Closes the current C++ scope.
        }

        // Assigns or initializes a value used by the algorithm state.
        visited_array[ep_id] = visited_array_tag;

        // Repeats the block while the condition remains true.
        while (!candidate_set.empty()) {
            // Assigns or initializes a value used by the algorithm state.
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            // Assigns or initializes a value used by the algorithm state.
            dist_t candidate_dist = -current_node_pair.first;

            
            bool flag_stop_search;
            // Branches only when the condition is true.
            if (bare_bone_search) {
                // Assigns or initializes a value used by the algorithm state.
                flag_stop_search = candidate_dist > lowerBound;
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            } else {
                // Branches only when the condition is true.
                if (stop_condition) {
                    // Uses a pointer to access a member of the object it points to.
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                // Executes this C++ statement as part of the surrounding HNSW implementation.
                } else {
                    // Assigns or initializes a value used by the algorithm state.
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
            // Branches only when the condition is true.
            if (flag_stop_search) {
                
                break;
            // Closes the current C++ scope.
            }
            
            candidate_set.pop();

            // Assigns or initializes a value used by the algorithm state.
            tableint current_node_id = current_node_pair.second;
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            int *data = (int *) get_linklist0(current_node_id);
            // Assigns or initializes a value used by the algorithm state.
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            // Branches only when the condition is true.
            if (collect_metrics) {
                
                metric_hops++;
                // Assigns or initializes a value used by the algorithm state.
                metric_distance_computations+=size;
            // Closes the current C++ scope.
            }

// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef USE_SSE
            // Dereferences or casts through a pointer to read or write the object stored at an address.
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            // Dereferences or casts through a pointer to read or write the object stored at an address.
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            // Dereferences or casts through a pointer to read or write the object stored at an address.
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
// Ends the current conditional compilation block.
#endif

            // Iterates over a bounded range of values or elements.
            for (size_t j = 1; j <= size; j++) {
                // Dereferences or casts through a pointer to read or write the object stored at an address.
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef USE_SSE
                // Dereferences or casts through a pointer to read or write the object stored at an address.
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                // Dereferences or casts through a pointer to read or write the object stored at an address.
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                // Executes this C++ statement as part of the surrounding HNSW implementation.
                                _MM_HINT_T0);  ////////////
// Ends the current conditional compilation block.
#endif
                // Branches only when the condition is true.
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    // Assigns or initializes a value used by the algorithm state.
                    visited_array[candidate_id] = visited_array_tag;

                    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    // Assigns or initializes a value used by the algorithm state.
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    
                    bool flag_consider_candidate;
                    // Branches only when the condition is true.
                    if (!bare_bone_search && stop_condition) {
                        // Uses a pointer to access a member of the object it points to.
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    // Executes this C++ statement as part of the surrounding HNSW implementation.
                    } else {
                        // Assigns or initializes a value used by the algorithm state.
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    // Closes the current C++ scope.
                    }

                    // Branches only when the condition is true.
                    if (flag_consider_candidate) {
                        
                        candidate_set.emplace(-dist, candidate_id);
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef USE_SSE
                        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        // Executes this C++ statement as part of the surrounding HNSW implementation.
                                        offsetLevel0_,  ///////////
                                        // Executes this C++ statement as part of the surrounding HNSW implementation.
                                        _MM_HINT_T0);  ////////////////////////
// Ends the current conditional compilation block.
#endif

                        // Branches only when the condition is true.
                        if (bare_bone_search || 
                            // Dereferences or casts through a pointer to read or write the object stored at an address.
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            
                            top_candidates.emplace(dist, candidate_id);
                            // Branches only when the condition is true.
                            if (!bare_bone_search && stop_condition) {
                                // Uses a pointer to access a member of the object it points to.
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            // Closes the current C++ scope.
                            }
                        // Closes the current C++ scope.
                        }

                        // Assigns or initializes a value used by the algorithm state.
                        bool flag_remove_extra = false;
                        // Branches only when the condition is true.
                        if (!bare_bone_search && stop_condition) {
                            // Uses a pointer to access a member of the object it points to.
                            flag_remove_extra = stop_condition->should_remove_extra();
                        // Executes this C++ statement as part of the surrounding HNSW implementation.
                        } else {
                            // Assigns or initializes a value used by the algorithm state.
                            flag_remove_extra = top_candidates.size() > ef;
                        // Closes the current C++ scope.
                        }
                        // Repeats the block while the condition remains true.
                        while (flag_remove_extra) {
                            // Assigns or initializes a value used by the algorithm state.
                            tableint id = top_candidates.top().second;
                            
                            top_candidates.pop();
                            // Branches only when the condition is true.
                            if (!bare_bone_search && stop_condition) {
                                // Uses a pointer to access a member of the object it points to.
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                // Uses a pointer to access a member of the object it points to.
                                flag_remove_extra = stop_condition->should_remove_extra();
                            // Executes this C++ statement as part of the surrounding HNSW implementation.
                            } else {
                                // Assigns or initializes a value used by the algorithm state.
                                flag_remove_extra = top_candidates.size() > ef;
                            // Closes the current C++ scope.
                            }
                        // Closes the current C++ scope.
                        }

                        // Branches only when the condition is true.
                        if (!top_candidates.empty())
                            // Assigns or initializes a value used by the algorithm state.
                            lowerBound = top_candidates.top().first;
                    // Closes the current C++ scope.
                    }
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        // Uses a pointer to access a member of the object it points to.
        visited_list_pool_->releaseVisitedList(vl);
        // Returns the computed value or object to the caller.
        return top_candidates;
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void getNeighborsByHeuristic2(
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        const size_t M) {
        // Branches only when the condition is true.
        if (top_candidates.size() < M) {
            // Returns the computed value or object to the caller.
            return;
        // Closes the current C++ scope.
        }

        // Uses a heap-backed priority queue to keep best candidates ordered by distance.
        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        // Uses a dynamic array to store a resizable sequence of values.
        std::vector<std::pair<dist_t, tableint>> return_list;
        // Repeats the block while the condition remains true.
        while (top_candidates.size() > 0) {
            
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            
            top_candidates.pop();
        // Closes the current C++ scope.
        }

        // Repeats the block while the condition remains true.
        while (queue_closest.size()) {
            // Branches only when the condition is true.
            if (return_list.size() >= M)
                
                break;
            // Assigns or initializes a value used by the algorithm state.
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            // Assigns or initializes a value used by the algorithm state.
            dist_t dist_to_query = -curent_pair.first;
            
            queue_closest.pop();
            // Assigns or initializes a value used by the algorithm state.
            bool good = true;

            // Iterates over a bounded range of values or elements.
            for (std::pair<dist_t, tableint> second_pair : return_list) {
                // Assigns or initializes a value used by the algorithm state.
                dist_t curdist =
                        // Executes this C++ statement as part of the surrounding HNSW implementation.
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        // Executes this C++ statement as part of the surrounding HNSW implementation.
                                        getDataByInternalId(curent_pair.second),
                                        
                                        dist_func_param_);
                // Branches only when the condition is true.
                if (curdist < dist_to_query) {
                    // Assigns or initializes a value used by the algorithm state.
                    good = false;
                    
                    break;
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
            // Branches only when the condition is true.
            if (good) {
                // Returns the computed value or object to the caller.
                return_list.push_back(curent_pair);
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        // Iterates over a bounded range of values or elements.
        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        // Closes the current C++ scope.
        }
    // Closes the current C++ scope.
    }


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    linklistsizeint *get_linklist0(tableint internal_id) const {
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    // Closes the current C++ scope.
    }


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    // Closes the current C++ scope.
    }


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    // Closes the current C++ scope.
    }


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        // Returns the computed value or object to the caller.
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    tableint mutuallyConnectNewElement(
        // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
        const void *data_point,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        tableint cur_c,
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        int level,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        bool isUpdate) {
        // Assigns or initializes a value used by the algorithm state.
        size_t Mcurmax = level ? maxM_ : maxM0_;
        
        getNeighborsByHeuristic2(top_candidates, M_);
        // Branches only when the condition is true.
        if (top_candidates.size() > M_)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        // Uses a dynamic array to store a resizable sequence of values.
        std::vector<tableint> selectedNeighbors;
        
        selectedNeighbors.reserve(M_);
        // Repeats the block while the condition remains true.
        while (top_candidates.size() > 0) {
            
            selectedNeighbors.push_back(top_candidates.top().second);
            
            top_candidates.pop();
        // Closes the current C++ scope.
        }

        // Assigns or initializes a value used by the algorithm state.
        tableint next_closest_entry_point = selectedNeighbors.back();

        // Opens a new C++ scope for the preceding declaration or control statement.
        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            // Uses a mutex or lock object to protect shared state during threaded operations.
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            // Branches only when the condition is true.
            if (isUpdate) {
                
                lock.lock();
            // Closes the current C++ scope.
            }
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            linklistsizeint *ll_cur;
            // Branches only when the condition is true.
            if (level == 0)
                // Assigns or initializes a value used by the algorithm state.
                ll_cur = get_linklist0(cur_c);
            // Handles the alternative path when the previous condition was false.
            else
                // Assigns or initializes a value used by the algorithm state.
                ll_cur = get_linklist(cur_c, level);

            // Dereferences or casts through a pointer to read or write the object stored at an address.
            if (*ll_cur && !isUpdate) {
                // Handles an exceptional error path by throwing or catching a C++ exception.
                throw std::runtime_error("The newly inserted element should have blank link list");
            // Closes the current C++ scope.
            }
            
            setListCount(ll_cur, selectedNeighbors.size());
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            tableint *data = (tableint *) (ll_cur + 1);
            // Iterates over a bounded range of values or elements.
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                // Branches only when the condition is true.
                if (data[idx] && !isUpdate)
                    // Handles an exceptional error path by throwing or catching a C++ exception.
                    throw std::runtime_error("Possible memory corruption");
                // Branches only when the condition is true.
                if (level > element_levels_[selectedNeighbors[idx]])
                    // Handles an exceptional error path by throwing or catching a C++ exception.
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                // Assigns or initializes a value used by the algorithm state.
                data[idx] = selectedNeighbors[idx];
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        // Iterates over a bounded range of values or elements.
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            // Uses a mutex or lock object to protect shared state during threaded operations.
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            linklistsizeint *ll_other;
            // Branches only when the condition is true.
            if (level == 0)
                // Assigns or initializes a value used by the algorithm state.
                ll_other = get_linklist0(selectedNeighbors[idx]);
            // Handles the alternative path when the previous condition was false.
            else
                // Assigns or initializes a value used by the algorithm state.
                ll_other = get_linklist(selectedNeighbors[idx], level);

            // Assigns or initializes a value used by the algorithm state.
            size_t sz_link_list_other = getListCount(ll_other);

            // Branches only when the condition is true.
            if (sz_link_list_other > Mcurmax)
                // Handles an exceptional error path by throwing or catching a C++ exception.
                throw std::runtime_error("Bad value of sz_link_list_other");
            // Branches only when the condition is true.
            if (selectedNeighbors[idx] == cur_c)
                // Handles an exceptional error path by throwing or catching a C++ exception.
                throw std::runtime_error("Trying to connect an element to itself");
            // Branches only when the condition is true.
            if (level > element_levels_[selectedNeighbors[idx]])
                // Handles an exceptional error path by throwing or catching a C++ exception.
                throw std::runtime_error("Trying to make a link on a non-existent level");

            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            tableint *data = (tableint *) (ll_other + 1);

            // Assigns or initializes a value used by the algorithm state.
            bool is_cur_c_present = false;
            // Branches only when the condition is true.
            if (isUpdate) {
                // Iterates over a bounded range of values or elements.
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    // Branches only when the condition is true.
                    if (data[j] == cur_c) {
                        // Assigns or initializes a value used by the algorithm state.
                        is_cur_c_present = true;
                        
                        break;
                    // Closes the current C++ scope.
                    }
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            // Branches only when the condition is true.
            if (!is_cur_c_present) {
                // Branches only when the condition is true.
                if (sz_link_list_other < Mcurmax) {
                    // Assigns or initializes a value used by the algorithm state.
                    data[sz_link_list_other] = cur_c;
                    
                    setListCount(ll_other, sz_link_list_other + 1);
                // Executes this C++ statement as part of the surrounding HNSW implementation.
                } else {
                    // finding the "weakest" element to replace it with the new one
                    // Assigns or initializes a value used by the algorithm state.
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                
                                                dist_func_param_);
                    // Heuristic:
                    // Uses a heap-backed priority queue to keep best candidates ordered by distance.
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    
                    candidates.emplace(d_max, cur_c);

                    // Iterates over a bounded range of values or elements.
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        // Executes this C++ statement as part of the surrounding HNSW implementation.
                        candidates.emplace(
                                // Executes this C++ statement as part of the surrounding HNSW implementation.
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                
                                                dist_func_param_), data[j]);
                    // Closes the current C++ scope.
                    }

                    
                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    // Assigns or initializes a value used by the algorithm state.
                    int indx = 0;
                    // Repeats the block while the condition remains true.
                    while (candidates.size() > 0) {
                        // Assigns or initializes a value used by the algorithm state.
                        data[indx] = candidates.top().second;
                        
                        candidates.pop();
                        
                        indx++;
                    // Closes the current C++ scope.
                    }

                    
                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        // Returns the computed value or object to the caller.
        return next_closest_entry_point;
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void resizeIndex(size_t new_max_elements) {
        // Branches only when the condition is true.
        if (new_max_elements < cur_element_count)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        
        element_levels_.resize(new_max_elements);

        // Uses a dynamic array to store a resizable sequence of values.
        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        if (data_level0_memory_new == nullptr)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        // Assigns or initializes a value used by the algorithm state.
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        if (linkLists_new == nullptr)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        // Assigns or initializes a value used by the algorithm state.
        linkLists_ = linkLists_new;

        // Assigns or initializes a value used by the algorithm state.
        max_elements_ = new_max_elements;
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    size_t indexFileSize() const {
        // Assigns or initializes a value used by the algorithm state.
        size_t size = 0;
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(offsetLevel0_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(max_elements_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(cur_element_count);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(size_data_per_element_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(label_offset_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(offsetData_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(maxlevel_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(enterpoint_node_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(maxM_);

        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(maxM0_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(M_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(mult_);
        // Assigns or initializes a value used by the algorithm state.
        size += sizeof(ef_construction_);

        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        size += cur_element_count * size_data_per_element_;

        // Iterates over a bounded range of values or elements.
        for (size_t i = 0; i < cur_element_count; i++) {
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            // Assigns or initializes a value used by the algorithm state.
            size += sizeof(linkListSize);
            // Assigns or initializes a value used by the algorithm state.
            size += linkListSize;
        // Closes the current C++ scope.
        }
        // Returns the computed value or object to the caller.
        return size;
    // Closes the current C++ scope.
    }

    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    void saveIndex(const std::string &location) {
        
        std::ofstream output(location, std::ios::binary);

        
        writeBinaryPOD(output, offsetLevel0_);
        
        writeBinaryPOD(output, max_elements_);
        
        writeBinaryPOD(output, cur_element_count);
        
        writeBinaryPOD(output, size_data_per_element_);
        
        writeBinaryPOD(output, label_offset_);
        
        writeBinaryPOD(output, offsetData_);
        
        writeBinaryPOD(output, maxlevel_);
        
        writeBinaryPOD(output, enterpoint_node_);
        
        writeBinaryPOD(output, maxM_);

        
        writeBinaryPOD(output, maxM0_);
        
        writeBinaryPOD(output, M_);
        
        writeBinaryPOD(output, mult_);
        
        writeBinaryPOD(output, ef_construction_);

        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        // Iterates over a bounded range of values or elements.
        for (size_t i = 0; i < cur_element_count; i++) {
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            
            writeBinaryPOD(output, linkListSize);
            // Branches only when the condition is true.
            if (linkListSize)
                
                output.write(linkLists_[i], linkListSize);
        // Closes the current C++ scope.
        }
        
        output.close();
    // Closes the current C++ scope.
    }


    // Dereferences or casts through a pointer to read or write the object stored at an address.
    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        
        std::ifstream input(location, std::ios::binary);

        // Branches only when the condition is true.
        if (!input.is_open())
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Cannot open file");

        
        clear();
        // get file size:
        
        input.seekg(0, input.end);
        // Assigns or initializes a value used by the algorithm state.
        std::streampos total_filesize = input.tellg();
        
        input.seekg(0, input.beg);

        
        readBinaryPOD(input, offsetLevel0_);
        
        readBinaryPOD(input, max_elements_);
        
        readBinaryPOD(input, cur_element_count);

        // Assigns or initializes a value used by the algorithm state.
        size_t max_elements = max_elements_i;
        // Branches only when the condition is true.
        if (max_elements < cur_element_count)
            // Assigns or initializes a value used by the algorithm state.
            max_elements = max_elements_;
        // Assigns or initializes a value used by the algorithm state.
        max_elements_ = max_elements;
        
        readBinaryPOD(input, size_data_per_element_);
        
        readBinaryPOD(input, label_offset_);
        
        readBinaryPOD(input, offsetData_);
        
        readBinaryPOD(input, maxlevel_);
        
        readBinaryPOD(input, enterpoint_node_);

        
        readBinaryPOD(input, maxM_);
        
        readBinaryPOD(input, maxM0_);
        
        readBinaryPOD(input, M_);
        
        readBinaryPOD(input, mult_);
        
        readBinaryPOD(input, ef_construction_);

        // Uses a pointer to access a member of the object it points to.
        data_size_ = s->get_data_size();
        // Uses a pointer to access a member of the object it points to.
        fstdistfunc_ = s->get_dist_func();
        // Uses a pointer to access a member of the object it points to.
        dist_func_param_ = s->get_dist_func_param();

        // Assigns or initializes a value used by the algorithm state.
        auto pos = input.tellg();

        /// Optional - check if index is ok:
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        // Iterates over a bounded range of values or elements.
        for (size_t i = 0; i < cur_element_count; i++) {
            // Branches only when the condition is true.
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                // Handles an exceptional error path by throwing or catching a C++ exception.
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            // Closes the current C++ scope.
            }

            
            unsigned int linkListSize;
            
            readBinaryPOD(input, linkListSize);
            // Branches only when the condition is true.
            if (linkListSize != 0) {
                
                input.seekg(linkListSize, input.cur);
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        // throw exception if it either corrupted or old index
        // Branches only when the condition is true.
        if (input.tellg() != total_filesize)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        
        input.clear();
        /// Optional check end

        
        input.seekg(pos, input.beg);

        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        // Branches only when the condition is true.
        if (data_level0_memory_ == nullptr)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        // Uses a dynamic array to store a resizable sequence of values.
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        // Uses a dynamic array to store a resizable sequence of values.
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        // Branches only when the condition is true.
        if (linkLists_ == nullptr)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        // Uses a dynamic array to store a resizable sequence of values.
        element_levels_ = std::vector<int>(max_elements);
        // Assigns or initializes a value used by the algorithm state.
        revSize_ = 1.0 / mult_;
        // Assigns or initializes a value used by the algorithm state.
        ef_ = 10;
        // Iterates over a bounded range of values or elements.
        for (size_t i = 0; i < cur_element_count; i++) {
            // Assigns or initializes a value used by the algorithm state.
            label_lookup_[getExternalLabel(i)] = i;
            
            unsigned int linkListSize;
            
            readBinaryPOD(input, linkListSize);
            // Branches only when the condition is true.
            if (linkListSize == 0) {
                // Assigns or initializes a value used by the algorithm state.
                element_levels_[i] = 0;
                // Assigns or initializes a value used by the algorithm state.
                linkLists_[i] = nullptr;
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            } else {
                // Assigns or initializes a value used by the algorithm state.
                element_levels_[i] = linkListSize / size_links_per_element_;
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                linkLists_[i] = (char *) malloc(linkListSize);
                // Branches only when the condition is true.
                if (linkLists_[i] == nullptr)
                    // Handles an exceptional error path by throwing or catching a C++ exception.
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                
                input.read(linkLists_[i], linkListSize);
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        // Iterates over a bounded range of values or elements.
        for (size_t i = 0; i < cur_element_count; i++) {
            // Branches only when the condition is true.
            if (isMarkedDeleted(i)) {
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                num_deleted_ += 1;
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                if (allow_replace_deleted_) deleted_elements.insert(i);
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        
        input.close();

        // Returns the computed value or object to the caller.
        return;
    // Closes the current C++ scope.
    }


    // Declares that the following class or function is templated over one or more types.
    template<typename data_t>
    // Uses a dynamic array to store a resizable sequence of values.
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        // Assigns or initializes a value used by the algorithm state.
        auto search = label_lookup_.find(label);
        // Uses a pointer to access a member of the object it points to.
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Label not found");
        // Closes the current C++ scope.
        }
        // Uses a pointer to access a member of the object it points to.
        tableint internalId = search->second;
        
        lock_table.unlock();

        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        char* data_ptrv = getDataByInternalId(internalId);
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        size_t dim = *((size_t *) dist_func_param_);
        // Uses a dynamic array to store a resizable sequence of values.
        std::vector<data_t> data;
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        data_t* data_ptr = (data_t*) data_ptrv;
        // Iterates over a bounded range of values or elements.
        for (size_t i = 0; i < dim; i++) {
            // Dereferences or casts through a pointer to read or write the object stored at an address.
            data.push_back(*data_ptr);
            // Assigns or initializes a value used by the algorithm state.
            data_ptr += 1;
        // Closes the current C++ scope.
        }
        // Returns the computed value or object to the caller.
        return data;
    // Closes the current C++ scope.
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void markDelete(labeltype label) {
        // lock all operations with element by label
        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        // Assigns or initializes a value used by the algorithm state.
        auto search = label_lookup_.find(label);
        // Branches only when the condition is true.
        if (search == label_lookup_.end()) {
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Label not found");
        // Closes the current C++ scope.
        }
        // Uses a pointer to access a member of the object it points to.
        tableint internalId = search->second;
        
        lock_table.unlock();

        
        markDeletedInternal(internalId);
    // Closes the current C++ scope.
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void markDeletedInternal(tableint internalId) {
        
        assert(internalId < cur_element_count);
        // Branches only when the condition is true.
        if (!isMarkedDeleted(internalId)) {
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            num_deleted_ += 1;
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            if (allow_replace_deleted_) {
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                deleted_elements.insert(internalId);
            // Closes the current C++ scope.
            }
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else {
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            throw std::runtime_error("The requested to delete element is already deleted");
        // Closes the current C++ scope.
        }
    // Closes the current C++ scope.
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        // Assigns or initializes a value used by the algorithm state.
        auto search = label_lookup_.find(label);
        // Branches only when the condition is true.
        if (search == label_lookup_.end()) {
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Label not found");
        // Closes the current C++ scope.
        }
        // Uses a pointer to access a member of the object it points to.
        tableint internalId = search->second;
        
        lock_table.unlock();

        
        unmarkDeletedInternal(internalId);
    // Closes the current C++ scope.
    }



    /*
    * Remove the deleted mark of the node.
    */
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void unmarkDeletedInternal(tableint internalId) {
        
        assert(internalId < cur_element_count);
        // Branches only when the condition is true.
        if (isMarkedDeleted(internalId)) {
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            num_deleted_ -= 1;
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            if (allow_replace_deleted_) {
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                deleted_elements.erase(internalId);
            // Closes the current C++ scope.
            }
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else {
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            throw std::runtime_error("The requested to undelete element is not deleted");
        // Closes the current C++ scope.
        }
    // Closes the current C++ scope.
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    bool isMarkedDeleted(tableint internalId) const {
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        return *ll_cur & DELETE_MARK;
    // Closes the current C++ scope.
    }


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    unsigned short int getListCount(linklistsizeint * ptr) const {
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        return *((unsigned short int *)ptr);
    // Closes the current C++ scope.
    }


    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    // Closes the current C++ scope.
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    // Manages heap memory; pointers here own or reference dynamically allocated storage.
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        // Closes the current C++ scope.
        }

        // lock all operations with element by label
        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        if (!replace_deleted) {
            
            addPoint(data_point, label, -1);
            // Returns the computed value or object to the caller.
            return;
        // Closes the current C++ scope.
        }
        // check if there is vacant place
        
        tableint internal_id_replaced;
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        bool is_vacant_place = !deleted_elements.empty();
        // Branches only when the condition is true.
        if (is_vacant_place) {
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            internal_id_replaced = *deleted_elements.begin();
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            deleted_elements.erase(internal_id_replaced);
        // Closes the current C++ scope.
        }
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        // Branches only when the condition is true.
        if (!is_vacant_place) {
            
            addPoint(data_point, label, -1);
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else {
            // we assume that there are no concurrent operations on deleted element
            // Assigns or initializes a value used by the algorithm state.
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            
            setExternalLabel(internal_id_replaced, label);

            // Uses a mutex or lock object to protect shared state during threaded operations.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            
            label_lookup_.erase(label_replaced);
            // Assigns or initializes a value used by the algorithm state.
            label_lookup_[label] = internal_id_replaced;
            
            lock_table.unlock();

            
            unmarkDeletedInternal(internal_id_replaced);
            
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        // Assigns or initializes a value used by the algorithm state.
        int maxLevelCopy = maxlevel_;
        // Assigns or initializes a value used by the algorithm state.
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        // Branches only when the condition is true.
        if (entryPointCopy == internalId && cur_element_count == 1)
            // Returns the computed value or object to the caller.
            return;

        // Assigns or initializes a value used by the algorithm state.
        int elemLevel = element_levels_[internalId];
        
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        // Iterates over a bounded range of values or elements.
        for (int layer = 0; layer <= elemLevel; layer++) {
            
            std::unordered_set<tableint> sCand;
            
            std::unordered_set<tableint> sNeigh;
            // Uses a dynamic array to store a resizable sequence of values.
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            // Branches only when the condition is true.
            if (listOneHop.size() == 0)
                
                continue;

            
            sCand.insert(internalId);

            // Iterates over a bounded range of values or elements.
            for (auto&& elOneHop : listOneHop) {
                
                sCand.insert(elOneHop);

                // Branches only when the condition is true.
                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    
                    continue;

                
                sNeigh.insert(elOneHop);

                // Uses a dynamic array to store a resizable sequence of values.
                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                // Iterates over a bounded range of values or elements.
                for (auto&& elTwoHop : listTwoHop) {
                    
                    sCand.insert(elTwoHop);
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }

            // Iterates over a bounded range of values or elements.
            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                // Uses a heap-backed priority queue to keep best candidates ordered by distance.
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                // Assigns or initializes a value used by the algorithm state.
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                // Assigns or initializes a value used by the algorithm state.
                size_t elementsToKeep = std::min(ef_construction_, size);
                // Iterates over a bounded range of values or elements.
                for (auto&& cand : sCand) {
                    // Branches only when the condition is true.
                    if (cand == neigh)
                        
                        continue;

                    // Assigns or initializes a value used by the algorithm state.
                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    // Branches only when the condition is true.
                    if (candidates.size() < elementsToKeep) {
                        
                        candidates.emplace(distance, cand);
                    // Executes this C++ statement as part of the surrounding HNSW implementation.
                    } else {
                        // Branches only when the condition is true.
                        if (distance < candidates.top().first) {
                            
                            candidates.pop();
                            
                            candidates.emplace(distance, cand);
                        // Closes the current C++ scope.
                        }
                    // Closes the current C++ scope.
                    }
                // Closes the current C++ scope.
                }

                // Retrieve neighbours using heuristic and set connections.
                // Assigns or initializes a value used by the algorithm state.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                // Opens a new C++ scope for the preceding declaration or control statement.
                {
                    // Uses a mutex or lock object to protect shared state during threaded operations.
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                    linklistsizeint *ll_cur;
                    // Assigns or initializes a value used by the algorithm state.
                    ll_cur = get_linklist_at_level(neigh, layer);
                    // Assigns or initializes a value used by the algorithm state.
                    size_t candSize = candidates.size();
                    
                    setListCount(ll_cur, candSize);
                    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                    tableint *data = (tableint *) (ll_cur + 1);
                    // Iterates over a bounded range of values or elements.
                    for (size_t idx = 0; idx < candSize; idx++) {
                        // Assigns or initializes a value used by the algorithm state.
                        data[idx] = candidates.top().second;
                        
                        candidates.pop();
                    // Closes the current C++ scope.
                    }
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        
        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void repairConnectionsForUpdate(
        // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
        const void *dataPoint,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        tableint entryPointInternalId,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        tableint dataPointInternalId,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        int dataPointLevel,
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        int maxLevel) {
        // Assigns or initializes a value used by the algorithm state.
        tableint currObj = entryPointInternalId;
        // Branches only when the condition is true.
        if (dataPointLevel < maxLevel) {
            // Assigns or initializes a value used by the algorithm state.
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            // Iterates over a bounded range of values or elements.
            for (int level = maxLevel; level > dataPointLevel; level--) {
                // Assigns or initializes a value used by the algorithm state.
                bool changed = true;
                // Repeats the block while the condition remains true.
                while (changed) {
                    // Assigns or initializes a value used by the algorithm state.
                    changed = false;
                    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                    unsigned int *data;
                    // Uses a mutex or lock object to protect shared state during threaded operations.
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    // Assigns or initializes a value used by the algorithm state.
                    data = get_linklist_at_level(currObj, level);
                    // Assigns or initializes a value used by the algorithm state.
                    int size = getListCount(data);
                    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                    tableint *datal = (tableint *) (data + 1);
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef USE_SSE
                    // Dereferences or casts through a pointer to read or write the object stored at an address.
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
// Ends the current conditional compilation block.
#endif
                    // Iterates over a bounded range of values or elements.
                    for (int i = 0; i < size; i++) {
// Starts or continues a conditional compilation branch selected by compiler macros.
#ifdef USE_SSE
                        // Dereferences or casts through a pointer to read or write the object stored at an address.
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
// Ends the current conditional compilation block.
#endif
                        // Assigns or initializes a value used by the algorithm state.
                        tableint cand = datal[i];
                        // Assigns or initializes a value used by the algorithm state.
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        // Branches only when the condition is true.
                        if (d < curdist) {
                            // Assigns or initializes a value used by the algorithm state.
                            curdist = d;
                            // Assigns or initializes a value used by the algorithm state.
                            currObj = cand;
                            // Assigns or initializes a value used by the algorithm state.
                            changed = true;
                        // Closes the current C++ scope.
                        }
                    // Closes the current C++ scope.
                    }
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        // Branches only when the condition is true.
        if (dataPointLevel > maxLevel)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        // Iterates over a bounded range of values or elements.
        for (int level = dataPointLevel; level >= 0; level--) {
            // Uses a heap-backed priority queue to keep best candidates ordered by distance.
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    
                    currObj, dataPoint, level);

            // Uses a heap-backed priority queue to keep best candidates ordered by distance.
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            // Repeats the block while the condition remains true.
            while (topCandidates.size() > 0) {
                // Branches only when the condition is true.
                if (topCandidates.top().second != dataPointInternalId)
                    
                    filteredTopCandidates.push(topCandidates.top());

                
                topCandidates.pop();
            // Closes the current C++ scope.
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            // Branches only when the condition is true.
            if (filteredTopCandidates.size() > 0) {
                // Assigns or initializes a value used by the algorithm state.
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                // Branches only when the condition is true.
                if (epDeleted) {
                    
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    // Branches only when the condition is true.
                    if (filteredTopCandidates.size() > ef_construction_)
                        
                        filteredTopCandidates.pop();
                // Closes the current C++ scope.
                }

                // Assigns or initializes a value used by the algorithm state.
                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }
    // Closes the current C++ scope.
    }


    // Uses a dynamic array to store a resizable sequence of values.
    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        unsigned int *data = get_linklist_at_level(internalId, level);
        // Assigns or initializes a value used by the algorithm state.
        int size = getListCount(data);
        // Uses a dynamic array to store a resizable sequence of values.
        std::vector<tableint> result(size);
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        tableint *ll = (tableint *) (data + 1);
        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memcpy(result.data(), ll, size * sizeof(tableint));
        // Returns the computed value or object to the caller.
        return result;
    // Closes the current C++ scope.
    }


    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    tableint addPoint(const void *data_point, labeltype label, int level) {
        // Assigns or initializes a value used by the algorithm state.
        tableint cur_c = 0;
        // Opens a new C++ scope for the preceding declaration or control statement.
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            // Uses a mutex or lock object to protect shared state during threaded operations.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            // Assigns or initializes a value used by the algorithm state.
            auto search = label_lookup_.find(label);
            // Branches only when the condition is true.
            if (search != label_lookup_.end()) {
                // Uses a pointer to access a member of the object it points to.
                tableint existingInternalId = search->second;
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                if (allow_replace_deleted_) {
                    // Branches only when the condition is true.
                    if (isMarkedDeleted(existingInternalId)) {
                        // Manages heap memory; pointers here own or reference dynamically allocated storage.
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    // Closes the current C++ scope.
                    }
                // Closes the current C++ scope.
                }
                
                lock_table.unlock();

                // Branches only when the condition is true.
                if (isMarkedDeleted(existingInternalId)) {
                    
                    unmarkDeletedInternal(existingInternalId);
                // Closes the current C++ scope.
                }
                
                updatePoint(data_point, existingInternalId, 1.0);

                // Returns the computed value or object to the caller.
                return existingInternalId;
            // Closes the current C++ scope.
            }

            // Branches only when the condition is true.
            if (cur_element_count >= max_elements_) {
                // Handles an exceptional error path by throwing or catching a C++ exception.
                throw std::runtime_error("The number of elements exceeds the specified limit");
            // Closes the current C++ scope.
            }

            // Assigns or initializes a value used by the algorithm state.
            cur_c = cur_element_count;
            
            cur_element_count++;
            // Assigns or initializes a value used by the algorithm state.
            label_lookup_[label] = cur_c;
        // Closes the current C++ scope.
        }

        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        // Assigns or initializes a value used by the algorithm state.
        int curlevel = getRandomLevel(mult_);
        // Branches only when the condition is true.
        if (level > 0)
            // Assigns or initializes a value used by the algorithm state.
            curlevel = level;

        // Assigns or initializes a value used by the algorithm state.
        element_levels_[cur_c] = curlevel;

        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> templock(global);
        // Assigns or initializes a value used by the algorithm state.
        int maxlevelcopy = maxlevel_;
        // Branches only when the condition is true.
        if (curlevel <= maxlevelcopy)
            
            templock.unlock();
        // Assigns or initializes a value used by the algorithm state.
        tableint currObj = enterpoint_node_;
        // Assigns or initializes a value used by the algorithm state.
        tableint enterpoint_copy = enterpoint_node_;

        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        // Branches only when the condition is true.
        if (curlevel) {
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            // Branches only when the condition is true.
            if (linkLists_[cur_c] == nullptr)
                // Handles an exceptional error path by throwing or catching a C++ exception.
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        // Closes the current C++ scope.
        }

        // Branches only when the condition is true.
        if ((signed)currObj != -1) {
            // Branches only when the condition is true.
            if (curlevel < maxlevelcopy) {
                // Assigns or initializes a value used by the algorithm state.
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                // Iterates over a bounded range of values or elements.
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    // Assigns or initializes a value used by the algorithm state.
                    bool changed = true;
                    // Repeats the block while the condition remains true.
                    while (changed) {
                        // Assigns or initializes a value used by the algorithm state.
                        changed = false;
                        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                        unsigned int *data;
                        // Uses a mutex or lock object to protect shared state during threaded operations.
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        // Assigns or initializes a value used by the algorithm state.
                        data = get_linklist(currObj, level);
                        // Assigns or initializes a value used by the algorithm state.
                        int size = getListCount(data);

                        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                        tableint *datal = (tableint *) (data + 1);
                        // Iterates over a bounded range of values or elements.
                        for (int i = 0; i < size; i++) {
                            // Assigns or initializes a value used by the algorithm state.
                            tableint cand = datal[i];
                            // Branches only when the condition is true.
                            if (cand < 0 || cand > max_elements_)
                                // Handles an exceptional error path by throwing or catching a C++ exception.
                                throw std::runtime_error("cand error");
                            // Assigns or initializes a value used by the algorithm state.
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            // Branches only when the condition is true.
                            if (d < curdist) {
                                // Assigns or initializes a value used by the algorithm state.
                                curdist = d;
                                // Assigns or initializes a value used by the algorithm state.
                                currObj = cand;
                                // Assigns or initializes a value used by the algorithm state.
                                changed = true;
                            // Closes the current C++ scope.
                            }
                        // Closes the current C++ scope.
                        }
                    // Closes the current C++ scope.
                    }
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }

            // Assigns or initializes a value used by the algorithm state.
            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            // Iterates over a bounded range of values or elements.
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                // Branches only when the condition is true.
                if (level > maxlevelcopy || level < 0)  // possible?
                    // Handles an exceptional error path by throwing or catching a C++ exception.
                    throw std::runtime_error("Level error");

                // Uses a heap-backed priority queue to keep best candidates ordered by distance.
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        
                        currObj, data_point, level);
                // Branches only when the condition is true.
                if (epDeleted) {
                    
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    // Branches only when the condition is true.
                    if (top_candidates.size() > ef_construction_)
                        
                        top_candidates.pop();
                // Closes the current C++ scope.
                }
                // Assigns or initializes a value used by the algorithm state.
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            // Closes the current C++ scope.
            }
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else {
            // Do nothing for the first element
            // Assigns or initializes a value used by the algorithm state.
            enterpoint_node_ = 0;
            // Assigns or initializes a value used by the algorithm state.
            maxlevel_ = curlevel;
        // Closes the current C++ scope.
        }

        // Releasing lock for the maximum level
        // Branches only when the condition is true.
        if (curlevel > maxlevelcopy) {
            // Assigns or initializes a value used by the algorithm state.
            enterpoint_node_ = cur_c;
            // Assigns or initializes a value used by the algorithm state.
            maxlevel_ = curlevel;
        // Closes the current C++ scope.
        }
        // Returns the computed value or object to the caller.
        return cur_c;
    // Closes the current C++ scope.
    }


    // Uses a heap-backed priority queue to keep best candidates ordered by distance.
    std::priority_queue<std::pair<dist_t, labeltype >>
    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        // Uses a heap-backed priority queue to keep best candidates ordered by distance.
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        // Branches only when the condition is true.
        if (cur_element_count == 0) return result;

        // Assigns or initializes a value used by the algorithm state.
        tableint currObj = enterpoint_node_;
        // Assigns or initializes a value used by the algorithm state.
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        // Iterates over a bounded range of values or elements.
        for (int level = maxlevel_; level > 0; level--) {
            // Assigns or initializes a value used by the algorithm state.
            bool changed = true;
            // Repeats the block while the condition remains true.
            while (changed) {
                // Assigns or initializes a value used by the algorithm state.
                changed = false;
                // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                unsigned int *data;

                // Assigns or initializes a value used by the algorithm state.
                data = (unsigned int *) get_linklist(currObj, level);
                // Assigns or initializes a value used by the algorithm state.
                int size = getListCount(data);
                
                metric_hops++;
                // Assigns or initializes a value used by the algorithm state.
                metric_distance_computations+=size;

                // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                tableint *datal = (tableint *) (data + 1);
                // Iterates over a bounded range of values or elements.
                for (int i = 0; i < size; i++) {
                    // Assigns or initializes a value used by the algorithm state.
                    tableint cand = datal[i];
                    // Branches only when the condition is true.
                    if (cand < 0 || cand > max_elements_)
                        // Handles an exceptional error path by throwing or catching a C++ exception.
                        throw std::runtime_error("cand error");
                    // Assigns or initializes a value used by the algorithm state.
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    // Branches only when the condition is true.
                    if (d < curdist) {
                        // Assigns or initializes a value used by the algorithm state.
                        curdist = d;
                        // Assigns or initializes a value used by the algorithm state.
                        currObj = cand;
                        // Assigns or initializes a value used by the algorithm state.
                        changed = true;
                    // Closes the current C++ scope.
                    }
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        // Uses a heap-backed priority queue to keep best candidates ordered by distance.
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        // Branches only when the condition is true.
        if (bare_bone_search) {
            // Assigns or initializes a value used by the algorithm state.
            top_candidates = searchBaseLayerST<true>(
                    
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        } else {
            // Assigns or initializes a value used by the algorithm state.
            top_candidates = searchBaseLayerST<false>(
                    
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        // Closes the current C++ scope.
        }

        // Repeats the block while the condition remains true.
        while (top_candidates.size() > k) {
            
            top_candidates.pop();
        // Closes the current C++ scope.
        }
        // Repeats the block while the condition remains true.
        while (top_candidates.size() > 0) {
            // Assigns or initializes a value used by the algorithm state.
            std::pair<dist_t, tableint> rez = top_candidates.top();
            
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            
            top_candidates.pop();
        // Closes the current C++ scope.
        }
        // Returns the computed value or object to the caller.
        return result;
    // Closes the current C++ scope.
    }


    // Uses a dynamic array to store a resizable sequence of values.
    std::vector<std::pair<dist_t, labeltype >>
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    searchStopConditionClosest(
        // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
        const void *query_data,
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        BaseSearchStopCondition<dist_t>& stop_condition,
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        // Uses a dynamic array to store a resizable sequence of values.
        std::vector<std::pair<dist_t, labeltype >> result;
        // Branches only when the condition is true.
        if (cur_element_count == 0) return result;

        // Assigns or initializes a value used by the algorithm state.
        tableint currObj = enterpoint_node_;
        // Assigns or initializes a value used by the algorithm state.
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        // Iterates over a bounded range of values or elements.
        for (int level = maxlevel_; level > 0; level--) {
            // Assigns or initializes a value used by the algorithm state.
            bool changed = true;
            // Repeats the block while the condition remains true.
            while (changed) {
                // Assigns or initializes a value used by the algorithm state.
                changed = false;
                // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                unsigned int *data;

                // Assigns or initializes a value used by the algorithm state.
                data = (unsigned int *) get_linklist(currObj, level);
                // Assigns or initializes a value used by the algorithm state.
                int size = getListCount(data);
                
                metric_hops++;
                // Assigns or initializes a value used by the algorithm state.
                metric_distance_computations+=size;

                // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                tableint *datal = (tableint *) (data + 1);
                // Iterates over a bounded range of values or elements.
                for (int i = 0; i < size; i++) {
                    // Assigns or initializes a value used by the algorithm state.
                    tableint cand = datal[i];
                    // Branches only when the condition is true.
                    if (cand < 0 || cand > max_elements_)
                        // Handles an exceptional error path by throwing or catching a C++ exception.
                        throw std::runtime_error("cand error");
                    // Assigns or initializes a value used by the algorithm state.
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    // Branches only when the condition is true.
                    if (d < curdist) {
                        // Assigns or initializes a value used by the algorithm state.
                        curdist = d;
                        // Assigns or initializes a value used by the algorithm state.
                        currObj = cand;
                        // Assigns or initializes a value used by the algorithm state.
                        changed = true;
                    // Closes the current C++ scope.
                    }
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }

        // Uses a heap-backed priority queue to keep best candidates ordered by distance.
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        // Assigns or initializes a value used by the algorithm state.
        size_t sz = top_candidates.size();
        
        result.resize(sz);
        // Repeats the block while the condition remains true.
        while (!top_candidates.empty()) {
            // Assigns or initializes a value used by the algorithm state.
            result[--sz] = top_candidates.top();
            
            top_candidates.pop();
        // Closes the current C++ scope.
        }

        
        stop_condition.filter_results(result);

        // Returns the computed value or object to the caller.
        return result;
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void checkIntegrity() {
        // Assigns or initializes a value used by the algorithm state.
        int connections_checked = 0;
        // Uses a dynamic array to store a resizable sequence of values.
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        // Iterates over a bounded range of values or elements.
        for (int i = 0; i < cur_element_count; i++) {
            // Iterates over a bounded range of values or elements.
            for (int l = 0; l <= element_levels_[i]; l++) {
                // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                // Assigns or initializes a value used by the algorithm state.
                int size = getListCount(ll_cur);
                // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                tableint *data = (tableint *) (ll_cur + 1);
                
                std::unordered_set<tableint> s;
                // Iterates over a bounded range of values or elements.
                for (int j = 0; j < size; j++) {
                    
                    assert(data[j] < cur_element_count);
                    // Assigns or initializes a value used by the algorithm state.
                    assert(data[j] != i);
                    
                    inbound_connections_num[data[j]]++;
                    
                    s.insert(data[j]);
                    
                    connections_checked++;
                // Closes the current C++ scope.
                }
                // Assigns or initializes a value used by the algorithm state.
                assert(s.size() == size);
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }
        // Branches only when the condition is true.
        if (cur_element_count > 1) {
            // Assigns or initializes a value used by the algorithm state.
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            // Iterates over a bounded range of values or elements.
            for (int i=0; i < cur_element_count; i++) {
                
                assert(inbound_connections_num[i] > 0);
                // Assigns or initializes a value used by the algorithm state.
                min1 = std::min(inbound_connections_num[i], min1);
                // Assigns or initializes a value used by the algorithm state.
                max1 = std::max(inbound_connections_num[i], max1);
            // Closes the current C++ scope.
            }
            
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        // Closes the current C++ scope.
        }
        
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    // Closes the current C++ scope.
    }
// Closes the current class, struct, namespace, or block scope.
};
// Executes this C++ statement as part of the surrounding HNSW implementation.
}  // namespace hnswlib
