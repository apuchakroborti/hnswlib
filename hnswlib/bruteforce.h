// Ensures this header is included only once in each translation unit.
#pragma once

#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>

// Places the following declarations inside the hnswlib namespace.
namespace hnswlib {
// Declares that the following class or function is templated over one or more types.
template<typename dist_t>
// Declares a C++ type that groups related data and behavior.
class BruteforceSearch : public AlgorithmInterface<dist_t> {
 // Sets the access level for the following class members.
 public:
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    char *data_;
    
    size_t maxelements_;
    
    size_t cur_element_count;
    
    size_t size_per_element_;

    
    size_t data_size_;
    
    DISTFUNC <dist_t> fstdistfunc_;
    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    void *dist_func_param_;
    // Uses a mutex or lock object to protect shared state during threaded operations.
    std::mutex index_lock;

    
    std::unordered_map<labeltype, size_t > dict_external_to_internal;


    // Dereferences or casts through a pointer to read or write the object stored at an address.
    BruteforceSearch(SpaceInterface <dist_t> *s)
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        : data_(nullptr),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            maxelements_(0),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            cur_element_count(0),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            size_per_element_(0),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            data_size_(0),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            dist_func_param_(nullptr) {
    // Closes the current C++ scope.
    }


    // Dereferences or casts through a pointer to read or write the object stored at an address.
    BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
        // Executes this C++ statement as part of the surrounding HNSW implementation.
        : data_(nullptr),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            maxelements_(0),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            cur_element_count(0),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            size_per_element_(0),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            data_size_(0),
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            dist_func_param_(nullptr) {
        
        loadIndex(location, s);
    // Closes the current C++ scope.
    }


    // Dereferences or casts through a pointer to read or write the object stored at an address.
    BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements) {
        // Assigns or initializes a value used by the algorithm state.
        maxelements_ = maxElements;
        // Uses a pointer to access a member of the object it points to.
        data_size_ = s->get_data_size();
        // Uses a pointer to access a member of the object it points to.
        fstdistfunc_ = s->get_dist_func();
        // Uses a pointer to access a member of the object it points to.
        dist_func_param_ = s->get_dist_func_param();
        // Assigns or initializes a value used by the algorithm state.
        size_per_element_ = data_size_ + sizeof(labeltype);
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        data_ = (char *) malloc(maxElements * size_per_element_);
        // Branches only when the condition is true.
        if (data_ == nullptr)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
        // Assigns or initializes a value used by the algorithm state.
        cur_element_count = 0;
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    ~BruteforceSearch() {
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        free(data_);
    // Closes the current C++ scope.
    }


    // Manages heap memory; pointers here own or reference dynamically allocated storage.
    void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) {
        
        int idx;
        // Opens a new C++ scope for the preceding declaration or control statement.
        {
            // Uses a mutex or lock object to protect shared state during threaded operations.
            std::unique_lock<std::mutex> lock(index_lock);

            // Assigns or initializes a value used by the algorithm state.
            auto search = dict_external_to_internal.find(label);
            // Branches only when the condition is true.
            if (search != dict_external_to_internal.end()) {
                // Uses a pointer to access a member of the object it points to.
                idx = search->second;
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            } else {
                // Branches only when the condition is true.
                if (cur_element_count >= maxelements_) {
                    // Handles an exceptional error path by throwing or catching a C++ exception.
                    throw std::runtime_error("The number of elements exceeds the specified limit\n");
                // Closes the current C++ scope.
                }
                // Assigns or initializes a value used by the algorithm state.
                idx = cur_element_count;
                // Assigns or initializes a value used by the algorithm state.
                dict_external_to_internal[label] = idx;
                
                cur_element_count++;
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }
        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
    // Closes the current C++ scope.
    }


    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void removePoint(labeltype cur_external) {
        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock<std::mutex> lock(index_lock);

        // Assigns or initializes a value used by the algorithm state.
        auto found = dict_external_to_internal.find(cur_external);
        // Branches only when the condition is true.
        if (found == dict_external_to_internal.end()) {
            // Returns the computed value or object to the caller.
            return;
        // Closes the current C++ scope.
        }

        
        dict_external_to_internal.erase(found);

        // Uses a pointer to access a member of the object it points to.
        size_t cur_c = found->second;
        // Dereferences or casts through a pointer to read or write the object stored at an address.
        labeltype label = *((labeltype*)(data_ + size_per_element_ * (cur_element_count-1) + data_size_));
        // Assigns or initializes a value used by the algorithm state.
        dict_external_to_internal[label] = cur_c;
        // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
        memcpy(data_ + size_per_element_ * cur_c,
                // Dereferences or casts through a pointer to read or write the object stored at an address.
                data_ + size_per_element_ * (cur_element_count-1),
                
                data_size_+sizeof(labeltype));
        
        cur_element_count--;
    // Closes the current C++ scope.
    }


    // Uses a heap-backed priority queue to keep best candidates ordered by distance.
    std::priority_queue<std::pair<dist_t, labeltype >>
    // Uses a void pointer so generic vector bytes can be passed without fixing their concrete element type here.
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        // Assigns or initializes a value used by the algorithm state.
        assert(k <= cur_element_count);
        // Uses a heap-backed priority queue to keep best candidates ordered by distance.
        std::priority_queue<std::pair<dist_t, labeltype >> topResults;
        // Assigns or initializes a value used by the algorithm state.
        dist_t lastdist = std::numeric_limits<dist_t>::max();
        // Iterates over a bounded range of values or elements.
        for (int i = 0; i < cur_element_count; i++) {
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            // Branches only when the condition is true.
            if (dist <= lastdist || topResults.size() < k) {
                // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
                labeltype label = *((labeltype *) (data_ + size_per_element_ * i + data_size_));
                // Dereferences or casts through a pointer to read or write the object stored at an address.
                if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                    
                    topResults.emplace(dist, label);
                    // Branches only when the condition is true.
                    if (topResults.size() > k)
                        
                        topResults.pop();
                    // Branches only when the condition is true.
                    if (!topResults.empty())
                        // Assigns or initializes a value used by the algorithm state.
                        lastdist = topResults.top().first;
                // Closes the current C++ scope.
                }
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }
        // Returns the computed value or object to the caller.
        return topResults;
    // Closes the current C++ scope.
    }


    // Uses a reference or address operation so the code can avoid copying or can pass an address onward.
    void saveIndex(const std::string &location) {
        
        std::ofstream output(location, std::ios::binary);
        
        std::streampos position;

        
        writeBinaryPOD(output, maxelements_);
        
        writeBinaryPOD(output, size_per_element_);
        
        writeBinaryPOD(output, cur_element_count);

        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        output.write(data_, maxelements_ * size_per_element_);

        
        output.close();
    // Closes the current C++ scope.
    }


    // Dereferences or casts through a pointer to read or write the object stored at an address.
    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {
        
        std::ifstream input(location, std::ios::binary);
        
        std::streampos position;

        
        readBinaryPOD(input, maxelements_);
        
        readBinaryPOD(input, size_per_element_);
        
        readBinaryPOD(input, cur_element_count);

        // Uses a pointer to access a member of the object it points to.
        data_size_ = s->get_data_size();
        // Uses a pointer to access a member of the object it points to.
        fstdistfunc_ = s->get_dist_func();
        // Uses a pointer to access a member of the object it points to.
        dist_func_param_ = s->get_dist_func_param();
        // Assigns or initializes a value used by the algorithm state.
        size_per_element_ = data_size_ + sizeof(labeltype);
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        data_ = (char *) malloc(maxelements_ * size_per_element_);
        // Branches only when the condition is true.
        if (data_ == nullptr)
            // Handles an exceptional error path by throwing or catching a C++ exception.
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate data");

        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        input.read(data_, maxelements_ * size_per_element_);

        
        input.close();
    // Closes the current C++ scope.
    }
// Closes the current class, struct, namespace, or block scope.
};
// Executes this C++ statement as part of the surrounding HNSW implementation.
}  // namespace hnswlib
