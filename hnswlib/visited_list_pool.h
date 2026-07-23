// Ensures this header is included only once in each translation unit.
#pragma once


#include <mutex>
#include <string.h>
#include <deque>

// Places the following declarations inside the hnswlib namespace.
namespace hnswlib {
// Creates a shorter type alias used by the implementation.
typedef unsigned short int vl_type;

// Declares a C++ type that groups related data and behavior.
class VisitedList {
 // Sets the access level for the following class members.
 public:
    
    vl_type curV;
    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    vl_type *mass;
    
    unsigned int numelements;

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    VisitedList(int numelements1) {
        // Assigns or initializes a value used by the algorithm state.
        curV = -1;
        // Assigns or initializes a value used by the algorithm state.
        numelements = numelements1;
        // Manages heap memory; pointers here own or reference dynamically allocated storage.
        mass = new vl_type[numelements];
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    void reset() {
        
        curV++;
        // Branches only when the condition is true.
        if (curV == 0) {
            // Copies or initializes raw memory through pointers, usually for compact vector or link storage.
            memset(mass, 0, sizeof(vl_type) * numelements);
            
            curV++;
        // Closes the current C++ scope.
        }
    // Closes the current C++ scope.
    }

    // Manages heap memory; pointers here own or reference dynamically allocated storage.
    ~VisitedList() { delete[] mass; }
// Closes the current class, struct, namespace, or block scope.
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

// Declares a C++ type that groups related data and behavior.
class VisitedListPool {
    
    std::deque<VisitedList *> pool;
    // Uses a mutex or lock object to protect shared state during threaded operations.
    std::mutex poolguard;
    
    int numelements;

 // Sets the access level for the following class members.
 public:
    // Executes this C++ statement as part of the surrounding HNSW implementation.
    VisitedListPool(int initmaxpools, int numelements1) {
        // Assigns or initializes a value used by the algorithm state.
        numelements = numelements1;
        // Iterates over a bounded range of values or elements.
        for (int i = 0; i < initmaxpools; i++)
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            pool.push_front(new VisitedList(numelements));
    // Closes the current C++ scope.
    }

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    VisitedList *getFreeVisitedList() {
        // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
        VisitedList *rez;
        // Opens a new C++ scope for the preceding declaration or control statement.
        {
            // Uses a mutex or lock object to protect shared state during threaded operations.
            std::unique_lock <std::mutex> lock(poolguard);
            // Branches only when the condition is true.
            if (pool.size() > 0) {
                // Assigns or initializes a value used by the algorithm state.
                rez = pool.front();
                
                pool.pop_front();
            // Executes this C++ statement as part of the surrounding HNSW implementation.
            } else {
                // Manages heap memory; pointers here own or reference dynamically allocated storage.
                rez = new VisitedList(numelements);
            // Closes the current C++ scope.
            }
        // Closes the current C++ scope.
        }
        // Uses a pointer to access a member of the object it points to.
        rez->reset();
        // Returns the computed value or object to the caller.
        return rez;
    // Closes the current C++ scope.
    }

    // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
    void releaseVisitedList(VisitedList *vl) {
        // Uses a mutex or lock object to protect shared state during threaded operations.
        std::unique_lock <std::mutex> lock(poolguard);
        
        pool.push_front(vl);
    // Closes the current C++ scope.
    }

    // Executes this C++ statement as part of the surrounding HNSW implementation.
    ~VisitedListPool() {
        // Repeats the block while the condition remains true.
        while (pool.size()) {
            // Declares or uses a pointer variable that stores an address to memory rather than the value itself.
            VisitedList *rez = pool.front();
            
            pool.pop_front();
            // Manages heap memory; pointers here own or reference dynamically allocated storage.
            delete rez;
        // Closes the current C++ scope.
        }
    // Closes the current C++ scope.
    }
// Closes the current class, struct, namespace, or block scope.
};
// Executes this C++ statement as part of the surrounding HNSW implementation.
}  // namespace hnswlib
