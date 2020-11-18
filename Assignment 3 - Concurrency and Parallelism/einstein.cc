#include<iostream>
#include"tensor.h"

using namespace Tensor;


std::ostream & operator << (std::ostream& out, Index_Set<>) { return out; }
template<unsigned id, unsigned... ids>
std::ostream & operator << (std::ostream& out, Index_Set<id, ids...>) {
    return out << id << ' ' << Index_Set<ids...>();
}




int main() {
    
    std::cout << "START THREAD WORK\n\n";
    
    // Declaration of some tensors
    tensor<int> t1(2,3);
    tensor<int> t2(3,4,5);
    tensor<int,rank<3>> t3(2,4,5);
    tensor<int,rank<5>> t4(2,3,4,5,6);
    tensor<int> t5(2,3);
    
    // Declaration of some indices
    auto i = new_index;
    auto j = new_index;
    auto k = new_index;
    auto l = new_index;
    auto m = new_index;
    auto n = new_index;
    auto o = new_index;
    
    // Initialization of all tensors
    int count=0;
    for(auto iter=t1.begin(); iter!=t1.end(); ++iter)
                *iter = count++;
    
    count=0;
    for(auto iter=t2.begin(); iter!=t2.end(); ++iter)
                *iter = count++;
    
    count=0;
    for(auto iter=t3.begin(); iter!=t3.end(); ++iter)
                *iter = count++;
    
    count=0;
    for(auto iter=t4.begin(); iter!=t4.end(); ++iter)
                *iter = count++;
    
    // First expression
    tensor<int> ris(3,4,5);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    ris(i,k,l) = t1(i,j)*t2(j,k,l)+t3(i,k,l);
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time-start_time).count();
    
    std::cout << "\n\nExpression result: ";
    for(auto iter=ris.begin(); iter!=ris.end(); ++iter)
                std::cout << *iter << ' ';
    std::cout << '\n';
    
    std::cout << "\nExpression time: " << elapsed_time << "\n\n";
}
