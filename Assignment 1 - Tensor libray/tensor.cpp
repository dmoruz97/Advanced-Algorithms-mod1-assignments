#include <iostream>

#include "tensor.h"

int main(){

    /**
     *  NO COMPILE TIME KNOWLEDGE
     */

    std::cout << "\nNO COMPILE TIME KNOWLEDGE\n\n";

    /**
     * Creation of tensor of int and initialization by giving dimensions and/or data
     */

    std::vector<int> data = {1, 2, 5, 6, 45, 34, 23, 65, 0, 0, 43, 23, 87, 90, 65, 99, 54, 43, 21, 37, 83, 45, 91, 25, 3, 76, 5, 0, 33, 67,
                             23, 5, 66, 65, 72, 48, 98, 65, 1, 12, 26, 18, 94, 54, 32, 77, 54, 65, 73, 81, 34, 56, 78, 3, 6, 0, 65, 41, 30, 37};
    tensor_t<int> tensor_t_3({5, 3, 4}, data);

    std::cout << "Tensor of rank " << tensor_t_3.get_rank() << ": \n" << tensor_t_3;

    /**
     * Direct access: direct access and modification of an entry providing the indexes
     */

    int& element = tensor_t_3({1, 2, 3});
    std::cout << "Element in position (1, 2, 3): " << element << "\n";

    element = 0;
    std::cout << "Element in position (1, 2, 3): " << element << "\n\n";

    /**
     * Slicing: fixing one index producing a lower rank tensor sharing the same data. The function takes the number of dimension to fix and the initial pointer
     */

    tensor_t<int> sliced_tensor_t_3 = tensor_t_3.slice(0, 2);
    std::cout << "Sliced tensor of rank " << sliced_tensor_t_3.get_rank() << ": \n" << sliced_tensor_t_3;

    /**
     * Slicing: fixing two indexes producing a lower rank tensor sharing the same data. The function takes the number of dimensions to fix and the initial pointers
     */

    std::tuple<size_t, size_t> first_fixed_dimension =  std::make_tuple(0, 2);
    std::tuple<size_t, size_t> second_fixed_dimension =  std::make_tuple(1, 1);
    tensor_t<int> sliced_twice_tensor_t_3 = tensor_t_3.slice({first_fixed_dimension, second_fixed_dimension});
    std::cout << "Sliced tensor of rank " << sliced_twice_tensor_t_3.get_rank() << ": \n" << sliced_twice_tensor_t_3 <<"\n";

    /**
     * Flattening: flattening two indices into one and three into one. The function takes the indexes to flatten, they must be consecutive dimensions.
     */

    tensor_t<int> first_flattened_tensor_t_3 = tensor_t_3.flatten({1,2});
    std::cout << "Flattened tensor of rank " << first_flattened_tensor_t_3.get_rank() << ": \n" << first_flattened_tensor_t_3;

    tensor_t<int> second_flattened_tensor_t_3 = tensor_t_3.flatten({0,1,2});
    std::cout << "Flattened tensor of rank " << second_flattened_tensor_t_3.get_rank() << ": \n" << second_flattened_tensor_t_3 << "\n";

    /**
     * Windowing: generation of a sub-window by changing the starting point and the end point of each index.
     * The function takes a vector of tuple. Each tuple has the form: (index, starting-point, end-point)
     */

    std::tuple<size_t, size_t, size_t> first_tuple = std::make_tuple(0, 1, 4);
    std::tuple<size_t, size_t, size_t> second_tuple = std::make_tuple(1, 0, 2);
    std::tuple<size_t, size_t, size_t> third_tuple = std::make_tuple(2, 0, 2);
    tensor_t<int> window_tensor_t_3 = tensor_t_3.window({first_tuple, second_tuple, third_tuple});
    std::cout << "Sub-window of rank " << window_tensor_t_3.get_rank() << ": \n" << window_tensor_t_3;

    /**
     * Iterator: forward iterator to the full content of the sub-window
     */
    std::cout << "Elements read with the iterator: ";
    for(tensor_t<int>::iterator it = window_tensor_t_3.begin(); it != window_tensor_t_3.end(); ++it)
        std::cout << *it << " ";
    std::cout << "\n\n";


    /**
     * RANK COMPILE TIME INFORMATION
     */

    std::cout << "\nRANK COMPILE TIME INFORMATION\n\n";

    /**
     * Creation of tensor of char and initialization by giving dimensions and/or data
     */

    std::vector<double> double_data = {12.3, 63., 90.4, 56.4, 12.6, 37.1, 54.5, 65.0, 72.9, 64.4, 48.7, 47.3,
                                     41.2, 34.6, 77.5, 53.8, 72.5, 64.1, 73.8, 74.3, 88.2, 65.7, 63.5, 42.7};
    tensor<double,3> tensor_t_r_3({2, 4, 3}, double_data);

    std::cout << "Tensor of rank 3: \n" << tensor_t_r_3;

    /**
     * Direct access: direct access and modification of an entry providing the indexes
     */

    double& element_t_r = tensor_t_r_3({1, 2, 1});
    std::cout << "Element in position (1, 2, 1): " << element_t_r << "\n";

    element_t_r = 23.1;
    std::cout << "Element in position (1, 2, 1): " << element_t_r << "\n\n";

    /**
     * Slicing: fixing one index producing a lower rank tensor sharing the same data. The function takes the number of dimension to fix and the initial pointer
     */

    tensor<double, 2> sliced_tensor_t_r_3 = tensor_t_r_3.slice(1, 1);
    std::cout << "Sliced tensor of rank 2: \n" << sliced_tensor_t_r_3;

    /**
     * Flattening: flattening two indices into one. The function takes the indexes to flatten, they must be consecutive dimensions.
     */

    tensor<double,2> flattened_tensor_t_r_3 = tensor_t_r_3.flatten(1,2);
    std::cout << "Flattened tensor of rank 3: \n" << flattened_tensor_t_r_3;

    /**
     * Iterator: forward iterator to the full content of the flattened tensor
     */
    std::cout << "Elements read with the iterator: ";
    for(tensor<double, 2>::iterator it = flattened_tensor_t_r_3.begin(); it != flattened_tensor_t_r_3.end(); ++it)
        std::cout << *it << " ";

    std::cout << "\n\n";


    /**
     * Windowing: generation of a sub-window by changing the starting point and the end point of each index.
     * The function takes a vector of tuple. Each tuple has the form: (index, starting-point, end-point)
     */

    std::tuple<size_t, size_t, size_t> first_tuple_t_r = std::make_tuple(0, 0, 1);
    std::tuple<size_t, size_t, size_t> second_tuple_t_r = std::make_tuple(1, 1, 3);
    std::tuple<size_t, size_t, size_t> third_tuple_t_r = std::make_tuple(2, 1, 2);
    tensor<double,3> window_tensor_t_r_3 = tensor_t_r_3.window({first_tuple_t_r, second_tuple_t_r, third_tuple_t_r});
    std::cout << "Sub-window of rank 3: \n" << window_tensor_t_r_3;



    /**
     * PARTIAL SPECIALIZATION OF TENSOR<T,R>
     */

    std::cout << "\nPARTIAL SPECIALIZATION OF TENSOR<T,R>\n\n";

    /**
    * Creation of tensor of char and initialization by giving dimensions and/or data
    */

    std::vector<int> int_data = {12, 63, 90, 56, 12, 37, 54, 65, 72, 64};
    tensor<int,1> spec_tensor(10, int_data);

    std::cout << "Tensor of rank 1: \n" << spec_tensor << "\n";

    /**
     * Iterator: forward iterator to the full content of the tensor
     */
    std::cout << "Elements read with the iterator: ";
    for(tensor<int,1>::iterator it = spec_tensor.begin(); it != spec_tensor.end(); ++it)
        std::cout << *it << " ";

    std::cout << "\n\n";
    /**
     * Direct access: direct access and modification of an entry providing the index
     */

    int& spec_element = spec_tensor(2);
    std::cout << "Element in position (2): " << spec_element << "\n";

    spec_element = 23;
    std::cout << "Element in position (2): " << spec_element << "\n\n";

    /**
     * Slicing: fixing one index producing a lower rank tensor sharing the same data. The function takes the index to fix
     */

    int sliced_spec_tensor = spec_tensor.slice(1);
    std::cout << "Sliced tensor of rank 1: " << sliced_spec_tensor << "\n";

    /**
     * Flattening: since the tensor has only one dimension, it's not possible to flatten two indices into one
     */

    /**
     * Windowing: generation of a sub-window by changing the starting point and the end point of the index.
     * The function takes the starting-point and the end-point of the window
     */

    tensor<int,1> window_spec_tensor = spec_tensor.window(2,7);
    std::cout << "Sub-window of rank 1: \n" << window_spec_tensor << "\n";


    /**
     * DON'T DO IT: FLATTEN A SLICED OR WINDOWED TENSOR
     */

    std::cout << "\nDON'T DO IT: FLATTEN A SLICED OR WINDOWED TENSOR\n\n";

    std::vector<int> data_to_pass = {12, 63, 90, 56, 12, 37, 54, 65, 72, 64,
                                     43, 81, 54, 65, 82, 38, 71, 90, 61, 72,
                                     32, 48, 93, 84, 64, 72, 99, 26, 74, 44,
                                     64, 27, 73, 62, 75, 65};
    tensor_t<int> tensor_t_2({2,9,2}, data_to_pass);
    std::cout << "Tensor with rank " << tensor_t_2.get_rank() << ": \n"<< tensor_t_2;

    /**
     * Sliced tensor:
     */

    tensor_t<int> sliced_tensor_t_2 = tensor_t_2.slice(1, 2);

    std::cout << "Sliced tensor with rank " << sliced_tensor_t_2.get_rank() << ": \n" << sliced_tensor_t_2;

    /**
     * Flattening a sliced tensor:
     */

    tensor_t<int> flattened_sliced_tensor_t_2 = sliced_tensor_t_2.flatten({0,1});

    std::cout << "If we try to flatten a sliced tensor we notice that printed data do not correspond to the expected values: \n" << flattened_sliced_tensor_t_2 << "\n";

    return 0;
}

