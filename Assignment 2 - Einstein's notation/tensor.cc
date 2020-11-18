#include<iostream>
#include<chrono>

#include"tensor.h"

int main() {
    srand(time(nullptr));

    // EINSTEIN'S NOTATIONS FOR DYNAMIC RANK TENSORS
    std::cout << "\n****************************\n";
    std::cout << "*** DYNAMIC RANK TENSORS ***";
    std::cout << "\n****************************";
    
    std::cout << "\n\nFIRST EXAMPLE";
    tensor::tensor<int> a({5, 2});
    tensor::tensor<int> b({2, 5});

    for(auto iter=a.begin(); iter!=a.end(); ++iter)
        *iter = rand() % 100;
    for(auto iter=b.begin(); iter!=b.end(); ++iter)
        *iter = rand() % 100;

    tensor::tensor<int> c = a({tensor::index::i, tensor::index::j}) * b({tensor::index::j, tensor::index::i});
    int t_tensor = static_cast<int>(c);

    tensor::tensor<int> d = a({tensor::index::i, tensor::index::j}) + b({tensor::index::j, tensor::index::i});

    std::cout << "\n\nTensor a: ";
    for(auto iter=a.begin(); iter!=a.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::cout << "Tensor b: ";
    for(auto iter=b.begin(); iter!=b.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::cout << "Tensor c = a * b: ";
    for(auto iter=c.begin(); iter!=c.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::cout << "Tensor c converted to type T: ";
    std:: cout << t_tensor << '\n';

    std::cout << "Tensor d = a + b: ";
    for(auto iter = d.begin(); iter != d.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';



    std::cout << "\n\nSECOND EXAMPLE";

    tensor::tensor<int> a2(2,3,5);
    tensor::tensor<int> b2(2,3,4);
    for(auto iter=a2.begin(); iter!=a2.end(); ++iter)
        *iter = rand() % 100;
    for(auto iter=b2.begin(); iter!=b2.end(); ++iter)
        *iter = rand() % 100;

    //tensor::tensor<int> c2(2,5,4);
    //c2({tensor::index::k, tensor::index::n, tensor::index::m}) = a2({tensor::index::k, tensor::index::l, tensor::index::n}) * b2({tensor::index::l, tensor::index::m});

    tensor::tensor<int> c2(5,4);
    c2 = a2({tensor::index::k, tensor::index::l, tensor::index::n}) * b2({tensor::index::k, tensor::index::l, tensor::index::m});

    std::cout << "\n\nTensor a2: ";
    for(auto iter=a2.begin(); iter!=a2.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::cout << "Tensor b2: ";
    for(auto iter=b2.begin(); iter!=b2.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::cout << "Tensor c2 = a2 * b2 : ";
    for(auto iter=c2.begin(); iter!=c2.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';



    std::cout << "\n\nTHIRD EXAMPLE";
    tensor::tensor<int> a3({2,3,4});
    tensor::tensor<int> b3({3,2,4});

    for(auto iter=a3.begin(); iter!=a3.end(); ++iter)
        *iter = rand() % 100;
    for(auto iter=b3.begin(); iter!=b3.end(); ++iter)
        *iter = rand() % 100;

    tensor::tensor<int> c3 = a3({tensor::index::i, tensor::index::j, tensor::index::k}) - b3({tensor::index::j, tensor::index::i, tensor::index::k});

    std::cout << "\n\nTensor a3: ";
    for(auto iter=a3.begin(); iter!=a3.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::cout << "Tensor b3: ";
    for(auto iter=b3.begin(); iter!=b3.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::cout << "Tensor c3 = a3 - b3: ";
    for(auto iter=c3.begin(); iter!=c3.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';



    std::cout << "\n\nFOURTH EXAMPLE (trace)";
    tensor::tensor<int> a4({3,3});
    for(auto iter=a4.begin(); iter!=a4.end(); ++iter)
        *iter = rand() % 100;

    tensor::tensor<int> trace_result = static_cast<tensor::tensor<int>> (a4({tensor::index::i, tensor::index::i}));

    std::cout << "\n\nTensor a4: ";
    for(auto iter=a4.begin(); iter!=a4.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::cout << "Tensor 'trace_result' (result): ";
    for(auto iter=trace_result.begin(); iter!=trace_result.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    
    
    std::cout << "\n\nFIFTH EXAMPLE (small expression)";
    tensor::tensor<int> one({1,2,3,4});
    std::cout << "\n\nTensor one: ";
    for(auto iter=one.begin(); iter!=one.end(); ++iter) {
        *iter = rand() % 100;
        std::cout << *iter << ' ';
    }
    std:: cout << '\n';
    
    tensor::tensor<int> two({3,4});
    std::cout << "Tensor two: ";
    for(auto iter=two.begin(); iter!=two.end(); ++iter) {
        *iter = rand() % 100;
        std::cout << *iter << ' ';
    }
    std:: cout << '\n';
    
    tensor::tensor<int> three({1,5});
    std::cout << "Tensor three: ";
    for(auto iter=three.begin(); iter!=three.end(); ++iter) {
        *iter = rand() % 100;
        std::cout << *iter << ' ';
    }
    std:: cout << '\n';
    
    tensor::tensor<int> four({2,5});
    std::cout << "Tensor four: ";
    for(auto iter=four.begin(); iter!=four.end(); ++iter) {
        *iter = rand() % 100;
        std::cout << *iter << ' ';
    }
    std:: cout << '\n';
    
    tensor::tensor<int> ris = (one({tensor::index::i, tensor::index::j, tensor::index::k, tensor::index::l}) * two({tensor::index::k, tensor::index::l}) * three({tensor::index::i, tensor::index::m})) + four({tensor::index::j,tensor::index::m});
    
    std::cout << "Result of expression (one * two * three + four): ";
    for(auto iter=ris.begin(); iter!=ris.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';





    // EINSTEIN'S NOTATIONS FOR FIXED RANK TENSORS
    std::cout << "\n\n\n**************************\n";
    std::cout << "*** FIXED RANK TENSORS ***";
    std::cout << "\n**************************";

    std::cout << "\n\nFIRST EXAMPLE";
    std::vector<size_t> dimensions = {2,3,4,2};
    tensor::tensor<int, tensor::rank<4>> fixed_rank(dimensions);
    tensor::index::index_tr<1> a_prova_fixed;
    tensor::index::index_tr<2> b_prova_fixed;
    tensor::index::index_tr<1> c_prova_fixed;
    tensor::index::index_tr<8> d_prova_fixed;

    for(auto iter = fixed_rank.begin(); iter != fixed_rank.end(); ++iter)
        *iter = rand() % 100;

    std::cout << "\n\nFixed rank: ";
    for(auto iter=fixed_rank.begin(); iter!=fixed_rank.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    tensor::tensor<int, tensor::rank<2>> t_prova_fixed = fixed_rank(a_prova_fixed, b_prova_fixed, c_prova_fixed, d_prova_fixed);

    std::cout << "Fixed rank after operator (): ";
    for(auto iter=t_prova_fixed.begin(); iter!=t_prova_fixed.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';



    std::cout << "\n\nSECOND EXAMPLE";

    std::vector<size_t> dimensions1 = {2,3,4};
    tensor::tensor<int, tensor::rank<3>> frt1(dimensions1);
    for(auto iter=frt1.begin(); iter!=frt1.end(); ++iter)
        *iter = rand() % 100;
    std::cout << "\n\nFrt1: ";
    for(auto iter=frt1.begin(); iter!=frt1.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::vector<size_t> dimensions2 = {4,3,2};
    tensor::tensor<int, tensor::rank<3>> frt2(dimensions2);
    for(auto iter=frt2.begin(); iter!=frt2.end(); ++iter)
        *iter = rand() % 100;
    std::cout << "Frt2: ";
    for(auto iter=frt2.begin(); iter!=frt2.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    tensor::index::index_tr<1> ind_fix1;
    tensor::index::index_tr<2> ind_fix2;
    tensor::index::index_tr<3> ind_fix3;

    tensor::tensor<int, tensor::rank<3>> frt_result = frt1(ind_fix1,ind_fix2,ind_fix3) + frt2(ind_fix3,ind_fix2,ind_fix1);

    std::cout << "Result after operator +: ";
    for(auto iter=frt_result.begin(); iter!=frt_result.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';



    std::cout << "\n\nTHIRD EXAMPLE (Proxy tensor)\n\n";

    tensor::proxy_tensor<int, tensor::rank<3>, tensor::index::index_tr<1>, tensor::index::index_tr<2>, tensor::index::index_tr<3>> first_proxy_tensor({2,3,4});
    for(auto iter=first_proxy_tensor.begin(); iter!=first_proxy_tensor.end(); ++iter)
        *iter = rand() % 100;
    std::cout << "First proxy tensor: ";
    for(auto iter=first_proxy_tensor.begin(); iter!=first_proxy_tensor.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    tensor::proxy_tensor<int, tensor::rank<2>, tensor::index::index_tr<4>, tensor::index::index_tr<2>> second_proxy_tensor({5,3});
    for(auto iter=second_proxy_tensor.begin(); iter!=second_proxy_tensor.end(); ++iter)
        *iter = rand() % 100;
    std::cout << "Second proxy tensor: ";
    for(auto iter=second_proxy_tensor.begin(); iter!=second_proxy_tensor.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    tensor::tensor<int, tensor::rank<3>> tensor_product_result = first_proxy_tensor * second_proxy_tensor;

    std::cout << "Result (operator *): ";
    for(auto iter=tensor_product_result.begin(); iter!=tensor_product_result.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';






    // EINSTEIN'S NOTATIONS FOR RANK ONE TENSORS
    std::cout << "\n\n**********************\n";
    std::cout << "*** RANK 1 TENSORS ***\n";
    std::cout << "**********************\n";

    tensor::tensor<int,tensor::rank<1>> aa(2);
    for(auto iter=aa.begin(); iter!=aa.end(); ++iter)
        *iter = rand() % 100;

    tensor::tensor<int,tensor::rank<1>> bb(2);
    for(auto iter=bb.begin(); iter!=bb.end(); ++iter)
        *iter = rand() % 100;

    std::cout << "\nTensor aa: ";
    for(auto iter=aa.begin(); iter!=aa.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    std::cout << "Tensor bb: ";
    for(auto iter=bb.begin(); iter!=bb.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << '\n';

    tensor::tensor<int,tensor::rank<1>> rr_x = static_cast<tensor::tensor<int,tensor::rank<1>>> (aa(tensor::index::i) * bb(tensor::index::i));
    tensor::tensor<int,tensor::rank<1>> rr_plus = static_cast<tensor::tensor<int,tensor::rank<1>>> (aa(tensor::index::i) + bb(tensor::index::i));
    tensor::tensor<int,tensor::rank<1>> rr_minus = static_cast<tensor::tensor<int,tensor::rank<1>>> (aa(tensor::index::i) - bb(tensor::index::i));

    std::cout << "\nResult (operator *): ";
    for(auto iter=rr_x.begin(); iter!=rr_x.end(); ++iter)
        std::cout << *iter << ' ';

    std::cout << "\nResult (operator +): ";
    for(auto iter=rr_plus.begin(); iter!=rr_plus.end(); ++iter)
        std::cout << *iter << ' ';

    std::cout << "\nResult (operator -): ";
    for(auto iter=rr_minus.begin(); iter!=rr_minus.end(); ++iter)
        std::cout << *iter << ' ';
    std:: cout << "\n\n";






    // OPERATOR () - TIME TEST
    std::cout << "\n*******************************\n";
    std::cout << "*** Operator () - Time test ***\n";
    std::cout << "*******************************\n";

    std::vector<size_t> dimensions_test = {100,2,4,3,2,100};

    // For dynamic rank tensor
    tensor::tensor<int> test_dynamic(dimensions_test);
    for(auto iter=test_dynamic.begin(); iter!=test_dynamic.end(); ++iter)
        *iter = rand() % 100;

    auto start_time_dynamic = std::chrono::high_resolution_clock::now();
    tensor::tensor<int> result_dynamic = static_cast<tensor::tensor<int>> (test_dynamic({tensor::index::i, tensor::index::j,tensor::index::k,tensor::index::l,tensor::index::j,tensor::index::i}));
    auto end_time_dynamic = std::chrono::high_resolution_clock::now();
    double elapsed_time_dynamic = std::chrono::duration<double>(end_time_dynamic-start_time_dynamic).count();

    std::cout << "\nDynamic rank tensor: " << elapsed_time_dynamic << '\n';

    // For fixed rank tensor
    tensor::tensor<int,tensor::rank<6>> test_fixed(dimensions_test);
    for(auto iter=test_fixed.begin(); iter!=test_fixed.end(); ++iter)
        *iter = rand() % 100;
    tensor::index::index_tr<1> ind_test_1;
    tensor::index::index_tr<2> ind_test_2;
    tensor::index::index_tr<3> ind_test_3;
    tensor::index::index_tr<4> ind_test_4;

    auto start_time_fixed = std::chrono::high_resolution_clock::now();
    tensor::tensor<int,tensor::rank<2>> result_fixed = test_fixed(ind_test_1,ind_test_2,ind_test_3,ind_test_4,ind_test_2,ind_test_1);
    auto end_time_fixed = std::chrono::high_resolution_clock::now();
    double elapsed_time_fixed = std::chrono::duration<double>(end_time_fixed-start_time_fixed).count();

    std::cout << "Fixed rank tensor: " << elapsed_time_fixed << "\n\n";


    // OPERATOR * - TIME TEST
    std::cout << "\n******************************\n";
    std::cout << "*** Operator * - Time test ***\n";
    std::cout << "******************************\n";

    std::vector<size_t> dimensions1_test = {5,10,4,20};
    std::vector<size_t> dimensions2_test = {4,20,6};

    tensor::index::index_tr<5> ind_test_5;

    // For dynamic rank tensor
    tensor::tensor<int> test_dynamic1(dimensions1_test);
    for(auto iter=test_dynamic1.begin(); iter!=test_dynamic1.end(); ++iter)
        *iter = rand() % 100;
    tensor::tensor<int> test_dynamic2(dimensions2_test);
    for(auto iter=test_dynamic2.begin(); iter!=test_dynamic2.end(); ++iter)
        *iter = rand() % 100;

    auto start_time_dynamic1 = std::chrono::high_resolution_clock::now();

    tensor::tensor<int> result_dynamic1 = static_cast<tensor::tensor<int>> (test_dynamic1({tensor::index::i, tensor::index::j,tensor::index::k,tensor::index::l}) * test_dynamic2({tensor::index::k, tensor::index::l,tensor::index::m}));

    auto end_time_dynamic1 = std::chrono::high_resolution_clock::now();
    double elapsed_time_dynamic1 = std::chrono::duration<double>(end_time_dynamic1-start_time_dynamic1).count();

    std::cout << "\nDynamic rank tensor: " << elapsed_time_dynamic1 << '\n';

    // For fixed rank tensor

    tensor::tensor<int, tensor::rank<4>> test_1(dimensions1_test);
    tensor::tensor<int, tensor::rank<3>> test_2(dimensions2_test);
    for(auto iter=test_1.begin(); iter!=test_1.end(); ++iter)
        *iter = rand() % 100;
    for(auto iter=test_2.begin(); iter!=test_2.end(); ++iter)
        *iter = rand() % 100;



    tensor::proxy_tensor<int, tensor::rank<4>, tensor::index::index_tr<1>, tensor::index::index_tr<2>, tensor::index::index_tr<3>, tensor::index::index_tr<4>> test_fixed1(dimensions1_test);
    for(auto iter=test_fixed1.begin(); iter!=test_fixed1.end(); ++iter)
        *iter = rand() % 100;

    tensor::proxy_tensor<int, tensor::rank<3>, tensor::index::index_tr<3>, tensor::index::index_tr<4>, tensor::index::index_tr<5>> test_fixed2(dimensions2_test);
    for(auto iter=test_fixed2.begin(); iter!=test_fixed2.end(); ++iter)
        *iter = rand() % 100;

    auto start_time_fixed2 = std::chrono::high_resolution_clock::now();

    tensor::tensor<int,tensor::rank<3>> result_fixed2 = test_fixed1 * test_fixed2;

    auto end_time_fixed2 = std::chrono::high_resolution_clock::now();
    double elapsed_time_fixed2 = std::chrono::duration<double>(end_time_fixed2-start_time_fixed2).count();

    std::cout << "Fixed rank tensor: " << elapsed_time_fixed2 << "\n\n";

}
