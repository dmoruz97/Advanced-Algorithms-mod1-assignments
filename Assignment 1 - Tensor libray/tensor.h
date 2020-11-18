#include <cstring>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

template <typename T>
class tensor_t;

template <typename T, size_t R>
class tensor;

template <typename T>
std::ostream& operator << (std::ostream& out, tensor_t<T>&);

template <typename T, size_t R>
std::ostream& operator << (std::ostream& out, tensor<T, R>&);

template <typename T>
std::ostream& operator << (std::ostream& out, tensor<T, 1>&);


// Namespace contains the iterators used by all tensors
namespace iterator_namespace {

    template<typename T,typename index,size_t R=0>
    class iterator {
        public:
            // Operator *
            T& operator *() const {
                return *it_ptr;
            }
        
            // Operator =
            T& operator =(T arg){
                *it_ptr = arg;
                return *it_ptr;
            }

            // Operator ==
            bool operator ==(iterator& i) {
                return it_ptr == i.it_ptr;
            }

            // Operator !=
            bool operator !=(iterator i) {
                return it_ptr != i.it_ptr;
            }

            // Operator ++
            iterator& operator ++(){ // Prefix increment operator
                size_t ind = width.size()-1;

               ++(temp_vect[ind]);

                while(temp_vect[ind] == width[ind] && ind > 0){
                    temp_vect[ind]=0;
                    --ind;
                    ++temp_vect[ind];
                }
                
                it_ptr=start_ptr;

                for (int i=0; i!=width.size(); ++i) {
                    //std::cout << "temp " << temp_vect[i];
                    it_ptr += stride[i] * (temp_vect[i]);
                }
               
                return *this;
            }
            iterator operator ++(int){   //  Postfix increment operator
                iterator temp = *this;
                ++(*this);
                return temp;
            }
    
            friend class tensor_t<T>;
            friend class tensor<T,R>;

        private:
            iterator(const index& width_a, const index&  stride_a, T* ptr) : width(width_a),stride(stride_a), it_ptr(ptr),start_ptr(ptr){
                    for (int i=0;i!=width.size();++i){
                    temp_vect.push_back(0);
                }
            }
            const index& width;   // Reference to width
            const index& stride;  // Reference to stride
            T* it_ptr; // To optimize access
            T* start_ptr;

            std::vector<size_t> temp_vect;    // support vector
    };

    // Iterator over a single index
    template<typename T,size_t R=0>
    class index_iterator {
        public:
            // Operator *
            T& operator *() const {
                return *it_ptr;
            }
        
            // Operator =
            T& operator =(T arg){
                *it_ptr = arg;
                return *it_ptr;
            }

            // Operator ==
            bool operator ==(index_iterator& i) {
                return it_ptr == i.it_ptr;
            }
        
            // Operator !=
            bool operator !=(index_iterator i) {
                return it_ptr != i.it_ptr;
            }

            // Operator ++
            index_iterator& operator ++(){ // Prefix increment operator
                it_ptr += stride;
                return *this;
            }
            index_iterator operator ++(int){   //  Postfix increment operator
                index_iterator temp = *this;
                ++(*this);
                return temp;
            }
        
            friend class tensor_t<T>;
            friend class tensor<T,R>;
            
        private:
            // ind must starts from 0 (as the index of the tensor)
            index_iterator(size_t stride_n, T* start_ptr) : stride(stride_n), it_ptr(start_ptr){}

            size_t stride;   // every s positions there is a value that belongs to the index specified by user
            T* it_ptr;
    };
}
// END namespace iterator_namespace

/* Class tensor for compile-time information: TYPE (only) */
template <typename T>
class tensor_t {
    
    private:
        std::shared_ptr<std::vector<T> > data;    // data stored in a consecutive area
        
        std::vector<size_t> width; // array containing the size of each dimension
        std::vector<size_t> stride; // array containing the stride of each dimension
    
        T* start_ptr;   // pointer to the first element of vector data (to optimize the access)
        
    public:    
        /* Constructors */
        tensor_t(){}

        template<size_t R>
        tensor_t(tensor<T,R> ten) {
            data = ten.data;
            width = std::vector<size_t>(ten.width.begin(), ten.width.end());
            stride = std::vector<size_t>(ten.stride.begin(), ten.stride.end());

            start_ptr = ten.start_ptr;
        }

        tensor_t(const std::vector<size_t>& dims, const std::vector<T>& data_param = std::vector<T>(0)) : width(dims) {
            assert(data_param.empty() || data_param.size() == get_dim(dims));

            if(!data_param.empty()) {
                data = std::make_shared< std::vector<T> >(data_param);
            } else {
                data = std::make_shared< std::vector<T> >(get_dim(dims));
            }

            set_strides(dims);
            start_ptr = &(data->operator[](0));
        }
        
        tensor_t(const tensor_t<T>& copy) = default;   // Default Copy-constructor

        tensor_t(const tensor_t<T>&& move): data(move.data),stride(move.stride),width(move.width),start_ptr(move.start_ptr){
           
        }
    
        /* Copying function */
        tensor_t<T> copy(){
            tensor_t<T> c(this);
            c.data=std::make_shared<std::vector<T>>(*data);
            c.start_ptr = &((c.data)->operator[](0));
            return c;
        }
    
        // Return the rank of the tensor
        size_t get_rank() const {
            return width.size();
        }
    
        /* Direct access */
        T& operator() (const size_t indexes[]) const{
            T* temp_ptr = start_ptr;

            for (int i=0; i!=width.size(); ++i) {
                assert(indexes[i] < width[i]);
                temp_ptr += stride[i] * indexes[i];
            }
            
            return *temp_ptr;
        }
    
        T& operator() (const std::vector<size_t>& indexes) const{
            if(indexes.size() != get_rank()){
                //
            }
            assert(indexes.size() == get_rank());
            return operator()(&indexes[0]);
        }

        /* Slicing */
        tensor_t<T> slice(size_t fixed_dimension, size_t begin_pointer){

            assert(fixed_dimension < get_rank() && begin_pointer < width[fixed_dimension]);

            std::vector<size_t> slicing_strides(get_rank() - 1);
            std::vector<size_t> slicing_width(get_rank() - 1);

            int j = 0;
            for(int i = 0; i < get_rank(); i++){
                if(i != fixed_dimension){
                    slicing_strides[j] = stride[i];
                    slicing_width[j] = width[i];

                    j++;
                }
            }

            T* new_start_pointer = stride[fixed_dimension] * begin_pointer + start_ptr;

            return tensor_t(slicing_width, new_start_pointer, slicing_strides,data);
        }

        tensor_t<T> slice(std::vector< std::tuple<size_t, size_t> > params){

            std::vector<size_t> slicing_strides(get_rank() - params.size());
            std::vector<size_t> slicing_width(get_rank() - params.size());

            int j = 0;
            for(int i = 0; i < get_rank(); i++){

                bool present = false;
                for(std::vector<std::tuple<size_t, size_t>>::iterator it = params.begin(); it != params.end(); ++it)
                    if(std::get<0>(*it) == i)
                        present = true;

                if(!present){
                    slicing_strides[j] = stride[i];
                    slicing_width[j] = width[i];

                    j++;
                }
            }

            T* new_start_ptr = start_ptr;
            for(std::vector<std::tuple<size_t, size_t>>::iterator it = params.begin(); it != params.end(); ++it)
                new_start_ptr += stride[std::get<0>(*it)] * std::get<1>(*it);

            return tensor_t(slicing_width, new_start_ptr, slicing_strides, data);
        }

        /* Windowing of more dimensions, params contains tuple with attributes (index, start, end)*/
        tensor_t<T> window(std::vector< std::tuple<size_t, size_t, size_t> > params){
            assert(!params.empty());

            for(auto iterator = params.begin(); iterator != params.end(); ++iterator)
                //if(start > end || end > width[index]) an exception is thrown
                assert(std::get<1>(*iterator) <= std::get<2>(*iterator) && std::get<2>(*iterator) <= width[ std::get<0>(*iterator) ]);

            tensor_t<T> res(*this);

            std::vector<size_t> start_ptr_indexes(width.size(), 0);

            for(auto iterator = params.begin(); iterator != params.end(); ++iterator){
                res.width[ std::get<0>(*iterator) ] = std::get<2>(*iterator) - std::get<1>(*iterator) + 1;

                start_ptr_indexes[ std::get<0>(*iterator) ] = std::get<1>(*iterator);
            }

            T* temp_ptr = start_ptr;

            for (int i=0; i != width.size(); ++i) {
                temp_ptr += stride[i] * start_ptr_indexes[i];
            }

            res.start_ptr = temp_ptr;
            return res;
        }

        /* Flattening */
        tensor_t<T> flatten(std::vector<size_t> dimensions){

            /*return a copy of the tensor*/
            if(dimensions.size() == 1 || dimensions.empty()){
                tensor_t<T> new_tensor = *this;
                return new_tensor;
            }

            for(std::vector<size_t>::iterator iterator = dimensions.begin(); iterator != dimensions.end(); ++iterator)
                assert(*iterator < get_rank());

            /*check for repetitions*/
            std::sort(dimensions.begin(), dimensions.end());

            std::vector<size_t>::iterator it = std::unique(dimensions.begin(), dimensions.end());
            bool has_repetitions = !(it == dimensions.end() );

            assert(!has_repetitions);

            std::vector<size_t> flattening_width(get_rank() - dimensions.size() + 1);

            size_t position_max = dimensions[0];
            size_t width_max = width[ dimensions[0] ];
            size_t new_dimension = 1;

            for(int i = 0; i < dimensions.size(); ++i){
                new_dimension *= width[ dimensions[i] ];

                if(width[ dimensions[i] ] > width_max){
                    width_max = width[ dimensions[i] ];
                    position_max = dimensions[i];
                }
            }

            int j = 0;
            for(int i = 0; i < get_rank(); i++){
                if( std::find(dimensions.begin(), dimensions.end(), i) != dimensions.end() ) {
                    /* dimension contains element i */
                    if(i == position_max){
                        flattening_width[j] = new_dimension;
                        j++;
                    }
                } else {
                    /* dimension does not contain element i */
                    flattening_width[j] = width[i];
                    j++;
                }
            }

            return tensor_t<T>(flattening_width, start_ptr,{},data);
        }
    
        /* Iterators */
        typedef iterator_namespace::iterator<T,std::vector<size_t>> iterator;
    
        iterator begin(){
            return iterator(width, stride, start_ptr);
        }
        iterator end(){
            iterator end_it = begin();
            end_it.it_ptr += stride[0]*width[0];
            for (int i=0; i<width.size(); ++i) {
                end_it.temp_vect[i]=width[i]-1;
            }
            return end_it;
        }
    
        typedef iterator_namespace::index_iterator<T> index_iterator;
    
        index_iterator begin(size_t ind,std::vector<size_t> fixed){
            T* start=start_ptr;
            for (int i=0;i<width.size();++i){
                if(i!=ind){
                    start+=stride[i]*fixed[i];
                }
            }
            return index_iterator(stride[ind], start);
        }
        index_iterator end(size_t ind,std::vector<size_t> fixed){
            index_iterator end_it = begin(ind,fixed);
            end_it.it_ptr+=stride[ind]*width[ind];
            return end_it;
        }
    
    private:
        size_t get_dim(std::vector<size_t> dims){
            if (dims.size() == 0){
                return 0;
            }
            size_t len = 1;
            for (unsigned int i : dims){
                len *= i;
            }            
            return len;
        }

        void set_strides(std::vector<size_t> dims){
            if (dims.size() == 0){
                return;
            }
            
            stride.insert(stride.begin(),1);

            unsigned s = 1;
            auto i = dims.size()-1;
            for (; i!=0; --i){
                s = dims[i]*s;
                stride.insert(stride.begin(),s);
            }
        }

        //private constructor uses during slicing     //tolto = std::vector<size_t>(0)
        tensor_t(const std::vector<size_t>& dims, T* start_ptr_param, const std::vector<size_t>& strides_param ,std::shared_ptr<std::vector<T>> new_data): data(new_data){
            if(strides_param.empty())
                set_strides(dims);
            else
                stride = std::vector<size_t>(strides_param);

            width = std::vector<size_t>(dims);
            start_ptr = start_ptr_param;
        }
        
        friend std::ostream& operator << <> (std::ostream&, tensor_t<T>&);
};

/* Overload operator << */
template<typename T>
std::ostream& operator << (std::ostream& out, tensor_t<T>& tensor){
    std::string string = "";

    switch (tensor.get_rank()) {
        case 1:
            for(size_t i = 0; i < tensor.width[0]; ++i) {
                string.append(std::to_string(tensor({i})));
                string.append(" ");
            }
            string.append("\n");
            break;
            
        case 2:
            for(size_t i = 0; i < tensor.width[0]; ++i){
                for(size_t j = 0; j < tensor.width[1]; ++j){
                    string.append(std::to_string(tensor({i, j})));
                    string.append(" ");
                }
                string.append("\n");
            }
            string.append("\n");
            break;
            
        case 3:
            for(size_t z = 0; z < tensor.width[2]; ++z) {
                for(size_t i = 0; i < tensor.width[0]; ++i) {
                    for(size_t j = 0; j < tensor.width[1]; ++j) {
                        string.append(std::to_string(tensor({i, j, z})));
                        string.append(" ");
                    }
                    string.append("\n");
                }
                string.append("\n \n");
            }
            break;
    }
    
    return out << string;
}

/* Class tensor for compile-time information: TYPE + RANK */
template <typename T, size_t R>
class tensor {
    
    private:
        std::shared_ptr<std::vector<T>> data;
        std::array<size_t,R> stride;
        std::array<size_t,R> width;
        T* start_ptr;

        size_t get_dim(const std::array<size_t,R>& dims){
            if (dims.size() == 0){
                return 0;
            }
            size_t len = 1;
            for (unsigned int i : dims){
                len *= i;
            }            
            return len;
        }

        void set_strides(const std::array<size_t,R>& dims){
            if (dims.empty()){
                return;
            }
            
            stride[R-1] = 1;

            unsigned s = 1;
            auto i = dims.size()-1;
            for (; i!=0; --i){
                s = dims[i]*s;
                stride[i-1]=s;
            }
        }
    
        tensor(const std::array<size_t,R>& new_width , T* new_ptr, std::shared_ptr<std::vector<T>> new_data, const std::array<size_t,R>& new_stride = {}) :data(new_data)/*, width(new_witdh), stride(new_stride)*/ {

            if(std::all_of(new_stride.begin(), new_stride.end(), [](int i){return i == 0;}))
                set_strides(new_width);
            else
                stride = std::array<size_t,R>(new_stride);

            width = std::array<size_t,R>(new_width);
            start_ptr = new_ptr;
        }


    public:
        /* Constructors */
        tensor(const std::array<size_t,R>& dims, const std::vector<T> data_param = std::vector<T>(0)): width(dims){

            if(data_param.empty())
                data = std::make_shared<std::vector<T>>(get_dim(dims));
            else
                data = std::make_shared<std::vector<T>>(data_param);

            set_strides(dims);
            start_ptr = &(data->operator[](0));
        }
       
        tensor(const tensor<T,R>& copy): data(copy.data),stride(copy.stride),width(copy.width),start_ptr(copy.start_ptr) {
        }

        tensor(tensor<T,R>&& move): data(move.data),stride(move.stride),width(move.width),start_ptr(move.start_ptr){
           
        }
    
        /* Copy function */
        tensor<T,R> copy(){
            tensor<T,R> c(this);
            c.data=std::make_shared<std::vector<T>>(*data);
            c.start_ptr = &((c.data)->operator[](0));
            return c;
        }
        
        /* Direct access */
        T& operator() (const size_t indexes[R]) const{
            T* temp_ptr = start_ptr;
            for (int i=0; i!=R ; ++i) {
                assert(indexes[i] < width[i]);
                temp_ptr += stride[i] * indexes[i];
            }
            return *temp_ptr;
        }

        T& operator() (const std::array<size_t, R>& indexes) const{
            assert(indexes.size() == R);
            return operator()(&indexes[0]);
        }

    
        /* Slicing */
        tensor<T,R-1> slice(size_t index,size_t where){

            std::array<size_t,R-1> slicing_strides;
            std::array<size_t,R-1> slicing_width;

            int j = 0;
            for(int i = 0; i < R; i++){
                if(i != index){
                    slicing_strides[j] = stride[i];
                    slicing_width[j] = width[i];
                    j++;
                }
            }

            T* new_start_pointer = stride[index] * where + start_ptr;

            return tensor<T,R-1>(slicing_width, new_start_pointer, data, slicing_strides);
        }

        /* Windowing of one dimension (index, start, end) */
        tensor<T,R> window(size_t index, size_t start, size_t end){
            tensor<T,R> res(*this);
            res.width[index]=end-start;
            res.start_ptr = start_ptr+(stride[index]*start);
            return res;
        }

        /* Windowing of more dimensions, params contains tuple with attributes (index, start, end) */
        tensor<T,R> window(std::vector< std::tuple<size_t, size_t, size_t> > params){
            assert(!params.empty());

            for(auto iterator = params.begin(); iterator != params.end(); ++iterator)
                //if(start > end || end > width[index]) an exception is thrown
                assert(std::get<1>(*iterator) <= std::get<2>(*iterator) && std::get<2>(*iterator) <= width[ std::get<0>(*iterator) ]);


            tensor<T,R> res(*this);

            std::array<size_t, R> start_ptr_indexes;
            start_ptr_indexes.fill(0);

            for(auto iterator = params.begin(); iterator != params.end(); ++iterator){
                res.width[ std::get<0>(*iterator) ] = std::get<2>(*iterator) - std::get<1>(*iterator) + 1;

                start_ptr_indexes[ std::get<0>(*iterator) ] = std::get<1>(*iterator);
            }

            T* temp_ptr = start_ptr;

            for (int i = 0; i != R; ++i) {
                temp_ptr += stride[i] * start_ptr_indexes[i];
            }

            res.start_ptr = temp_ptr;
            return res;
        }
    
        /* Flattening */
        tensor<T,R-1> flatten(size_t first_dimension,size_t second_dimension){

            std::array<size_t,R-1> flattening_width;
            size_t position_max = (width[first_dimension] > width[second_dimension]) ? first_dimension : second_dimension;

            int j = 0;
            for(int i = 0; i < R; i++) {
                if (i != first_dimension && i != second_dimension) {
                    flattening_width[j] = width[i];
                    j++;
                } else {
                    if (i == position_max) {
                        flattening_width[j] = width[first_dimension] * width[second_dimension];
                        j++;
                    }
                }
            }

            return tensor<T,R-1>(flattening_width,start_ptr,data);
        }

        /* Iterators */
        typedef iterator_namespace::iterator<T,std::array<size_t,R>,R> iterator;

        iterator begin(){
            
            return iterator(width, stride, start_ptr);
        }
        iterator end(){
            iterator end_it = begin();
            end_it.it_ptr += stride[0]*width[0];
            for (int i=0; i<width.size(); ++i) {
                end_it.temp_vect[i]=width[i]-1;
            }
            return end_it;
        }

        typedef iterator_namespace::index_iterator<T,R> index_iterator;
    
        index_iterator begin(size_t ind,std::array<size_t,R-1> fixed){
            T* start=start_ptr;
            for (int i=0;i<width.size();++i){
                if(i!=ind){
                    start+=stride[i]*fixed[i];
                }
            }
            return index_iterator(stride[ind], start);
        }
        index_iterator end(size_t ind,std::array<size_t,R-1> fixed){
            index_iterator end_it = begin(ind,fixed);
            end_it.it_ptr+=stride[ind]*width[ind];
            return end_it;
        }
    
        friend class tensor<T,R+1>;
        friend class tensor_t<T>;
    
        friend std::ostream& operator << <>
        (std::ostream&, tensor<T,R>&);
};

/* Overload operator << */
template<typename T, size_t R>
std::ostream& operator << (std::ostream& out, tensor<T,R>& tensor){
    std::string string = "";

    switch (R) {
        case 1:
            for(size_t i = 0; i < tensor.width[0]; ++i) {
                size_t indexes[] = {i};
                string.append(std::to_string(tensor(indexes)));
                string.append(" ");
            }
            string.append("\n");
            break;

        case 2:
            for(size_t i = 0; i < tensor.width[0]; ++i){
                for(size_t j = 0; j < tensor.width[1]; ++j){
                    size_t indexes[] = {i, j};
                    string.append(std::to_string(tensor(indexes)));
                    string.append(" ");
                }
                string.append("\n");
            }
            string.append("\n");
            break;

        case 3:
            for(size_t z = 0; z < tensor.width[2]; ++z) {
                for(size_t i = 0; i < tensor.width[0]; ++i) {
                    for(size_t j = 0; j < tensor.width[1]; ++j) {
                        size_t indexes[] = {i, j, z};
                        string.append(std::to_string(tensor(indexes)));
                        string.append(" ");
                    }
                    string.append("\n");
                }
                string.append("\n \n");
            }
            break;
    }

    return out << string;
}

// Specialization for tensor with Rank 1
template <typename T>
class tensor<T,1> {
    
     private:
        std::shared_ptr<std::vector<T>> data;
        T* start_ptr;
        size_t width;
    
        tensor(const size_t new_witdh , T* new_ptr, std::shared_ptr<std::vector<T>> new_data) :width(new_witdh), data(new_data), start_ptr(new_ptr) {
        }

    public:
        /* Constructors */
        tensor(const tensor<T,1>& copy) : data(copy.data),width(copy.width),start_ptr(copy.start_ptr) {
        }

        tensor(tensor<T,1>&& move) : data(move.data),width(move.width),start_ptr(move.start_ptr){
            move.data=nullptr;
            move.start_ptr=nullptr;
        }
    
        tensor(const size_t& dim, const std::vector<T>& data_param = std::vector<T>(0)) : width(dim) {

            if(data_param.empty())
                data = std::make_shared<std::vector<T>>(dim);
            else
                data = std::make_shared<std::vector<T>>(data_param);

            start_ptr = &((*data)[0]);
        }

        /* Copying function */
        tensor<T,1> copy(){
            tensor<T,1> c(this);
            c.data=std::make_shared<std::vector<T>>(*data);
            c.start_ptr = &((c.data)->operator[](0));
            return c;
        }
    
        /* Direct access */
        T& operator() (const size_t index) const{
            T* temp_ptr = start_ptr;
            temp_ptr += index;
            return *temp_ptr;
        }

        /* Slicing */
        T& slice(const size_t index){
            return operator()(index);
        }
    
        /* Windowing */
        tensor<T,1> window(size_t begin, size_t end){
            return tensor<T,1>(end - begin + 1, start_ptr + begin, data);
        }
    
        /* Iterators */
        typedef iterator_namespace::index_iterator<T,1> iterator;

        iterator begin(){
            return iterator(1, start_ptr);
        }

        iterator end() {
            return iterator(1, start_ptr + width);
        }
    
        friend class tensor<T,2>;
        friend std::ostream& operator << <>(std::ostream&, tensor<T,1>&);
};

/* Overload operator << */
template<typename T>
std::ostream& operator << (std::ostream& out, tensor<T,1>& t){
    std::string string = "";

    for(size_t i = 0; i < t.width; ++i) {
         string.append(std::to_string(t(i)));
         string.append(" ");
    }

    string.append("\n");

    return out << string;
}

