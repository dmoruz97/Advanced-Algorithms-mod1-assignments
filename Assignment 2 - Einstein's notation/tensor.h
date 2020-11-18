#ifndef TENSOR
#define TENSOR

#include<array>
#include<cstdlib>
#include<cassert>
#include<utility>
#include<memory>
#include<type_traits>
#include<initializer_list>
#include<string>
#include<vector>
#include<algorithm>

#include <set>

// Struct my_set: simple template containing a pack of values
template<int... Ns>
struct my_set;

template<int... ints>
struct my_set {
};

// Struct set_find: template to know if an element is contained in a my_set
template<int T, class S>
struct set_find;

template<int T>
struct set_find<T, my_set<> > {   // specialization for empty set
    static constexpr size_t value = 0;
};

template<int T, int ... tail>
struct set_find<T, my_set<T, tail...> > {
    static constexpr size_t value = 1;
};

template<int T, int H, int... tail>
struct set_find<T, my_set<H, tail...> > {
    static constexpr size_t value = set_find<T, my_set<tail...> >::value;
};

// Class num_same_indexes: template that returns the number of same indexes in T not already present in my_set A
template<class A, int... T>
class num_same_indexes;

template<>
struct num_same_indexes<my_set<>> {        // specialization for empty set
    static constexpr size_t value = 0;
};

template<int T, int ... S>
struct num_same_indexes<my_set<S...>, T> {     // specialization for one value
    static constexpr size_t value = 0;
};

template<int T, int ... tail, int ... S>
struct num_same_indexes<my_set<S...>, T, tail...> {
    static constexpr size_t value = set_find<T, my_set<tail...>>::value * (2 - set_find<T,my_set<S...>>::value) + set_find<T, my_set<tail...>>::value * num_same_indexes<my_set<S..., T>, tail...>::value + (1 - set_find<T, my_set<tail...>>::value) * num_same_indexes<my_set<S...>, tail...>::value;
};

// Class num_matched_indexes: template that returns the number of same indexes between the two sets F and S
template<class F, class S>
struct num_matched_indexes;

template<int F, int S>
struct num_matched_indexes<my_set<F>, my_set<S>> {
    static constexpr size_t value = 0;
};

template<int F>
struct num_matched_indexes<my_set<F>, my_set<F>> {
    static constexpr size_t value = 1;
};

template<int F, int ... S>
struct num_matched_indexes<my_set<F>, my_set<S...>> {
    static constexpr size_t value = set_find<F, my_set<S...>>::value;
};

template<int H, int ... F, int ... S>
struct num_matched_indexes<my_set<H, F...>, my_set<S...>> {
    static constexpr size_t value =
            set_find<H, my_set<S...>>::value + num_matched_indexes<my_set<F...>, my_set<S...>>::value;
};

// Struct num_same_grouped_indexes: template that returns the number of groups formed by identical indexes
template<class A, int ... T>
struct num_same_grouped_indexes;

template<int T, int ... S>
struct num_same_grouped_indexes<my_set<S...>, T> {
    static constexpr size_t value = 0;
};

template<int H, int ... T, int ... S>
struct num_same_grouped_indexes<my_set<S...>, H, T...> {
    static constexpr size_t value = set_find<H, my_set<T...>>::value * (1 - set_find<H, my_set<S...>>::value) + (1 - set_find<H, my_set<S...>>::value) * num_same_grouped_indexes<my_set<H, S...>, T...>::value + set_find<H, my_set<S...>>::value * num_same_grouped_indexes<my_set<S...>, T...>::value;
};


using namespace std::rel_ops;

namespace tensor {

    // policy for dynamically ranked tensors
    struct dynamic {
        typedef std::vector<size_t> index_type;
        typedef std::vector<size_t> width_type;

        // Vector of indexes to use Eintein's notation
        typedef std::vector<size_t> indexes_notation_type;
    };

    // policy for fixed-rank tensors
    template<size_t R>
    struct rank {
        typedef std::array<size_t, R> index_type;
        typedef std::array<size_t, R> width_type;

        // Vector of indexes to use Eintein's notation
        typedef std::array<size_t, R> indexes_notation_type;
    };


    // Namespace for the Einstein's notation indexes
    namespace index {

        // INDEXES FOR DYNAMIC RANK TENSOR
        class index_t {
        public:
            //index_t() { value = 0; }
            index_t(size_t x) : value(x) {}

            size_t getValue() const {
                return value;
            }

            bool operator==(const index_t &ind) const {
                return value == ind.value;
            }

        private:
            const size_t value;
        };

        // Definition of some indexes that could be used
        const index_t i(0);
        const index_t j(1);
        const index_t k(2);
        const index_t l(3);
        const index_t m(4);
        const index_t n(5);
        const index_t o(6);
        const index_t p(7);
        const index_t q(8);
        const index_t r(9);


        // INDEXES FOR FIXED RANK TENSOR
        template<int n>
        struct index_tr {
            static constexpr int value = n;
        };
        // Definition of some indexes that could be used
        index_tr<0> i_r;
        index_tr<1> j_r;
        index_tr<2> k_r;
    }


    // tensor type
    template<typename T, class type=dynamic>
    class tensor;

    // proxy tensor
    template<typename T, class type, class ... S>
    class proxy_tensor;

    namespace reserved {
        // generic iterator used by all tensor classes (except rank 1 specializations)
        template<typename T, class type>
        class iterator {
        public:
            T &operator*() const { return *ptr; }

            iterator &operator++() {
                // I am using a right-major layout
                //start increasing the last index
                size_t index = stride.size() - 1;
                ++idx[index];
                ptr += stride[index];
                // as long as the current index has reached maximum width,
                // set it to 0 and increase the next index
                while (idx[index] == width[index] && index > 0) {
                    idx[index] = 0;
                    ptr -= width[index] * stride[index];
                    --index;
                    ++idx[index];
                    ptr += stride[index];
                }
                return *this;
            }

            iterator operator++(int) {
                iterator result(*this);
                operator++();
                return result;
            }

            iterator &operator--() {
                // I am using a right-major layout
                //start increasing the last index
                size_t index = stride.size() - 1;
                // as long as the current index has reached 0,
                // set it to width-1 and decrease the next index
                while (idx[index] == 0 && index > 0) {
                    idx[index] = width[index] - 1;
                    ptr + idx[index] * stride[index];
                    --index;
                }
                --idx[index];
                ptr -= stride[index];
                return *this;
            }

            iterator operator--(int) {
                iterator result(*this);
                operator--();
                return result;
            }

            iterator &operator-=(int v) {
                if (v < 0) return operator+=(-v);
                size_t index = stride.size() - 1;
                while (v > 0 && index >= 0) {
                    size_t val = v % width[index];
                    v /= width[index];
                    if (val <= idx[index]) {
                        idx[index] -= val;
                        ptr -= val * stride[index];
                    } else {
                        --v;
                        idx[index] += width[index] - val;
                        ptr += (width[index] - val) * stride[index];
                    }
                    --index;
                }
                return *this;
            }

            iterator &operator+=(int v) {
                if (v < 0) return operator-=(-v);
                size_t index = stride.size() - 1;
                while (v > 0 && index >= 0) {
                    size_t val = v % width[index];
                    v /= width[index];
                    idx[index] += val;
                    ptr += val * stride[index];
                    if (idx[index] >= width[index] && index > 0) {
                        idx[index] -= width[index];
                        ++idx[index - 1];
                        ptr += stride[index - 1] - width[index] * stride[index];
                    }
                    --index;
                }
                return *this;
            }

            iterator operator+(int v) const {
                iterator result(*this);
                result += v;
                return result;
            }

            iterator operator-(int v) const {
                iterator result(*this);
                result -= v;
                return result;
            }

            T &operator[](int v) const {
                iterator iter(*this);
                iter += v;
                return *iter;
            }

            typename type::index_type get_indexes() {
                return idx;
            }

            // defines equality as external friend function
            // inequality gest automatically defined by std::rel_ops
            friend bool operator==(const iterator &i, const iterator &j) { return i.ptr == j.ptr; }

            friend class tensor<T, type>;

            template<typename, class, class ...>
            friend class proxy_tensor;

        private:
            iterator(const typename type::width_type &w, const typename type::index_type &s, T *p) : width(w), stride(s), idx(s), ptr(p) {
                std::fill(idx.begin(), idx.end(), 0);
            }

            // maintain references to width and strides
            // uses policy for acual types
            const typename type::width_type &width;
            const typename type::index_type &stride;

            // maintains both indices and pointer to data
            // uses pointer to data for dereference and equality for efficiency
            typename type::index_type idx;
            T *ptr;
        };


        // iterator over single index
        // does not need to know actual tensor type
        template<typename T>
        class index_iterator {
        public:
            T &operator*() const { return *ptr; }

            index_iterator &operator++() {
                ptr += stride;
                return *this;
            }

            index_iterator operator++(int) {
                index_iterator result(*this);
                operator++();
                return result;
            }

            index_iterator &operator--() {
                ptr -= stride;
                return *this;
            }

            index_iterator operator--(int) {
                index_iterator result(*this);
                operator--();
                return result;
            }

            index_iterator &operator-=(int v) {
                ptr -= v * stride;
                return *this;
            }

            index_iterator &operator+=(int v) {
                ptr + -v * stride;
                return *this;
            }

            index_iterator operator+(int v) const {
                index_iterator result(*this);
                result += v;
                return result;
            }

            index_iterator operator-(int v) const {
                index_iterator result(*this);
                result -= v;
                return result;
            }

            T &operator[](int v) const { return *(ptr + v * stride); }


            friend bool operator==(const index_iterator &i, const index_iterator &j) { return i.ptr == j.ptr; }

            template<typename, typename>
            friend class ::tensor::tensor;

        private:
            index_iterator(size_t s, T *p) : stride(s), ptr(p) {}

            size_t stride;
            T *ptr;
        };
    }


    // tensor specialization for dynamic rank
    template<typename T>
    class tensor<T, dynamic> {
    public:
        // C-style constructor with explicit rank and pointer to array of dimensions
        // all other constructors are redirected to this one
        tensor(size_t rank, const size_t dimensions[]) : width(dimensions, dimensions + rank), stride(rank, 1UL) {
            for (size_t i = width.size() - 1UL; i != 0; --i) {
                stride[i - 1] = stride[i] * width[i];
            }
            data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
            start_ptr = &(data->operator[](0));
        }

        tensor(const std::vector<size_t> &dimensions) : tensor(dimensions.size(), &dimensions[0]) {}

        tensor(std::initializer_list<size_t> dimensions) : tensor(dimensions.size(), &*dimensions.begin()) {}

        template<size_t rank>
        tensor(const size_t dims[rank]) : tensor(rank, dims) {}

        template<typename...Dims>
        tensor(Dims...dims) : width({static_cast<const size_t>(dims)...}), stride(sizeof...(dims), 1UL) {
            for (size_t i = width.size() - 1UL; i != 0UL; --i) {
                stride[i - 1] = stride[i] * width[i];
            }
            data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
            start_ptr = &(data->operator[](0));
        }

        tensor(const tensor<T, dynamic> &X) = default;

        tensor(tensor<T, dynamic> &&X) = default;

        tensor<T, dynamic> &operator=(const tensor<T, dynamic> &X) = default;

        tensor<T, dynamic> &operator=(tensor<T, dynamic> &&X) = default;

        // all tensor types are friend
        // this are used by alien copy constructors, i.e. copy constructors copying different tensor types.
        template<typename, typename> friend
        class tensor;

        template<size_t R>
        tensor(const tensor<T, rank < R>>&X) : data(X.data),width(X.width.begin(), X.width.end()),stride(X.stride.begin(), X.stride.end()),start_ptr(X.start_ptr) {}

/*---------------------------------------------------------------------------------------------------------*/
/*--------- EINSTEIN's NOTATION ---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/

        // Function which returns the Einstein's notation indexes
        std::vector<size_t> get_indexes_notation() {
            return indexesNotation;
        }

        explicit operator T() const {
            assert(get_rank() == 1 && width[0] == 1);
            return *start_ptr;
        }

        // Overload operator ()
        tensor<T, dynamic> operator()(const std::vector<index::index_t> &indexes) {
            assert(indexes.size() <= width.size());

            //vector of vectors containing the positions of same indexes. if a_{ijikjk} same_indexes will be { {0,2}, {1,4}, {3,5} }
            std::vector<typename dynamic::index_type> same_grouped_indexes = get_same_grouped_indexes(indexes);

            //the first vector contains the same indexes, while the second contains the corresponding dimensions
            std::pair<typename dynamic::index_type, typename dynamic::index_type> all_same_indexes_and_dimensions = get_all_same_indexes_and_dimensions(
                    same_grouped_indexes);

            //the first vector contains the different indexes, while the second contains the corresponding dimensions
            std::pair<typename dynamic::index_type, typename dynamic::index_type> all_different_indexes_and_dimensions = get_all_different_indexes_and_dimensions(
                    all_same_indexes_and_dimensions.first);

            if (same_grouped_indexes.empty()) {
                indexesNotation = {};
                for (index::index_t x : indexes) {
                    indexesNotation.push_back(x.getValue());
                }
                return *this;
            }
            else {
                typename dynamic::index_type indexes_to_initialize;

                if (all_different_indexes_and_dimensions.second.empty()) {
                    indexes_to_initialize = {1};
                }
                else {
                    indexes_to_initialize = all_different_indexes_and_dimensions.second;
                }

                tensor<T, dynamic> tensor_out(indexes_to_initialize);

                for (tensor::iterator iter = begin(); iter != end(); ++iter) {
                    for (typename dynamic::index_type x : same_grouped_indexes) {
                        if (indexes_are_equal(all_same_indexes_and_dimensions.first, iter.get_indexes())) {

                            std::vector<size_t> vector;

                            if (all_different_indexes_and_dimensions.first.empty()){
                                vector = {0};
                            }
                            else {
                                vector = get_different_actual_position(all_different_indexes_and_dimensions.first, iter.get_indexes());
                            }

                            tensor_out(vector) += *iter;
                        }
                    }
                }

                for (int i = 0; i < all_different_indexes_and_dimensions.first.size(); ++i) {
                    tensor_out.indexesNotation.push_back(indexes.at(all_different_indexes_and_dimensions.first[i]).getValue());
                }

                return tensor_out;
            }
        }

        // Overload operator *
        tensor<T, dynamic> operator*(tensor<T, dynamic> t) {

            typename dynamic::index_type t_IndexesNotation = t.get_indexes_notation();

            //vector of pair: accessed_index = the first value correspond to the position of the same indexes, the second value contains the corresponding dimension
            std::vector<std::pair<size_t, size_t>> same_this;
            std::vector<std::pair<size_t, size_t>> same_other;

            std::pair<std::vector<std::pair<size_t, size_t>>, std::vector<std::pair<size_t, size_t>>> same_vectors = get_same_vectors(*this, indexesNotation, t, t_IndexesNotation);
            same_this = same_vectors.first;
            same_other = same_vectors.second;

            assert(!same_this.empty());

            //vector of pair: the first value correspond to the position of the different indexes, the second value is the corresponding dimension
            std::vector<std::pair<size_t, size_t>> different_this;
            std::vector<std::pair<size_t, size_t>> different_other;

            std::pair<std::vector<std::pair<size_t, size_t>>, std::vector<std::pair<size_t, size_t>>> different_vectors = get_different_vectors(
                    *this, indexesNotation, t, t_IndexesNotation);
            different_this = different_vectors.first;
            different_other = different_vectors.second;

            typename dynamic::index_type indexes_to_initilize_tensor;

            if (different_this.empty()) {
                indexes_to_initilize_tensor = {1};
            }
            else {
                indexes_to_initilize_tensor = get_merged_different_dimensions(different_this, different_other);
            }

            // Tensor to be returned, initilized with its correct dimensions
            tensor<T, dynamic> out_tensor(indexes_to_initilize_tensor);


            // "out_tensor" initialized to 0
            for (tensor::iterator iter = out_tensor.begin(); iter != out_tensor.end(); ++iter) {
                *iter = 0;
            }

            // For each element in the tensor to be returned
            for (tensor::iterator this_iter = begin(); this_iter != end(); ++this_iter) {
                for (tensor::iterator other_iter = t.begin(); other_iter != t.end(); ++other_iter) {


                    typename dynamic::index_type this_actual_position = this_iter.get_indexes();
                    typename dynamic::index_type other_actual_position = other_iter.get_indexes();

                    for (std::pair<size_t, T> x : same_this) {
                        for (std::pair<size_t, T> y : same_other) {
                            if (this_actual_position[x.first] == other_actual_position[y.first]) {
                                std::vector<size_t> accessed_index = get_different_positions(this_actual_position,other_actual_position,different_this,different_other, t);

                                if (accessed_index.size() != out_tensor.get_rank()) {
                                    accessed_index = {0};
                                }

                                out_tensor(accessed_index) += (*this_iter) * (*other_iter);
                            }
                        }
                    }
                }
            }

            out_tensor.indexesNotation = get_merged_different_indexes_notation(indexesNotation, t_IndexesNotation);

            return out_tensor;
        }

        // Overload operator +
        tensor<T, dynamic> operator+(tensor<T, dynamic> t) {
            return operators_plus_minus_temp(t, '+');
        }

        // Overload operator -
        tensor<T, dynamic> operator-(tensor<T, dynamic> t) {
            return operators_plus_minus_temp(t, '-');
        }
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/

        // rank accessor
        size_t get_rank() const { return width.size(); }

        // direct accessors. Similarly to std::vector, operator () does not perform range check
        // while at() does
        T &operator()(const size_t dimensions[]) const {
            const size_t rank = width.size();
            T *ptr = start_ptr;
            for (int i = 0; i != rank; ++i) ptr += dimensions[i] * stride[i];
            return *ptr;
        }

        T &at(const size_t dimensions[]) const {
            const size_t rank = width.size();
            T *ptr = start_ptr;
            for (int i = 0; i != rank; ++i) {
                assert(dimensions[i] < width[i]);
                ptr += dimensions[i] * stride[i];
            }
            return *ptr;
        }

        T &operator()(const std::vector<size_t> &dimensions) const {
            assert(dimensions.size() == get_rank());
            return operator()(&dimensions[0]);
        }

        T &at(const std::vector<size_t> &dimensions) const {
            assert(dimensions.size() == get_rank());
            return at(&dimensions[0]);
        }

        template<size_t rank>
        T &operator()(const size_t dimensions[rank]) const {
            assert(rank == get_rank());
            return operator()(static_cast<const size_t *>(dimensions));
        }

        template<size_t rank>
        T &at(const size_t dimensions[rank]) const {
            assert(rank == get_rank());
            return at(static_cast<const size_t *>(dimensions));
        }

        template<typename...Dims>
        T &operator()(Dims...dimensions) const {
            assert(sizeof...(dimensions) == get_rank());
            return operator()({static_cast<const size_t>(dimensions)...});
        }

        template<typename...Dims>
        T &at(Dims...dimensions) const {
            assert(sizeof...(dimensions) == get_rank());
            return at({static_cast<const size_t>(dimensions)...});
        }


        // slice operation create a new tensor type sharing the data and removing the sliced index
        tensor<T, dynamic> slice(size_t index, size_t i) const {
            const size_t rank = width.size();
            assert(index < rank);
            tensor<T, dynamic> result;
            result.data = data;
            result.width.insert(result.width.end(), width.begin(), width.begin() + index);
            result.width.insert(result.width.end(), width.begin() + index + 1, width.end());
            result.stride.insert(result.stride.end(), stride.begin(), stride.begin() + index);
            result.stride.insert(result.stride.end(), stride.begin() + index + 1, stride.end());
            result.start_ptr = start_ptr + i * stride[index];

            return result;
        }

        // operator [] slices the first (leftmost) index
        tensor<T, dynamic> operator[](size_t i) const { return slice(0, i); }

        // window operation on a single index
        tensor<T, dynamic> window(size_t index, size_t begin, size_t end) const {
            tensor<T, dynamic> result(*this);
            result.width[index] = end - begin;
            result.start_ptr += result.stride[index] * begin;
            return result;
        }

        //window operations on all indices
        tensor<T, dynamic> window(const size_t begin[], const size_t end[]) const {
            tensor<T, dynamic> result(*this);
            const size_t r = get_rank();
            for (int i = 0; i != r; ++i) {
                result.width[i] = end[i] - begin[i];
                result.start_ptr += result.stride[i] * begin[i];
            }
            return result;
        }

        tensor<T, dynamic> window(const std::vector<size_t> &begin, const std::vector<size_t> &end) const {
            return window(&(begin[0]), &(end[0]));
        }

        // flaten operation
        // do not use over windowed and sliced ranges
        tensor<T, dynamic> flatten(size_t begin, size_t end) const {
            tensor<T, dynamic> result;
            result.stride.insert(result.stride.end(), stride.begin(), stride.begin() + begin);
            result.stride.insert(result.stride.end(), stride.begin() + end, stride.end());
            result.width.insert(result.width.end(), width.begin(), width.begin() + begin);
            result.width.insert(result.width.end(), width.begin() + end, width.end());
            for (int i = begin; i != end; ++i) result.width[end] *= width[i];
            result.start_prt = start_ptr;
            result.data = data;
            return result;
        }

        // specialized iterator type
        typedef reserved::iterator<T, dynamic> iterator;

        iterator begin() const { return iterator(width, stride, start_ptr); }

        iterator end() const {
            iterator result = begin();
            result.idx[0] = width[0];
            result.ptr += width[0] * stride[0];
            return result;
        }

        // specialized index_iterator type
        typedef reserved::index_iterator<T> index_iterator;

        // begin and end methods producing index_iterator require the index to be iterated over
        // and all the values for the other indices
        index_iterator begin(size_t index, const size_t dimensions[]) const {
            return index_iterator(stride[index], &operator()(dimensions) - dimensions[index] * stride[index]);
        }

        index_iterator end(size_t index, const size_t dimensions[]) const {
            return index_iterator(stride[index],
                                  &operator()(dimensions) + (width[index] - dimensions[index]) * stride[index]);
        }

        template<size_t rank>
        index_iterator begin(size_t index, const size_t dimensions[rank]) const {
            return index_iterator(stride[index], &operator()(dimensions) - dimensions[index] * stride[index]);
        }

        template<size_t rank>
        index_iterator end(size_t index, const size_t dimensions[rank]) const {
            return index_iterator(stride[index],
                                  &operator()(dimensions) + (width[index] - dimensions[index]) * stride[index]);
        }

        index_iterator begin(size_t index, const std::vector<size_t> &dimensions) const {
            return index_iterator(stride[index], &operator()(dimensions) - dimensions[index] * stride[index]);
        }

        index_iterator end(size_t index, const std::vector<size_t> &dimensions) const {
            return index_iterator(stride[index],
                                  &operator()(dimensions) + (width[index] - dimensions[index]) * stride[index]);
        }

    private:
        tensor() = default;

        std::shared_ptr<std::vector<T>> data;
        dynamic::width_type width;
        dynamic::index_type stride;
        T *start_ptr;

/*---------------------------------------------------------------------------------------------------------*/
/*--------- EINSTEIN's NOTATION ---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
        // Vector of indexes to use Eintein's notation
        dynamic::indexes_notation_type indexesNotation;

        tensor<T, dynamic> operators_plus_minus_temp(tensor<T, dynamic> t, const char sign) {

            assert(get_rank() == t.get_rank());

            typename dynamic::indexes_notation_type t_IndexesNotation = t.get_indexes_notation();

            //vector contains pair corresponding to the paired indexes of the tensors to sum
            std::vector<std::pair<size_t, size_t>> pair_indexes = get_paired_indexes(t, indexesNotation, t_IndexesNotation);

            typename dynamic::index_type indexes_to_initilize_tensor;

            tensor<T, dynamic> out_tensor(width);

            out_tensor.indexesNotation = indexesNotation;

            for (tensor::iterator this_iter = begin(); this_iter != end(); ++this_iter) {
                for (tensor::iterator other_iter = t.begin(); other_iter != t.end(); ++other_iter) {

                    typename dynamic::index_type this_actual_position = this_iter.get_indexes();
                    typename dynamic::index_type other_actual_position = other_iter.get_indexes();

                    bool flag = true;
                    for (std::pair<size_t, T> x : pair_indexes) {
                        if (this_actual_position[x.first] != other_actual_position[x.second]) {
                            flag = false;
                        }
                    }

                    if (flag) {
                        if (sign == '+') {
                            out_tensor(this_actual_position) = (*this_iter) + (*other_iter);
                        }
                        else if (sign == '-') {
                            out_tensor(this_actual_position) = (*this_iter) - (*other_iter);
                        }
                    }
                }
            }

            return out_tensor;
        }

        std::vector<size_t> get_merged_different_indexes_notation(typename dynamic::index_type& f,typename dynamic::index_type& s){

            std::vector<size_t> result;

            for(int j = 0; j < f.size(); j++){
                if(std::find(std::begin(s), std::end(s), f[j]) == std::end(s)){
                    result.push_back(f[j]);
                }
            }
            for(int j = 0; j < s.size(); j++){
                if(std::find(std::begin(f), std::end(f), s[j]) == std::end(f)){
                    result.push_back(s[j]);
                }
            }

            return result;
        }

        std::vector<std::pair<size_t, size_t>>
        get_same_pairs(tensor<T, dynamic> &first_tensor, const typename dynamic::index_type &first_indexes_notation, const typename dynamic::index_type &second_indexes_notation) {
            std::vector<std::pair<size_t, size_t>> result;

            for (int i = 0; i < first_indexes_notation.size(); ++i) {
                for (int j = 0; j < second_indexes_notation.size(); j++) {
                    if (first_indexes_notation[i] == second_indexes_notation[j]) {
                        std::pair<size_t, T> pair(i, first_tensor.width[i]);
                        result.push_back(pair);
                    }
                }
            }
            return result;
        }

        std::vector<std::pair<size_t, size_t>> get_paired_indexes(tensor<T, dynamic> &second_tensor,const typename dynamic::index_type &first_indexes_notation,const typename dynamic::index_type &second_indexes_notation) {

            std::vector<std::pair<size_t, size_t>> result;

            for (size_t i = 0; i < first_indexes_notation.size(); i++) {

                auto iter = std::find(second_indexes_notation.begin(), second_indexes_notation.end(),
                                      first_indexes_notation[i]);
                assert(iter != second_indexes_notation.end());

                std::pair<size_t, size_t> pair(i, std::distance(second_indexes_notation.begin(), iter));
                result.push_back(pair);
            }

            return result;
        }

        std::pair<std::vector<std::pair<size_t, size_t>>, std::vector<std::pair<size_t, size_t>>>
        get_same_vectors(tensor<T, dynamic> &first_tensor, const typename dynamic::index_type &first_indexes_notation,tensor<T, dynamic> &second_tensor,const typename dynamic::index_type &second_indexes_notation) {

            std::vector<std::pair<size_t, size_t>> first = get_same_pairs(first_tensor, first_indexes_notation,
                                                                          second_indexes_notation);
            std::vector<std::pair<size_t, size_t>> second = get_same_pairs(second_tensor, second_indexes_notation,
                                                                           first_indexes_notation);

            std::pair<std::vector<std::pair<size_t, size_t>>, std::vector<std::pair<size_t, size_t>>> result(first,
                                                                                                             second);
            return result;
        };

        std::vector<std::pair<size_t, size_t>> get_different_pairs(tensor<T, dynamic> &first_tensor,const typename dynamic::index_type &first_indexes_notation,const typename dynamic::index_type &second_indexes_notation){

            std::vector<std::pair<size_t, size_t>> result;
            std::vector<std::pair<size_t, size_t>> same_first = get_same_pairs(first_tensor, first_indexes_notation,
                                                                               second_indexes_notation);

            for (int i = 0; i < first_indexes_notation.size(); ++i) {
                std::pair<size_t, size_t> pair(i, first_tensor.width[i]);

                //same_first does not contain the pair
                if (std::find(same_first.begin(), same_first.end(), pair) == same_first.end()) {
                    result.push_back(pair);
                }
            }
            return result;
        }

        std::pair<std::vector<std::pair<size_t, size_t>>, std::vector<std::pair<size_t, size_t>>>
        get_different_vectors(tensor<T, dynamic> &first_tensor,const typename dynamic::index_type &first_indexes_notation,tensor<T, dynamic> &second_tensor,const typename dynamic::index_type &second_indexes_notation) {

            std::vector<std::pair<size_t, size_t>> first = get_different_pairs(first_tensor, first_indexes_notation,
                                                                               second_indexes_notation);
            std::vector<std::pair<size_t, size_t>> second = get_different_pairs(second_tensor, second_indexes_notation,
                                                                                first_indexes_notation);

            std::pair<std::vector<std::pair<size_t, size_t>>, std::vector<std::pair<size_t, size_t>>> result(first,
                                                                                                             second);
            return result;
        }

        typename dynamic::index_type
        get_merged_different_dimensions(const std::vector<std::pair<size_t, size_t>> &different_this, const std::vector<std::pair<size_t, size_t>> &different_other) {

            std::vector<size_t> result;

            result.reserve(different_this.size() + different_other.size());

            for (std::pair<size_t, size_t> pair : different_this) {
                result.push_back(pair.second);
            }

            for (std::pair<size_t, size_t> pair : different_other) {
                result.push_back(pair.second);
            }

            return result;
        }

        std::vector<std::vector<size_t>> get_same_grouped_indexes(const std::vector<index::index_t> &indexes) {
            std::vector<typename dynamic::index_type> result;
            std::vector<index::index_t> copy_indexes = indexes;
            typename dynamic::index_type selected_indexes;

            for (size_t i = 0; i < copy_indexes.size(); ++i) {
                std::vector<size_t> local_duplicates = {i};
                for (size_t j = 0; j < copy_indexes.size(); ++j) {
                    if (i < j &&
                        (std::find(selected_indexes.begin(), selected_indexes.end(), i) == selected_indexes.end()) &&
                        copy_indexes[i] == copy_indexes[j]) {
                        local_duplicates.push_back(j);

                        selected_indexes.push_back(i);
                    }
                }

                if (local_duplicates.size() > 1) {
                    result.push_back(local_duplicates);
                }
            }

            return result;
        }

        std::pair<std::vector<size_t>, std::vector<size_t> >
        get_all_same_indexes_and_dimensions(const std::vector<typename dynamic::index_type> &grouped_indexes) {
            typename dynamic::index_type first;
            typename dynamic::index_type second;

            for (std::vector<size_t> x : grouped_indexes) {
                for (int i = 0; i < x.size(); ++i) {
                    first.push_back(x[i]);
                    second.push_back(width[x[i]]);
                }
            }

            std::pair<std::vector<size_t>, std::vector<size_t>> pair(first, second);
            return pair;
        }


        std::pair<typename dynamic::index_type, typename dynamic::index_type>
        get_all_different_indexes_and_dimensions(const typename dynamic::index_type &same_indexes) {
            typename dynamic::index_type first;
            typename dynamic::index_type second;

            for (int i = 0; i < width.size(); ++i) {
                //same_indexes does not contain the index i
                if (std::find(same_indexes.begin(), same_indexes.end(), i) == same_indexes.end()) {
                    first.push_back(i);
                    second.push_back(width[i]);
                }
            }

            std::pair<typename dynamic::index_type, typename dynamic::index_type> pair(first, second);
            return pair;
        }

        typename dynamic::index_type
        get_different_actual_position(const typename dynamic::index_type &different_indexes,
                                      const typename dynamic::index_type &actual_position) {
            typename dynamic::index_type result;

            for (int i = 0; i < actual_position.size(); ++i) {
                //different_indexes contains the index i
                if (std::find(different_indexes.begin(), different_indexes.end(), i) != different_indexes.end()) {
                    result.push_back(actual_position[i]);
                }
            }
            return result;
        }

        bool indexes_are_equal(const typename dynamic::index_type &same_indexes,
                               const typename dynamic::index_type &actual_position) {
            size_t value = actual_position[same_indexes[0]];
            for (int i = 1; i < same_indexes.size(); ++i) {
                if (actual_position[same_indexes[i]] != value)
                    return false;
            }
            return true;
        }

        typename dynamic::index_type get_different_positions(const typename dynamic::index_type &this_actual_position,const typename dynamic::index_type &other_actual_position,std::vector<std::pair<size_t, size_t>> different_this,std::vector<std::pair<size_t, size_t>> different_other,tensor<T, dynamic> &t) {

            typename dynamic::index_type result;

            for (int i = 0; i < this_actual_position.size(); ++i) {
                std::pair<size_t, size_t> pair_to_find(i, width[i]);
                //pair_to_find in different_this
                if (std::find(different_this.begin(), different_this.end(), pair_to_find) != different_this.end())
                    result.push_back(this_actual_position[i]);
            }

            for (int i = 0; i < other_actual_position.size(); ++i) {
                std::pair<size_t, size_t> pair_to_find(i, t.width[i]);
                //pair_to_find in different_other
                if (std::find(different_other.begin(), different_other.end(), pair_to_find) != different_other.end())
                    result.push_back(other_actual_position[i]);
            }

            return result;
        }

        T *get_start_pointer_of_data() {
            return &(data->operator[](0));
        }
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
    };


    template<typename T, size_t R, int ... indexes>
    class proxy_tensor<T, rank<R>, index::index_tr<indexes>...> : tensor<T, rank<R>> {
    public:
        proxy_tensor(const std::vector<size_t> &dimensions) : tensor<T, rank<R>>(dimensions) {
            assert(R == sizeof...(indexes));

            std::array<int, sizeof...(indexes)> a = {indexes...};
            for (int i = 0; i < a.size(); i++) {
                indexesNotation[i] = a[i];
            }
        };

        typename rank<sizeof...(indexes)>::index_type indexesNotation;

        // specialized iterator type
        typedef reserved::iterator<T, rank<R>> iterator;

        iterator begin() { return tensor<T, rank<R>>::begin(); }

        iterator end() { return tensor<T, rank<R>>::end(); }

        // Overload operator *
        template<size_t R2, int ... second_indexes>
        tensor<T, rank<R + R2 - num_matched_indexes<my_set<indexes...>, my_set<second_indexes...>>::value -
                       num_matched_indexes<my_set<second_indexes...>, my_set<indexes...>>::value>>
        operator*(proxy_tensor<T, rank<R2>, index::index_tr<second_indexes>...> s) {
            const size_t dimension_of_result_tensor =
                    R + R2 - num_matched_indexes<my_set<indexes...>, my_set<second_indexes...>>::value -
                    num_matched_indexes<my_set<second_indexes...>, my_set<indexes...>>::value;

            typename rank<R2>::index_type t_IndexesNotation = s.get_indexes_notation();

            //array of pair: accessed_index = the first value correspond to the position of the same indexes, the second value contains the corresponding dimension
            std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<indexes...>, my_set<second_indexes...>>::value> same_this;
            std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<second_indexes...>, my_set<indexes...>>::value> same_other;

            std::pair<std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<indexes...>,my_set<second_indexes...>>::value>,std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<second_indexes...>,my_set<indexes...>>::value>> same_vectors = get_same_vectors(*this, s);

            same_this = same_vectors.first;
            same_other = same_vectors.second;

            static_assert(!same_this.empty(), "\n*** No common indexes to sum! ***");

            //array of pair: the first value correspond to the position of the different indexes, the second value is the corresponding dimension
            std::array<std::pair<size_t, size_t>, R - num_matched_indexes<my_set<indexes...>, my_set<second_indexes...>>::value> different_this;

            std::array<std::pair<size_t, size_t>, R2 - num_matched_indexes<my_set<second_indexes...>, my_set<indexes...>>::value> different_other;

            std::pair<std::array<std::pair<size_t, size_t>,R - num_matched_indexes<my_set<indexes...>, my_set<second_indexes...>>::value>, std::array<std::pair<size_t, size_t>,R2 - num_matched_indexes<my_set<second_indexes...>, my_set<indexes...>>::value>> different_vectors = get_different_vectors(*this, s);

            different_this = different_vectors.first;
            different_other = different_vectors.second;

            std::vector<size_t> indexes_to_initilize_tensor;

            if (different_this.empty()) {
                indexes_to_initilize_tensor = {1};
            } else {
                indexes_to_initilize_tensor = get_merged_different_dimensions(different_this, different_other);
            }

            // Tensor to be returned, initilized with its correct dimensions
            tensor<T, rank<dimension_of_result_tensor>> out_tensor(indexes_to_initilize_tensor);


            // "out_tensor" initialized to 0
            for (auto iter = out_tensor.begin(); iter != out_tensor.end(); ++iter) {
                *iter = 0;
            }

            size_t result[R + R2 - num_matched_indexes<my_set<indexes...>, my_set<second_indexes...>>::value -
                          num_matched_indexes<my_set<second_indexes...>, my_set<indexes...>>::value];

            // For each element in the tensor to be returned
            for (auto this_iter = begin(); this_iter != end(); ++this_iter) {
                for (auto other_iter = s.begin(); other_iter != s.end(); ++other_iter) {

                    std::array<size_t, R> this_actual_position = this_iter.get_indexes();

                    std::array<size_t, R2> other_actual_position = other_iter.get_indexes();

                    for (std::pair<size_t, T> x : same_this) {
                        for (std::pair<size_t, T> y : same_other) {
                            if (this_actual_position[x.first] == other_actual_position[y.first]) {

                                const size_t* accessed_index = get_different_positions(result, this_actual_position,other_actual_position,different_this,different_other,*this, s);

                                if (different_this.empty()) {
                                    accessed_index = {0};
                                }

                                out_tensor(accessed_index) += (*this_iter) * (*other_iter);
                            }
                        }
                    }
                }

            }


            out_tensor.indexesNotation = get_merged_different_indexes_notation(indexesNotation, t_IndexesNotation, s);

            return out_tensor;

        }

        template<typename, class, class ...>
        friend class proxy_tensor;

    private:

        template<size_t R2, int ... second_indexes>
        std::array<size_t, R + R2 - num_matched_indexes<my_set<indexes...>, my_set<second_indexes...>>::value -
                   num_matched_indexes<my_set<second_indexes...>, my_set<indexes...>>::value> get_merged_different_indexes_notation(typename rank<R>::index_type& f,typename rank<R2>::index_type& s, proxy_tensor<T, rank<R2>, index::index_tr<second_indexes>...>){

            std::array<size_t, R + R2 - num_matched_indexes<my_set<indexes...>, my_set<second_indexes...>>::value -
                               num_matched_indexes<my_set<second_indexes...>, my_set<indexes...>>::value> result;
            int i = 0;

            for(int j = 0; j < R; j++){
                if(std::find(std::begin(s), std::end(s), f[j]) == std::end(s)){
                    result[i] = f[j];
                    i++;
                }
            }
            for(int j = 0; j < R2; j++){
                if(std::find(std::begin(f), std::end(f), s[j]) == std::end(f)){
                    result[i] = s[j];
                    i++;
                }
            }

            return result;
        }

        template<size_t R1, size_t R2, int ... first_indexes, int ... second_indexes>
        std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value>
        get_same_pairs(proxy_tensor<T, rank<R1>, index::index_tr<first_indexes>...> first_tensor,
                       proxy_tensor<T, rank<R2>, index::index_tr<second_indexes>...> second_tensor) {

            std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value> result;

            int temp = 0;

            for (int i = 0; i < first_tensor.indexesNotation.size(); ++i) {
                for (int j = 0; j < second_tensor.indexesNotation.size(); j++) {
                    if (first_tensor.indexesNotation[i] == second_tensor.indexesNotation[j]) {
                        std::pair<size_t, T> pair(i, first_tensor.width[i]);
                        result[temp] = pair;
                        ++temp;
                    }
                }
            }

            return result;
        }

        template<size_t R1, size_t R2, int ... first_indexes, int ... second_indexes>
        std::pair<std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value>, std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<second_indexes...>, my_set<first_indexes...>>::value>>
        get_same_vectors(const proxy_tensor<T, rank<R1>, index::index_tr<first_indexes>...> &first_tensor,
                         const proxy_tensor<T, rank<R2>, index::index_tr<second_indexes>...> &second_tensor) {

            std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value> first = get_same_pairs(
                    first_tensor, second_tensor);
            std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<second_indexes...>, my_set<first_indexes...>>::value> second = get_same_pairs(
                    second_tensor, first_tensor);

            std::pair<std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value>, std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<second_indexes...>, my_set<first_indexes...>>::value>> result(
                    first, second);
            return result;
        }

        template<size_t R1, size_t R2, int ... first_indexes, int ... second_indexes>
        std::array<std::pair<size_t, size_t>,
                R1 - num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value>
        get_different_pairs(const proxy_tensor<T, rank<R1>, index::index_tr<first_indexes>...> &first_tensor,
                            const proxy_tensor<T, rank<R2>, index::index_tr<second_indexes>...> &second_tensor) {

            std::array<std::pair<size_t, size_t>,R1 - num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value> result;

            int temp = 0;
            std::array<std::pair<size_t, size_t>, num_matched_indexes<my_set<first_indexes...>,my_set<second_indexes...>>::value> same_first = get_same_pairs(first_tensor, second_tensor);

            for (int i = 0; i < first_tensor.indexesNotation.size(); ++i) {
                std::pair<size_t, size_t> pair(i, first_tensor.width[i]);

                //same_first does not contain the pair
                if (std::find(same_first.begin(), same_first.end(), pair) == same_first.end()) {
                    result[temp] = pair;
                    ++temp;
                    //result.push_back(pair);
                }
            }
            return result;
        }

        template<size_t R1, size_t R2, int ... first_indexes, int ... second_indexes>
        std::pair<std::array<std::pair<size_t, size_t>, R1-num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value>, std::array<std::pair<size_t, size_t>,R2 - num_matched_indexes<my_set<second_indexes...>, my_set<first_indexes...>>::value>>
        get_different_vectors(const proxy_tensor<T, rank<R1>, index::index_tr<first_indexes>...> &first_tensor,
                              const proxy_tensor<T, rank<R2>, index::index_tr<second_indexes>...> &second_tensor) {

            std::array<std::pair<size_t, size_t>, R1-num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value> first = get_different_pairs(first_tensor, second_tensor);

            std::array<std::pair<size_t, size_t>, R2-num_matched_indexes<my_set<second_indexes...>, my_set<first_indexes...>>::value> second = get_different_pairs(second_tensor, first_tensor);

            std::pair<std::array<std::pair<size_t, size_t>, R1-num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value>, std::array<std::pair<size_t, size_t>,R2 - num_matched_indexes<my_set<second_indexes...>, my_set<first_indexes...>>::value>> result(first,second);

            return result;
        }

        template<size_t R1, size_t R2>
        std::vector<size_t>
        get_merged_different_dimensions(const std::array<std::pair<size_t, size_t>, R1> &different_this,
                                        const std::array<std::pair<size_t, size_t>, R2> &different_other) {

            std::vector<size_t> result;

            result.reserve(different_this.size() + different_other.size());

            for (std::pair<size_t, size_t> pair : different_this) {
                result.push_back(pair.second);
            }

            for (std::pair<size_t, size_t> pair : different_other) {
                result.push_back(pair.second);
            }

            return result;
        }

        template<size_t R1, size_t R2, int ... first_indexes, int ... second_indexes>
        size_t* get_different_positions(size_t* result, std::array<size_t, R1> this_actual_position, std::array<size_t,R2> other_actual_position,std::array<std::pair<size_t, size_t>,R1-num_matched_indexes<my_set<first_indexes...>, my_set<second_indexes...>>::value> different_this,std::array<std::pair<size_t, size_t>, R2-num_matched_indexes<my_set<second_indexes...>, my_set<first_indexes...>>::value> different_other,const proxy_tensor<T, rank<R1>, index::index_tr<first_indexes>...> &first_tensor,const proxy_tensor<T, rank<R2>, index::index_tr<second_indexes>...> &second_tensor) {

            int j = 0;

            for (int i = 0; i < this_actual_position.size(); ++i) {
                std::pair<size_t, size_t> pair_to_find(i, first_tensor.width[i]);
                //pair_to_find in different_this
                if (std::find(different_this.begin(), different_this.end(), pair_to_find) != different_this.end()){
                    result[j] = this_actual_position[i];
                    j++;
                }
            }

            for (int i = 0; i < other_actual_position.size(); ++i) {
                std::pair<size_t, size_t> pair_to_find(i, second_tensor.width[i]);
                //pair_to_find in different_other
                if (std::find(different_other.begin(), different_other.end(), pair_to_find) != different_other.end()) {
                    result[j] = other_actual_position[i];
                    j++;
                }
            }

            return result;
        }


    };


    // tensor specialization for fixed-rank
    template<typename T, size_t R>
    class tensor<T, rank<R>> {
    public:
        // C-style constructor with implicit rank and pointer to array of dimensions
        // all other constructors are redirected to this one
        tensor(const size_t dimensions[R]) {
            size_t *wptr = &(width[0]), *endp = &(width[0]) + R;
            while (wptr != endp) *(wptr++) = *(dimensions++);
            stride[R - 1] = 1;
            for (int i = R - 1; i != 0; --i) {
                stride[i - 1] = stride[i] * width[i];
            }
            data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
            start_ptr = &(data->operator[](0));
        }

        tensor(const std::vector<size_t> &dimensions) : tensor(&dimensions[0]) { assert(dimensions.size() == R); }

        template<typename...Dims>
        tensor(Dims...dims) : width({static_cast<const size_t>(dims)...}) {
            static_assert(sizeof...(dims) == R, "size mismatch");

            stride[R - 1] = 1UL;
            for (size_t i = R - 1UL; i != 0UL; --i) {
                stride[i - 1] = stride[i] * width[i];
            }
            data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
            start_ptr = &(data->operator[](0));
        }

        tensor(const tensor<T, rank < R>>

        &X) = default;

        tensor(tensor<T, rank<R>> &&X) = default;

        // all tensor types are friend
        // this are used by alien copy constructors, i.e. copy constructors copying different tensor types.
        template<typename, typename> friend
        class tensor;

        tensor(const tensor<T, dynamic> &X) : data(X.data), width(X.width.begin(), X.width.end()),
                                              stride(X.stride.begin(), X.stride.end()), start_ptr(X.start_ptr) {
            assert(X.get_rank() == R);
        }

/*---------------------------------------------------------------------------------------------------------*/
/*--------- EINSTEIN's NOTATION ---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/

        //typename rank<R>::index_type ==> std::array<size_t,R>

        // Function which returns the Einstein's notation indexes
        std::array<size_t, R> get_indexes_notation() {
            return indexesNotation;
        }

        void set_indexes_notation(size_t position, size_t value) {
            indexesNotation[position] = value;
        }

        template<typename K>
        std::vector<size_t> variadic_to_array(std::vector<size_t> app, K value) {
            app.push_back(value.value);
            return app;
        }

        template<typename First, typename... Rest>
        std::vector<size_t> variadic_to_array(std::vector<size_t> app, First firstValue, Rest... rest) {
            app = variadic_to_array(app, firstValue);
            return variadic_to_array(app, rest...);
        }

        // Overload operator ()
        template<int ... idxInt>
        tensor<T, rank<R - num_same_indexes<my_set<>, idxInt...>::value>> operator()(index::index_tr<idxInt>... inds) {

            assert(sizeof...(inds) == R);

            std::array<size_t, sizeof...(idxInt)> indexes = {idxInt...};

            //array of vectors containing the positions of same indexes. if a_{ijikjk} same_indexes will be { {0,2}, {1,4}, {3,5} }
            std::array<std::vector<size_t>, num_same_grouped_indexes<my_set<>, idxInt...>::value> same_grouped_indexes =
                    get_same_grouped_indexes < idxInt...>();

            //the first array contains the same indexes, while the second contains the corresponding dimensions
            std::pair<typename rank<num_same_indexes<my_set<>, idxInt...>::value>::index_type, typename rank<num_same_indexes<my_set<>, idxInt...>::value>::index_type> all_same_indexes_and_dimensions =
                    get_all_same_indexes_and_dimensions <same_grouped_indexes.size(), num_same_indexes<my_set<>, idxInt...>::value>(same_grouped_indexes);

            //the first array contains the different indexes, while the second contains the corresponding dimensions
            std::pair<typename rank<R - num_same_indexes<my_set<>, idxInt...>::value>::index_type, typename rank<R-num_same_indexes<my_set<>, idxInt...>::value>::index_type> all_different_indexes_and_dimensions = get_all_different_indexes_and_dimensions < all_different_indexes_and_dimensions.first.size() > (all_same_indexes_and_dimensions.first);

            size_t a = num_same_indexes<my_set<>, idxInt...>::value;

            std::vector<size_t> indexes_to_initialize;

            if (num_same_indexes<my_set<>, idxInt...>::value == R) {
                indexes_to_initialize.push_back(1);
            }
            else {
                for (size_t x : all_different_indexes_and_dimensions.second) {
                    indexes_to_initialize.push_back(x);
                }

            }

            tensor<T, rank<std::max(R - num_same_indexes<my_set<>, idxInt...>::value, size_t(1))>> tensor_out(indexes_to_initialize);

            if (same_grouped_indexes.empty()) {  // there are no indexes to contract,
                // so just copy original tensor into tensor_out
                auto iter_tensor_out = tensor_out.begin();
                for (tensor::iterator iter = begin(); iter != end(); ++iter) {
                    *iter_tensor_out = *iter;

                    iter_tensor_out++;
                }
            }

            for (tensor::iterator iter = begin(); iter != end(); ++iter) {
                for (std::vector<size_t> x : same_grouped_indexes) {
                    if (indexes_are_equal < num_same_indexes<my_set<>, idxInt...>::value >
                        (all_same_indexes_and_dimensions.first, iter.get_indexes())) {

                        std::array<size_t, std::max(R - num_same_indexes<my_set<>, idxInt...>::value, size_t(1))> array;

                        array = get_different_actual_position <std::max(R - num_same_indexes<my_set<>, idxInt...>::value, size_t(1)) > (all_different_indexes_and_dimensions.first, iter.get_indexes());

                        std::vector<size_t> vector(std::begin(array), std::end(array));
                        T &value = tensor_out(vector);

                        T der_iter = *iter;
                        value += *iter;
                    }
                }

                if (num_same_indexes<my_set<>, idxInt...>::value == 0) {
                    std::array<size_t, R> array = iter.get_indexes();
                    std::vector<size_t> vector(std::begin(array), std::end(array));
                    tensor_out(vector) = *iter;
                }
            }

            for (int i = 0; i < all_different_indexes_and_dimensions.first.size(); ++i) {
                tensor_out.set_indexes_notation(i, indexes[all_different_indexes_and_dimensions.first[i]]);
            }

            return tensor_out;
        }

        // Overload operator +
        tensor<T, rank<R>> operator+(const tensor<T, rank <R>>& t){
            return operators_plus_minus_temp(t, '+');
        }

        // Overload operator -
        tensor<T, rank<R>> operator-(const tensor<T, rank <R>>& t){
            return operators_plus_minus_temp(t, '-');
        }
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/

        // not static so that it can be called with . rather than ::
        constexpr size_t get_rank() const { return R; }

        // direct accessors as for dynamic tensor
        T &operator()(const size_t dimensions[R]) const {
            T *ptr = start_ptr;
            for (size_t i = 0; i != R; ++i) ptr += dimensions[i] * stride[i];
            return *ptr;
        }

        T &at(const size_t dimensions[R]) const {
            T *ptr = start_ptr;
            for (size_t i = 0; i != R; ++i) {
                assert(dimensions[i] < width[i]);
                ptr += dimensions[i] * stride[i];
            }
            return *ptr;
        }

        T &operator()(const std::vector<size_t> &dimensions) const {
            assert(dimensions.size() == R);
            return operator()(&dimensions[0]);
        }

        T &at(const std::vector<size_t> &dimensions) const {
            assert(dimensions.size() == R);
            return at(&dimensions[0]);
        }

        // could use std::enable_if rather than static assert!
        template<typename...Dims>
        T &operator()(Dims...dimensions) const {
            static_assert(sizeof...(dimensions) == R, "rank mismatch");
            return operator()({static_cast<const size_t>(dimensions)...});
        }

        template<typename...Dims>
        T &at(Dims...dimensions) const {
            static_assert(sizeof...(dimensions) == R, "rank mismatch");
            return at({static_cast<const size_t>(dimensions)...});
        }

        // specialized iterator type
        typedef reserved::iterator<T, rank<R>> iterator;

        iterator begin() { return iterator(width, stride, start_ptr); }

        iterator end() {
            iterator result = begin();
            result.idx[0] = width[0];
            result.ptr += width[0] * stride[0];
            return result;
        }

        // specialized index_iterator type
        typedef reserved::index_iterator<T> index_iterator;

        index_iterator begin(size_t index, const size_t dimensions[R]) const {
            return index_iterator(stride[index], &operator()(dimensions) - dimensions[index] * stride[index]);
        }

        index_iterator end(size_t index, const size_t dimensions[R]) const {
            return index_iterator(stride[index],
                                  &operator()(dimensions) + (width[index] - dimensions[index]) * stride[index]);
        }

        index_iterator begin(size_t index, const std::vector<size_t> &dimensions) const {
            return index_iterator(stride[index], &operator()(dimensions) - dimensions[index] * stride[index]);
        }

        index_iterator end(size_t index, const std::vector<size_t> &dimensions) const {
            return index_iterator(stride[index], &operator()(dimensions) + (width[index] - dimensions[index]) * stride[index]);
        }

        // slicing operations return lower-rank tensors
        tensor<T, rank<R - 1>> slice(size_t index, size_t i) const {
            assert(index < R);
            tensor<T, rank<R - 1>> result;
            result.data = data;
            for (size_t i = 0; i != index; ++i) {
                result.width[i] = width[i];
                result.stride[i] = stride[i];
            }
            for (size_t i = index; i != R - 1U; ++i) {
                result.width[i] = width[i + 1];
                result.stride[i] = stride[i + 1];
            }
            result.start_ptr = start_ptr + i * stride[index];

            return result;
        }

        tensor<T, rank<R - 1>> operator[](size_t i) const { return slice(0, i); }


        // window operations do not change rank
        tensor<T, rank<R>> window(size_t index, size_t begin, size_t end) const {
            tensor<T, rank<R>> result(*this);
            result.width[index] = end - begin;
            result.start_ptr += result.stride[index] * begin;
            return result;
        }

        tensor<T, rank<R>> window(const size_t begin[], const size_t end[]) const {
            tensor<T, rank<R>> result(*this);
            for (size_t i = 0; i != R; ++i) {
                result.width[i] = end[i] - begin[i];
                result.start_ptr += result.stride[i] * begin[i];
            }
            return result;
        }

        tensor<T, dynamic> window(const std::vector<size_t> &begin, const std::vector<size_t> &end) const {
            return window(&begin[0], &end[0]);
        }

        // flatten operations change rank in a way that is not known at compile time
        // would need a different interface to provide that info at compile time,
        // but the operation should not be time-critical
        tensor<T, dynamic> flatten(size_t begin, size_t end) const {
            tensor<T, dynamic> result;
            result.stride.insert(result.stride.end(), stride.begin(), stride.begin() + begin);
            result.stride.insert(result.stride.end(), stride.begin() + end, stride.end());
            result.width.insert(result.width.end(), width.begin(), width.begin() + begin);
            result.stride.insert(result.stride.end(), stride.begin() + end, stride.end());
            for (size_t i = begin; i != end; ++i) result.width[end] *= width[i];
            result.start_prt = start_ptr;
            result.data = data;
            return result;
        }


        friend class tensor<T, rank<R + 1>>;

        template<typename, class, class ...>
        friend
        class proxy_tensor;

        T *start_ptr;


    private:

        tensor() = default;

        std::shared_ptr<std::vector<T>> data;
        typename rank<R>::width_type width;
        typename rank<R>::index_type stride;
        //T* start_ptr;

/*---------------------------------------------------------------------------------------------------------*/
/*--------- EINSTEIN's NOTATION ---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
        // Vector of indexes to use Eintein's notation
        typename rank<R>::indexes_notation_type indexesNotation;

        tensor<T, rank<R>> operators_plus_minus_temp(tensor<T, rank<R>> t, const char sign) {

            assert(get_rank() == t.get_rank());

            typename rank<R>::indexes_notation_type t_IndexesNotation = t.get_indexes_notation();

            //vector contains pair corresponding to the paired indexes of the tensors to sum
            std::array<std::pair<size_t, size_t>, R> pair_indexes = get_paired_indexes(t, indexesNotation, t_IndexesNotation);

            std::vector<size_t> indexes_to_initialize(width.begin(), width.end());
            tensor<T, rank<R>> out_tensor(indexes_to_initialize);   // initialize only through an std::vector

            for (tensor::iterator this_iter = begin(); this_iter != end(); ++this_iter) {
                for (tensor::iterator other_iter = t.begin(); other_iter != t.end(); ++other_iter) {

                    typename rank<R>::index_type this_actual_position = this_iter.get_indexes();
                    typename rank<R>::index_type other_actual_position = other_iter.get_indexes();

                    bool flag = true;
                    for (std::pair<size_t, T> x : pair_indexes) {
                        if (this_actual_position[x.first] != other_actual_position[x.second]) {
                            //std::vector<size_t> accessed_index;
                            flag = false;
                        }
                    }

                    if (flag) {
                        std::vector<size_t> actual_position_vector(this_actual_position.begin(), this_actual_position.end());

                        if (sign == '+') {
                            //out_tensor(this_actual_position) = (*this_iter) + (*other_iter);
                            out_tensor(actual_position_vector) = (*this_iter) + (*other_iter);
                        } else if (sign == '-') {
                            //out_tensor(this_actual_position) = (*this_iter) - (*other_iter);
                            out_tensor(actual_position_vector) = (*this_iter) - (*other_iter);
                        }
                    }
                }
            }

            return out_tensor;
        }

        std::array<std::pair<size_t, size_t>, R> get_paired_indexes(tensor<T, rank<R>> &second_tensor, const typename rank<R>::index_type &first_indexes_notation, const typename rank<R>::index_type &second_indexes_notation) {

            std::array<std::pair<size_t, size_t>, R> result;
            int temp = 0;

            for (size_t i = 0; i < first_indexes_notation.size(); i++) {

                auto iter = std::find(second_indexes_notation.begin(), second_indexes_notation.end(), first_indexes_notation[i]);
                assert(iter != second_indexes_notation.end());

                std::pair<size_t, size_t> pair(i, std::distance(second_indexes_notation.begin(), iter));
                result[temp] = pair;
                ++temp;
                //result.push_back(pair);
            }

            return result;
        }


        template<int ... indexes>
        std::array<std::vector<size_t>, num_same_grouped_indexes<my_set<>, indexes...>::value>
        get_same_grouped_indexes() {

            std::array<std::vector<size_t>, num_same_grouped_indexes<my_set<>, indexes...>::value> result;
            int temp = 0;

            typename rank<R>::index_type copy_indexes = {indexes...};

            std::vector<size_t> selected_indexes;

            for (size_t i = 0; i < copy_indexes.size(); ++i) {

                std::vector<size_t> local_duplicates = {i};

                for (size_t j = 0; j < copy_indexes.size(); ++j) {
                    if (i < j &&
                        std::find(selected_indexes.begin(), selected_indexes.end(), i) == selected_indexes.end() &&
                        copy_indexes[i] == copy_indexes[j]) {
                        local_duplicates.push_back(j);
                    }
                }

                if (local_duplicates.size() > 1) {
                    selected_indexes.insert(selected_indexes.end(), local_duplicates.begin(), local_duplicates.end());

                    result[temp] = local_duplicates;
                    temp++;
                    //result.push_back(local_duplicates);
                }
            }

            return result;
        }

        template<int H, int S>
        std::pair<std::array<size_t, S>, std::array<size_t, S> >
        get_all_same_indexes_and_dimensions(const std::array<std::vector<size_t>, H> &grouped_indexes) {

            typename rank<S>::index_type first;
            int temp = 0;
            typename rank<S>::index_type second;
            int temp1 = 0;

            for (std::vector<size_t> x : grouped_indexes) {
                for (int i = 0; i < x.size(); ++i) {
                    //first.push_back(x[i]);
                    first[temp] = x[i];
                    temp++;
                    //second.push_back(width[ x[i] ]);
                    second[temp1] = width[x[i]];
                    ++temp1;
                }
            }

            std::pair<std::array<size_t, S>, std::array<size_t, S>> pair(first, second);
            return pair;
        }


        template<int S>
        std::pair<typename rank<S>::index_type, typename rank<S>::index_type>
        get_all_different_indexes_and_dimensions(const typename rank<R - S>::index_type &same_indexes) {

            typename rank<S>::index_type first;
            int temp = 0;
            typename rank<S>::index_type second;
            int temp1 = 0;

            for (int i = 0; i < width.size(); ++i) {
                //same_indexes does not contain the index i
                if (std::find(same_indexes.begin(), same_indexes.end(), i) == same_indexes.end()) {
                    //first.push_back(i);
                    first[temp] = i;
                    ++temp;
                    //second.push_back(width[i]);
                    second[temp1] = width[i];
                    ++temp1;
                }
            }

            std::pair<typename rank<S>::index_type, typename rank<S>::index_type> pair(first, second);
            return pair;
        }

        template<int S>
        typename rank<S>::index_type
        get_different_actual_position(const typename rank<S>::index_type &different_indexes,
                                      const typename rank<R>::index_type &actual_position) {

            typename rank<S>::index_type result;
            int temp = 0;

            for (int i = 0; i < actual_position.size(); i++) {
                //different_indexes contains the index i
                if (std::find(different_indexes.begin(), different_indexes.end(), i) != different_indexes.end()) {
                    result[temp] = actual_position[i];
                    temp++;
                }
            }
            return result;
        }

        template<int S>
        bool indexes_are_equal(const typename rank<S>::index_type &same_indexes,
                               const typename rank<R>::index_type &actual_position) {
            size_t value = actual_position[same_indexes[0]];
            for (int i = 1; i < same_indexes.size(); ++i) {
                if (actual_position[same_indexes[i]] != value)
                    return false;
            }
            return true;
        }

        T *get_start_pointer_of_data() {
            return &(data->operator[](0));
        }
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
    };


    // tensor specialization for rank 1
    // in this case splicing provides reference to data element
    template<typename T>
    class tensor<T, rank<1>> {
    public:

        tensor(size_t dimension) {
            data = std::make_shared<std::vector<T>>(dimension);
            width[0] = dimension;
            stride[0] = 1;
            start_ptr = &*(data->begin());
        }

        tensor(const std::vector<size_t> &dimensions) {
            assert(dimensions.size() == 1);
            data = std::make_shared<std::vector<T>>(dimensions.at(0));
            width[0] = dimensions.at(0);
            stride[0] = 1;
            start_ptr = &*(data->begin());
        }

/*---------------------------------------------------------------------------------------------------------*/
/*--------- EINSTEIN's NOTATION ---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/

        // Function which returns the Einstein's notation index
        size_t get_index_notation() {
            return indexNotation;
        }

        // Overload operator ()
        tensor<T, rank<1>> operator()(const std::vector<index::index_t> &indexes) {
            return operator()(indexes.at(0));
        }

        tensor<T, rank<1>> operator()(const std::array<index::index_t, 1> &indexes) {
            return operator()(indexes.at(0));
        }

        tensor<T, rank<1>> operator()(const index::index_t index) {
            indexNotation = index.getValue();

            return *this;
        }

        virtual // Overload operator *
        tensor<T, rank<1>> operator*(tensor<T, rank < 1>>

        t){
            // Check that the values of the dimension of the two rank 1 tensors have the same value
            assert(t.width.at(0) == width.at(0));

            // Check that the Einstein index of the two rank 1 tensors is the same and it is index_t i = 0 (the first index available)
            assert(t.indexNotation == indexNotation && indexNotation == 0);

            tensor<T, rank<1>> result(1);    // returns a tensor with rank 1 and dimension 1

            for (int i = 0; i != width.at(0); ++i) {
                result[0] += at(i) * t.at(i);
            }

            return result;
        }

        // Overload operator +
        tensor<T, rank<1>> operator+(tensor<T, rank < 1>>

        t){
            return operators_plus_minus_temp(t, '+');
        }

        // Overload operator -
        tensor<T, rank<1>> operator-(tensor<T, rank < 1>>

        t){
            return operators_plus_minus_temp(t, '-');
        }
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/

        // all tensor types are friend
        // this are used by alien copy constructors, i.e. copy constructors copying different tensor types.
        template<typename, typename> friend
        class tensor;

        constexpr size_t get_rank() const { return 1; }

        // direct accessors as for dynamic tensor
        T &operator()(size_t d) const {
            T &result = start_ptr[d * stride[0]];
            return start_ptr[d * stride[0]];
        }

        T &at(size_t d) const {
            assert(d < width[0]);
            return start_ptr[d * stride[0]];
        }

        T &operator()(const size_t dimensions[1]) const {
            return operator()(dimensions[0]);
        }

        T &at(const size_t dimensions[1]) const {
            return operator()(dimensions[0]);
        }

        T &operator()(const std::vector<size_t> &dimensions) const {
            assert(dimensions.size() == 1);
            return operator()(dimensions[0]);
        }

        T &at(const std::vector<size_t> &dimensions) const {
            assert(dimensions.size() == 1);
            return operator()(dimensions[0]);
        }

        // could use std::enable_if rather than static assert!
        T &slice(size_t index, size_t i) const {
            assert(index == 0);
            return *(start_ptr + i * stride[0]);
        }

        T &operator[](size_t i) { return *(start_ptr + i * stride[0]); }

        tensor<T, rank<1>> window(size_t begin, size_t end) const {
            tensor<T, rank<1>> result(*this);
            result.width[0] = end - begin;
            result.start_ptr += result.stride[0] * begin;
            return result;
        }


        typedef T *iterator;

        iterator begin(size_t= 0) { return start_ptr; }

        iterator end(size_t= 0) { return start_ptr + width[0] * stride[0]; }


        friend class tensor<T, rank<2>>;

        friend class tensor<T, rank<3>>;


    private:

        tensor() = default;

        std::shared_ptr<std::vector<T>> data;
        rank<1>::width_type width;
        rank<1>::index_type stride;
        T *start_ptr;

/*---------------------------------------------------------------------------------------------------------*/
/*--------- EINSTEIN's NOTATION ---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
        size_t indexNotation;

        void set_indexes_notation(size_t position, size_t value) {
            indexNotation = value;
        }

        tensor<T, rank<1>> operators_plus_minus_temp(tensor<T, rank<1>> t, const char sign) {
            // Check that the values of the dimension of the two rank 1 tensors have the same value
            assert(t.width.at(0) == width.at(0));

            // Check that the Einstein index of the two rank 1 tensors is the same and it is index_t i = 0 (the first index)
            assert(t.indexNotation == indexNotation && indexNotation == 0);

            tensor<T, rank<1>> result(width.at(0));

            if (sign == '+') {
                for (int i = 0; i != width.at(0); ++i) {
                    result[i] = at(i) + t.at(i);
                }
            } else if (sign == '-') {
                for (int i = 0; i != width.at(0); ++i) {
                    result[i] = at(i) - t.at(i);
                }
            }

            return result;
        }
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
    };


}; //namespace tensor

#endif //TENSOR
