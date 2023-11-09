#pragma once

namespace mtr {

/// N-Dimensional Tensor
///
/// Added by emc to test out performance improvements in array
/// accessors.
///
/// @tparam T The type of the elements in the tensor.
/// @tparam N The number of ranks in the tensor.
template <typename T, int N>
class Tensor {
  /// Tensor order (rank)
  static constexpr size_t order_ = N;
  
private:
  /// Tensor dimensions (number of elements in each rank).
  size_t dims_[N];
  /// Array strides (number of elements to skip to get to the next
  /// element in each rank). The last element is the total number of
  /// elements in the tensor.
  size_t strides_[N+1];
  std::shared_ptr <T []> array_;

public:
  //------------------------------------------------------------------
  // Default constructor
  Tensor () = delete;
  //------------------------------------------------------------------
  // Set dimensions from an array
  Tensor (const size_t *dims);
  //------------------------------------------------------------------
  // --- 1D to 7D array ---
  Tensor (size_t dim0=1,
          size_t dim1=1,
          size_t dim2=1,
          size_t dim3=1,
          size_t dim4=1,
          size_t dim5=1,
          size_t dim6=1);
  //------------------------------------------------------------------
  /// Construct the array, allocate storage.
  template<typename... dim_t>
  Tensor(size_t dim0, dim_t... dims) {
    // Set the first dimension and stride
    dims_[0] = dim0;
    strides_[0] = 1;
    strides_[1] = dim0;
    // Populate the rest of the array dimensions and strides
    const size_t nargs = sizeof...(dims);
    std::va_list args;
    va_start(args, dim0);
    // Fill in the provided dimensions and corresponding strides
    for (auto d=0; d<order_ && d<nargs; ++d) {
      dims_[d] = va_arg(args, size_t);
      strides_[d+1] = strides_[d]*dims_[d];
    }
    va_end(args);
    // Fill the rest of the dimensions with 1
    for (auto d=nargs; d<order_; ++d) {
      dims_[d] = 1;
      strides_[d+1] = strides_[d];
    }
    // Allocate the array
    array_ = std::shared_ptr<T[]> (new T[strides_[order_]]);
  }
  //------------------------------------------------------------------
  // Copy constructor (Shallow copy)
  Tensor(const Tensor & temp);

  //------------------------------------------------------------------
  // Overload operator() for multi-dimensional array access
  template<typename... dim_t>
  T& operator()(size_t idx0, dim_t... idxs) const {
    // Compute the index
    size_t idx = idx0;
    const size_t nargs = sizeof...(idxs);
    std::va_list args;
    va_start(args, idx0);
    for (auto d=1; d<=nargs; ++d) {
      idx += va_arg(args, size_t)*strides_[d];
    }
    va_end(args);
    // Return the element
    return array_[idx];
  }
    
  //------------------------------------------------------------------
  // Overload copy assignment operator
  Tensor& operator= (const Tensor& temp) {
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
      memcpy(dims_, temp.dims_, sizeof(dims_));
      array_ = temp.array_;
    }
    return *this;
  }

  //return array size
  size_t size() const { return strides_[order_]; }

  // return array dims
  size_t dims(size_t i) const { return dims_[i]; }
    
  // return array order (rank)
  size_t order() const { return order_; }
    
  //return pointer
  T* pointer() const { return array_.get(); }

  // Deconstructor
  ~Tensor () = default;

}; // End of Tensor

//---carray class declarations---


//return size
template <typename T>
inline size_t Tensor<T>::size() const {
  return length_;
}

template <typename T>
inline size_t Tensor<T>::dims(size_t i) const {
  assert(i < order_ && "Tensor order (rank) does not match constructor, dim[i] does not exist!");
  assert(i >= 0 && dims_[i]>0 && "Access to Tensor dims is out of bounds!");
  return dims_[i];
}

template <typename T>
inline size_t Tensor<T>::order() const {
  return order_;
}


template <typename T>
inline T* Tensor<T>::pointer() const{
  return array_.get();
}

//destructor
template <typename T>
Tensor<T>::~Tensor() {}

//----endof carray class definitions----


//6. ViewTensor
// indicies are [0:N-1]
template <typename T>
class ViewTensor {

private:
  size_t dims_[7];
  size_t length_; // Length of 1D array
  size_t order_;  // tensor order (rank)
  T * array_;
    
public:
    
  // Default constructor
  ViewTensor ();
    
  //--- 1D to 7D array ---
  ViewTensor(T *array,
             size_t dim0);

  ViewTensor(T *array,
             size_t dim0,
             size_t dim1);
    
  ViewTensor(T *some_array,
             size_t dim0,
             size_t dim1,
             size_t dim2);
    
  ViewTensor(T *some_array,
             size_t dim0,
             size_t dim1,
             size_t dim2,
             size_t dim3);
    
  ViewTensor (T *some_array,
              size_t dim0,
              size_t dim1,
              size_t dim2,
              size_t dim3,
              size_t dim4);

  ViewTensor (T *some_array,
              size_t dim0,
              size_t dim1,
              size_t dim2,
              size_t dim3,
              size_t dim4,
              size_t dim5);
 
  ViewTensor (T *some_array,
              size_t dim0,
              size_t dim1,
              size_t dim2,
              size_t dim3,
              size_t dim4,
              size_t dim5,
              size_t dim6);
    
  T& operator()(size_t i) const;
    
  T& operator()(size_t i,
                size_t j) const;
    
  T& operator()(size_t i,
                size_t j,
                size_t k) const;
    
  T& operator()(size_t i,
                size_t j,
                size_t k,
                size_t l) const;
  T& operator()(size_t i,
                size_t j,
                size_t k,
                size_t l,
                size_t m) const;
    
  T& operator()(size_t i,
                size_t j,
                size_t k,
                size_t l,
                size_t m,
                size_t n) const;
    
  T& operator()(size_t i,
                size_t j,
                size_t k,
                size_t l,
                size_t m,
                size_t n,
                size_t o) const;

  // calculate C = math(A,B)
  template <typename M>
  void operator=(M do_this_math);
    
  //return array size
  size_t size() const;
    
  // return array dims
  size_t dims(size_t i) const;
    
  // return array order (rank)
  size_t order() const;

  // return pointer
  T* pointer() const;
    
}; // end of ViewTensor

//class definitions

//constructors

//no dim
template <typename T>
ViewTensor<T>::ViewTensor() {
  array_ = NULL;
  length_ = order_ = 0;
  for (int i = 0; i < 7; i++) {
    dims_[i] = 0;
  }
}

//1D
template <typename T>
ViewTensor<T>::ViewTensor(T *array,
                          size_t dim0)
{
  dims_[0] = dim0;
  order_ = 1;
  length_ = dim0;
  array_ = array;
}

//2D
template <typename T>
ViewTensor<T>::ViewTensor(T *array,
                          size_t dim0,
                          size_t dim1)
{
  dims_[0] = dim0;
  dims_[1] = dim1;
  order_ = 2;
  length_ = dim0 * dim1;
  array_ = array;
}

//3D
template <typename T>
ViewTensor<T>::ViewTensor (T *array,
                           size_t dim0,
                           size_t dim1,
                           size_t dim2)
{
  dims_[0] = dim0;
  dims_[1] = dim1;
  dims_[2] = dim2;
  order_ = 3;
  length_ = dim0 * dim1 * dim2;
  array_ = array;
}

//4D
template <typename T>
ViewTensor<T>::ViewTensor(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2,
                          size_t dim3)
{
  dims_[0] = dim0;
  dims_[1] = dim1;
  dims_[2] = dim2;
  dims_[3] = dim3;
  order_ = 4;
  length_ = dim0 * dim1 * dim2 * dim3;
  array_ = array;
}

//5D
template <typename T>
ViewTensor<T>::ViewTensor(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2,
                          size_t dim3,
                          size_t dim4)
{
  dims_[0] = dim0;
  dims_[1] = dim1;
  dims_[2] = dim2;
  dims_[3] = dim3;
  dims_[4] = dim4;
  order_ = 5;
  length_ = dim0 * dim1 * dim2 * dim3 * dim4;
  array_ = array;
}

//6D
template <typename T>
ViewTensor<T>::ViewTensor(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2,
                          size_t dim3,
                          size_t dim4,
                          size_t dim5)
{
  dims_[0] = dim0;
  dims_[1] = dim1;
  dims_[2] = dim2;
  dims_[3] = dim3;
  dims_[4] = dim4;
  dims_[5] = dim5;
  order_ = 6;
  length_ = dim0 * dim1 * dim2 * dim3 * dim4 * dim5;
  array_ = array;
}

//7D
template <typename T>
ViewTensor<T>::ViewTensor(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2,
                          size_t dim3,
                          size_t dim4,
                          size_t dim5,
                          size_t dim6)
{
  dims_[0] = dim0;
  dims_[1] = dim1;
  dims_[2] = dim2;
  dims_[3] = dim3;
  dims_[4] = dim4;
  dims_[5] = dim5;
  dims_[6] = dim6;
  order_ = 7;
  length_ = dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6;
  array_ = array;
}

//overload () operator

//1D
template <typename T>
inline T& ViewTensor<T>::operator()(size_t i) const
{
  assert(order_ == 1 && "Tensor order (rank) does not match constructor in ViewTensor 1D!");
  assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewTensor 1D!");
    
  return array_[i];
}

/*
//specification for Tensor type
//1D
template <typename T>
inline T& ViewTensor<Tensor<T>>::operator()(size_t i) const
{
assert(i < dim1_ && "i is out of bounds in c_array 1D");  // die if >= dim1
    
return (*this_array_)(i);
}
*/

//2D
template <typename T>
inline T& ViewTensor<T>::operator()(size_t i,
                                    size_t j) const
{
   
  assert(order_ == 2 && "Tensor order (rank) does not match constructor in ViewTensor 2D!");
  assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewTensor 2D!");
  assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewTensor 2D!");
    
  return array_[j + (i *  dims_[1])];
}

//3D
template <typename T>
inline T& ViewTensor<T>::operator()(size_t i,
                                    size_t j,
                                    size_t k) const
{
  assert(order_ == 3 && "Tensor order (rank) does not match constructor in ViewTensor 3D!");
  assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewTensor 3D!");
  assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewCarray 3D!");
  assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewTensor 3D!");
    
  return array_[k + (j * dims_[2])
                + (i * dims_[2] *  dims_[1])];
}

//4D
template <typename T>
inline T& ViewTensor<T>::operator()(size_t i,
                                    size_t j,
                                    size_t k,
                                    size_t l) const
{
  assert(order_ == 4 && "Tensor order (rank) does not match constructor in ViewTensor 4D!");
  assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewTensor 4D");  // die if >= dim0
  assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewTensor 4D");  // die if >= dim1
  assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewTensor 4D");  // die if >= dim2
  assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewTensor 4D");  // die if >= dim3
    
  return array_[l + (k * dims_[3])
                + (j * dims_[3] * dims_[2])
                + (i * dims_[3] * dims_[2] *  dims_[1])];
}

//5D
template <typename T>
inline T& ViewTensor<T>::operator()(size_t i,
                                    size_t j,
                                    size_t k,
                                    size_t l,
                                    size_t m) const
{
  assert(order_ == 5 && "Tensor order (rank) does not match constructor in ViewTensor 5D!");
  assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewTensor 5D!");
  assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewTensor 5D!");
  assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewTensor 5D!");
  assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewTensor 5D!");
  assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewTensor 5D!");
    
  return array_[m + (l * dims_[4])
                + (k * dims_[4] * dims_[3])
                + (j * dims_[4] * dims_[3] * dims_[2])
                + (i * dims_[4] * dims_[3] * dims_[2] *  dims_[1])];
}

//6D
template <typename T>
inline T& ViewTensor<T>::operator()(size_t i,
                                    size_t j,
                                    size_t k,
                                    size_t l,
                                    size_t m,
                                    size_t n) const
{
  assert(order_ == 6 && "Tensor order (rank) does not match constructor in ViewTensor 6D!");
  assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewTensor 6D!");
  assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewTensor 6D!");
  assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewTensor 6D!");
  assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewTensor 6D!");
  assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewTensor 6D!");
  assert(n >= 0 && n < dims_[5] && "n is out of bounds in ViewTensor 6D!");
    
  return array_[n + (m * dims_[5])
                + (l * dims_[5] * dims_[4])
                + (k * dims_[5] * dims_[4] * dims_[3])
                + (j * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                + (i * dims_[5] * dims_[4] * dims_[3] * dims_[2] *  dims_[1])];
}

//7D
template <typename T>
inline T& ViewTensor<T>::operator()(size_t i,
                                    size_t j,
                                    size_t k,
                                    size_t l,
                                    size_t m,
                                    size_t n,
                                    size_t o) const
{
  assert(order_ == 7 && "Tensor order (rank) does not match constructor in ViewTensor 7D!");
  assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewTensor 7D!");
  assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewTensor 7D!");
  assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewTensor 7D!");
  assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewTensor 7D!");
  assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewTensor 7D!");
  assert(n >= 0 && n < dims_[5] && "n is out of bounds in ViewTensor 7D!");
  assert(o >= 0 && o < dims_[6] && "o is out of bounds in ViewTensor 7D!");
    
  return array_[o + (n * dims_[6])
                + (m * dims_[6] * dims_[5])
                + (l * dims_[6] * dims_[5] * dims_[4])
                + (k * dims_[6] * dims_[5] * dims_[4] * dims_[3])
                + (j * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                + (i * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2] *  dims_[1])];
}


// calculate this ViewFArray object = math(A,B)
template <typename T>
template <typename M>
void ViewTensor<T>::operator=(M do_this_math){
  do_this_math(*this); // pass in this ViewFArray object
}// end of math opperation

//return size
template <typename T>
inline size_t ViewTensor<T>::size() const {
  return length_;
}

template <typename T>
inline size_t ViewTensor<T>::dims(size_t i) const {
  assert(i < order_ && "ViewTensor order (rank) does not match constructor, dim[i] does not exist!");
  assert(i >= 0 && dims_[i]>0 && "Access to ViewTensor dims is out of bounds!");
  return dims_[i];
}

template <typename T>
inline size_t ViewTensor<T>::order() const {
  return order_;
}

template <typename T>
inline T* ViewTensor<T>::pointer() const {
  return array_;
}

//---end of ViewTensor class definitions----

} // end namespace
