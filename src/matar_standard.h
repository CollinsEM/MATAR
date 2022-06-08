/**********************************************************************************************
 © 2020. Triad National Security, LLC. All rights reserved.
 This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
 National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
 Department of Energy/National Nuclear Security Administration. All rights in the program are
 reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
 Security Administration. The Government is granted for itself and others acting on its behalf a
 nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
 derivative works, distribute copies to the public, perform publicly and display publicly, and
 to permit others to do so.
 This program is open source under the BSD-3 License.
 Redistribution and use in source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:
 
 1.  Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.
 
 2.  Redistributions in binary form must reproduce the above copyright notice, this list of
 conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.
 
 3.  Neither the name of the copyright holder nor the names of its contributors may be used
 to endorse or promote products derived from this software without specific prior
 written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************/


// MATAR Standard (non-Kokkos) Data Structures
//  1. FArray
//  2. ViewFArray
//  3. FMatrix
//  4. ViewFMatrix
//  5. CArray
//  6. ViewCArray
//  7. CMatrix
//  8. ViewCMatrix
//  9. RaggedRightArray
//  10. RaggedDownArray
//  11. DynamicRaggedRightArray
//  12. DynamicRaggedDownArray
//  13. SparseRowArray
//  14. SparseColArray


//1. FArray
// indicies are [0:N-1]
template <typename T>
class FArray {
    
private:
    size_t dims_[7];
    size_t length_;
    size_t order_;  // tensor order (rank)
    std::shared_ptr <T []> array_;
    
public:
    
    // default constructor
   FArray ();
   
    //overload constructors from 1D to 7D
     
   FArray(size_t dim0);
    
   FArray(size_t dim0,
          size_t dim1);
    
   FArray(size_t dim0,
          size_t dim1,
          size_t dim2);
    
   FArray(size_t dim0,
          size_t dim1,
          size_t dim2,
          size_t dim3);
    
   FArray(size_t dim0,
          size_t dim1,
          size_t dim2,
          size_t dim3,
          size_t dim4);

   FArray(size_t dim0,
          size_t dim1,
          size_t dim2,
          size_t dim3,
          size_t dim4,
          size_t dim5);

   FArray(size_t dim0,
          size_t dim1,
          size_t dim2,
          size_t dim3,
          size_t dim4,
          size_t dim5,
          size_t dim6);

    FArray (const FArray& temp);
    
    // overload operator() to access data as array(i,....,n);
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
    
    //overload = operator
    FArray& operator=(const FArray& temp);
    
    //return array size
    size_t size() const;

    // return array dims
    size_t dims(size_t i) const;
    
    // return array order (rank)
    size_t order() const;
    
    //return pointer
    T* pointer() const;
    
    // deconstructor
    ~FArray ();
    
}; // end of f_array_t

//---FArray class definnitions----

//constructors
template <typename T>
FArray<T>::FArray(){
    array_ = NULL;
    length_ = 0;
}

//1D
template <typename T>
FArray<T>::FArray(size_t dim0)
{
    dims_[0] = dim0;
    length_ = dim0;
    order_ = 1;
    array_ = std::shared_ptr <T []> (new T[length_]);
}

template <typename T>
FArray<T>::FArray(size_t dim0,
                  size_t dim1)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    order_ = 2;
    length_ = dim0*dim1;
    array_ = std::shared_ptr <T []> (new T[length_]);
}

//3D
template <typename T>
FArray<T>::FArray(size_t dim0,
                  size_t dim1,
                  size_t dim2)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = dim0*dim1*dim2;
    array_ = std::shared_ptr <T []> (new T[length_]);
}

//4D
template <typename T>
FArray<T>::FArray(size_t dim0,
                  size_t dim1,
                  size_t dim2,
                  size_t dim3)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    order_ = 4;
    length_ = dim0*dim1*dim2*dim3;
    array_ = std::shared_ptr <T []> (new T[length_]);
}

//5D
template <typename T>
FArray<T>::FArray(size_t dim0,
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
    length_ = dim0*dim1*dim2*dim3*dim4;
    array_ = std::shared_ptr <T []> (new T[length_]);
}

//6D
template <typename T>
FArray<T>::FArray(size_t dim0,
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
    length_ = dim0*dim1*dim2*dim3*dim4*dim5;
    array_ = std::shared_ptr <T []> (new T[length_]);
}


//7D
template <typename T>
FArray<T>::FArray(size_t dim0,
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
    length_ = dim0*dim1*dim2*dim3*dim4*dim5*dim6;
    array_ = std::shared_ptr <T []> (new T[length_]);
        
}

//Copy constructor

template <typename T>
FArray<T>::FArray(const FArray& temp) {
    
    // Do nothing if the assignment is of the form x = x
    
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for
        
        order_  = temp.order_;
        length_ = temp.length_;       
        array_ = temp.array_;
    } // end if
    
} // end constructor

//overload operator () for 1D to 7D
//indices are from [0:N-1]

//1D
template <typename T>
T& FArray<T>::operator()(size_t i) const
{
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in FArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 1D!");
    return array_[i];
}

//2D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j) const
{
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in FArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 2D!");
    return array_[i + j*dims_[0]];
}

//3D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k) const
{
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in FArray 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in Farray 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 3D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]];
}

//4D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k,
                         size_t l) const
{
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in FArray 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in FArray 4D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]];
}

//5D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k,
                         size_t l,
                         size_t m) const
{
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in FArray 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in FArray 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in FArray 5D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]];
}

//6D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k,
                         size_t l,
                         size_t m,
                         size_t n) const
{
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in FArray 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in FArray 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in FArray 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in FArray 6D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]
                    + n*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]];
}

//7D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k,
                         size_t l,
                         size_t m,
                         size_t n,
                         size_t o) const
{
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in FArray 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in FArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in FArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in FArray 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in FArray 7D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]
                    + n*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]
                    + o*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]*dims_[5]];
}
    
// = operator
//THIS = FArray <> TEMP(n,m,...)
template <typename T>
FArray<T>& FArray<T>::operator= (const FArray& temp)
{
    if(this != & temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_  = temp.order_;
        length_ = temp.length_;
        array_  = temp.array_;
    }
    return *this;
}

template <typename T>
inline size_t FArray<T>::size() const {
    return length_;
}

template <typename T>
inline size_t FArray<T>::dims(size_t i) const {
    assert(i < order_ && "FArray order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to FArray dims is out of bounds!");
    return dims_[i];
}

template <typename T>
inline size_t FArray<T>::order() const {
    return order_;
}


template <typename T>
inline T* FArray<T>::pointer() const {
    return array_.get();
}

//delete FArray
template <typename T>
FArray<T>::~FArray(){}

//---end of FArray class definitions----


//2. ViewFArray
// indicies are [0:N-1]
template <typename T>
class ViewFArray {

private:
    size_t dims_[7];
    size_t length_; // Length of 1D array
    size_t order_;  // tensor order (rank)
    T * array_;
    
public:
    
    // default constructor
    ViewFArray ();

    //---1D to 7D array ---
    ViewFArray(T *array,
               size_t dim0);
    
    ViewFArray (T *array,
                size_t dim0,
                size_t dim1);

    ViewFArray (T *array,
                size_t dim0,
                size_t dim1,
                size_t dim2);

    ViewFArray (T *array,
                size_t dim0,
                size_t dim1,
                size_t dim2,
                size_t dim3);
    
    ViewFArray (T *array,
                size_t dim0,
                size_t dim1,
                size_t dim2,
                size_t dim3,
                size_t dim4);

    ViewFArray (T *array,
                size_t dim0,
                size_t dim1,
                size_t dim2,
                size_t dim3,
                size_t dim4,
                size_t dim5);
    
    ViewFArray (T *array,
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
    
    //return array dims
    size_t dims(size_t i) const;
    
    // return array order (rank)
    size_t order() const;

    // return pointer
    T* pointer() const;
    
}; // end of viewFArray

//class definitions for viewFArray

//~~~~constructors for viewFArray for 1D to 7D~~~~~~~

//no dimension
template <typename T>
ViewFArray<T>::ViewFArray(){
  array_ = NULL;
  length_ = 0;
}

//1D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0)
{
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    array_  = array;
}

//2D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0,
                          size_t dim1)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    order_ = 2;
    length_ = dim0*dim1;
    array_  = array;
}

//3D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = dim0*dim1*dim2;
    array_  = array;
}

//4D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
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
    length_ = dim0*dim1*dim2*dim3;
    array_  = array;
}

//5D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
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
    length_ = dim0*dim1*dim2*dim3*dim4;
    array_  = array;
}

//6D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
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
    length_ = dim0*dim1*dim2*dim3*dim4*dim5;
    array_  = array;
}

//7D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
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
    length_ = dim0*dim1*dim2*dim3*dim4*dim5*dim6;
    array_  = array;
}

//~~~~~~operator () overload 
//for dimensions 1D to 7D
//indices for array are from 0...N-1

//1D
template <typename T>
T& ViewFArray<T>::operator()(size_t i) const
{
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in ViewFArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 1D!");
    return array_[i];
}

//2D
template <typename T>
T& ViewFArray<T>::operator()(size_t i,
                             size_t j) const
{
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in ViewFArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 2D!");
    return array_[i + j*dims_[0]];
}

//3D
template <typename T>
T& ViewFArray<T>::operator()(size_t i,
                             size_t j,
                             size_t k) const
{
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in ViewFArray 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 3D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]];
}

//4D
template <typename T>
T& ViewFArray<T>::operator()(size_t i,
                             size_t j,
                             size_t k,
                             size_t l) const
{
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in ViewFArray 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewFArray 4D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]];
}

//5D
template <typename T>
T& ViewFArray<T>::operator()(size_t i,
                             size_t j,
                             size_t k,
                             size_t l,
                             size_t m) const
{
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in ViewFArray 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewFArray 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewFArray 5D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]];
}

//6D
template <typename T>
T& ViewFArray<T>:: operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m,
                              size_t n) const
{
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in ViewFArray 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewFArray 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewFArray 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in ViewFArray 6D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]
                    + n*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]];
}

//7D
template <typename T>
T& ViewFArray<T>:: operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m,
                              size_t n,
                              size_t o) const
{
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in ViewFArray 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewFArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewFArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in ViewFArray 7D!");
    assert(o >= 0 && o < dims_[6] && "n is out of bounds in ViewFArray 7D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]
                    + n*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]
                    + o*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]*dims_[5]];
}

// calculate this ViewFArray object = math(A,B)
template <typename T>
template <typename M>
void ViewFArray<T>::operator=(M do_this_math){
    do_this_math(*this); // pass in this ViewFArray object
}// end of math opperation

template <typename T>
inline size_t ViewFArray<T>::dims(size_t i) const {
    assert(i < order_ && "ViewFArray order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to ViewFArray dims is out of bounds!");
    return dims_[i];
}

template <typename T>
inline size_t ViewFArray<T>::order() const {
    return order_;
}

template <typename T>
inline size_t ViewFArray<T>::size() const {
    return length_;
}

template <typename T>
inline T* ViewFArray<T>::pointer() const {
    return array_;
}

//---end of ViewFArray class definitions---


//3. FMatrix
// indicies are [1:N]
template <typename T>
class FMatrix {
private:
    size_t dims_[7];
    size_t length_; // Length of 1D array
    size_t order_;  // tensor order (rank)
    std::shared_ptr <T []> matrix_;

public:
    // Default constructor
    FMatrix ();

    //---1D to 7D matrix ---
    FMatrix (size_t dim1);

    FMatrix (size_t dim1,
             size_t dim2);

    FMatrix (size_t dim1,
             size_t dim2,
             size_t dim3);

    FMatrix (size_t dim1,
             size_t dim2,
             size_t dim3,
             size_t dim4);

    FMatrix (size_t dim1,
             size_t dim2,
             size_t dim3,
             size_t dim4,
             size_t dim5);

    FMatrix (size_t dim1,
             size_t dim2,
             size_t dim3,
             size_t dim4,
             size_t dim5,
             size_t dim6);

    FMatrix (size_t dim1,
             size_t dim2,
             size_t dim3,
             size_t dim4,
             size_t dim5,
             size_t dim6,
             size_t dim7);
    
    FMatrix (const FMatrix& temp);
    
    T& operator() (size_t i) const;
    
    T& operator() (size_t i,
                   size_t j) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n) const;

    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n,
                   size_t o) const;
    
    
    // Overload copy assignment operator
    FMatrix& operator=(const FMatrix& temp);

    // the length of the 1D storage array
    size_t size() const;

    // matrix dims
    size_t dims(size_t i) const;
    
    // return matrix order (rank)
    size_t order() const;
    
    //return pointer
    T* pointer() const;

    // Deconstructor
    ~FMatrix ();

}; // End of FMatrix

//---FMatrix class definitions---

//constructors
template <typename T>
FMatrix<T>::FMatrix(){
    matrix_ = NULL;
    length_ = 0;
}

//1D
template <typename T>
FMatrix<T>::FMatrix(size_t dim1)
{
    dims_[0] = dim1;
    order_ = 1;
    length_ = dim1;
    matrix_ = std::shared_ptr <T []> (new T[length_]);
}

//2D
template <typename T>
FMatrix<T>::FMatrix(size_t dim1,
                    size_t dim2)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    order_ = 2;
    length_ = dim1 * dim2;
    matrix_ = std::shared_ptr <T []> (new T[length_]);
}

//3D
template <typename T>
FMatrix<T>::FMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    order_ = 3;
    length_ = dim1 * dim2 * dim3;
    matrix_ = std::shared_ptr <T []> (new T[length_]);
}

//4D
template <typename T>
FMatrix<T>::FMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3,
                    size_t dim4)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    order_ = 4;
    length_ = dim1 * dim2 * dim3 * dim4;
    matrix_ = std::shared_ptr <T []> (new T[length_]);
}

//5D
template <typename T>
FMatrix<T>::FMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3,
                    size_t dim4,
                    size_t dim5)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    order_ = 5;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5;
    matrix_ = std::shared_ptr <T []> (new T[length_]);
}

//6D
template <typename T>
FMatrix<T>::FMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3,
                    size_t dim4,
                    size_t dim5,
                    size_t dim6)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    dims_[5] = dim6;
    order_ = 6;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5 * dim6;
    matrix_ = std::shared_ptr <T []> (new T[length_]);

}

template <typename T>
FMatrix<T>::FMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3,
                    size_t dim4,
                    size_t dim5,
                    size_t dim6,
                    size_t dim7)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    dims_[5] = dim6;
    dims_[6] = dim7;
    order_ = 7;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * dim7;
    matrix_ = std::shared_ptr <T []> (new T[length_]);
    
}

template <typename T>
FMatrix<T>::FMatrix(const FMatrix& temp) {
    
    // Do nothing if the assignment is of the form x = x
    
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for
        
        order_  = temp.order_;
        length_ = temp.length_;
        matrix_ = temp.matrix_;
    } // end if
    
} // end constructor


//overload operators

//1D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i) const
{
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in FMatrix 1D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in FMatrix 1D!");
    return matrix_[i - 1];
}

//2D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j) const
{
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in FMatrix 2D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in FMatrix 2D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in FMatrix 2D!");
    return matrix_[(i - 1) + ((j - 1) * dims_[0])];
}

//3D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k) const
{
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in FMatrix 3D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in FMatrix 3D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in FMatrix 3D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in FMatrix 3D!");
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])];
}

//4D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k,
                                  size_t l) const
{
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in FMatrix 4D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in FMatrix 4D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in FMatrix 4D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in FMatrix 4D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in FMatrix 4D!");
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])
                           + ((l - 1) * dims_[0] * dims_[1] * dims_[2])];
}

//5D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k,
                                  size_t l,
                                  size_t m) const
{
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in FMatrix 5D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in FMatrix 5D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in FMatrix 5D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in FMatrix 5D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in FMatrix 5D!");
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in FMatrix 5D!");
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])
                           + ((l - 1) * dims_[0] * dims_[1] * dims_[2])
                           + ((m - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3])];
}

//6D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k,
                                  size_t l,
                                  size_t m,
                                  size_t n) const
{
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in FMatrix 6D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in FMatrix 6D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in FMatrix 6D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in FMatrix 6D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in FMatrix 6D!");
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in FMatrix 6D!");
    assert(n >= 1 && n <= dims_[5] && "n is out of bounds in FMatrix 6D!");
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])
                           + ((l - 1) * dims_[0] * dims_[1] * dims_[2])
                           + ((m - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3])
                           + ((n - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4])];
}

//7D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k,
                                  size_t l,
                                  size_t m,
                                  size_t n,
                                  size_t o) const
{
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in FMatrix 7D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in FMatrix 7D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in FMatrix 7D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in FMatrix 7D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in FMatrix 7D!");
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in FMatrix 7D!");
    assert(n >= 1 && n <= dims_[5] && "n is out of bounds in FMatrix 7D!");
    assert(o >= 1 && o <= dims_[6] && "o is out of bounds in FMatrix 7D!");
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])
                           + ((l - 1) * dims_[0] * dims_[1] * dims_[2])
                           + ((m - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3])
                           + ((n - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4])
                           + ((o - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5])];
}


template <typename T>
inline FMatrix<T>& FMatrix<T>::operator= (const FMatrix& temp)
{
    // Do nothing if assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_  = temp.order_;
        length_ = temp.length_;
	matrix_ = temp.matrix_;
    }
    
    return *this;
}

template <typename T>
inline size_t FMatrix<T>::size() const {
    return length_;
}

template <typename T>
inline size_t FMatrix<T>::dims(size_t i) const {
    i--; // i starts at 1
    assert(i < order_ && "FMatrix order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to FMatrix dims is out of bounds!");
    return dims_[i];
}

template <typename T>
inline size_t FMatrix<T>::order() const {
    return order_;
}

template <typename T>
inline T* FMatrix<T>::pointer() const{
    return matrix_.get();
}

template <typename T>
FMatrix<T>::~FMatrix() {}

//----end of FMatrix class definitions----


//4. ViewFMatrix
//  indices are [1:N]
template <typename T>
class ViewFMatrix {

private:
    size_t dims_[7];
    size_t length_; // Length of 1D array
    size_t order_;  // tensor order (rank)
    T * matrix_;
    
public:
    
    // Default constructor
    ViewFMatrix ();
    
    //--- 1D to 7D matrix ---

    ViewFMatrix(T *matrix,
                size_t dim1);
    
    ViewFMatrix(T *some_matrix,
                size_t dim1,
                size_t dim2);
    
    ViewFMatrix(T *matrix,
                size_t dim1,
                size_t dim2,
                size_t dim3);
    
    ViewFMatrix(T *matrix,
                size_t dim1,
                size_t dim2,
                size_t dim3,
                size_t dim4);
    
    ViewFMatrix (T *matrix,
                 size_t dim1,
                 size_t dim2,
                 size_t dim3,
                 size_t dim4,
                 size_t dim5);
    
    ViewFMatrix (T *matrix,
                 size_t dim1,
                 size_t dim2,
                 size_t dim3,
                 size_t dim4,
                 size_t dim5,
                 size_t dim6);
    
    ViewFMatrix (T *matrix,
                 size_t dim1,
                 size_t dim2,
                 size_t dim3,
                 size_t dim4,
                 size_t dim5,
                 size_t dim6,
                 size_t dim7);
    
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
    
    T& operator() (size_t i,
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
    
    // length of 1D array
    size_t size() const;
    
    // matrix dims
    size_t dims(size_t i) const;
    
    // return matrix order (rank)
    size_t order() const;

    // return pointer
    T* pointer() const;
    
}; // end of ViewFMatrix

//constructors

//no dimension
template <typename T>
ViewFMatrix<T>::ViewFMatrix() {
  matrix_ = NULL;
  length_ = 0;
}

//1D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *matrix,
                            size_t dim1)
{
    dims_[0] = dim1;
    order_ = 1;
    length_ = dim1;
    matrix_ = matrix;
}

//2D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    order_ = 2;
    length_ = dim1 * dim2;
    matrix_ = matrix;
}

//3D
template <typename T>
ViewFMatrix<T>::ViewFMatrix (T *matrix,
                             size_t dim1,
                             size_t dim2,
                             size_t dim3)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    order_ = 3;
    length_ = dim1 * dim2 * dim3;
    matrix_ = matrix;
}

//4D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2,
                            size_t dim3,
                            size_t dim4)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    order_ = 4;
    length_ = dim1 * dim2 * dim3 * dim4;
    matrix_ = matrix;
}

//5D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2,
                            size_t dim3,
                            size_t dim4,
                            size_t dim5)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    order_ = 5;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5;
    matrix_ = matrix;
}

//6D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2,
                            size_t dim3,
                            size_t dim4,
                            size_t dim5,
                            size_t dim6)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    dims_[5] = dim6;
    order_ = 6;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5 * dim6;
    matrix_ = matrix;
}

//6D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2,
                            size_t dim3,
                            size_t dim4,
                            size_t dim5,
                            size_t dim6,
                            size_t dim7)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    dims_[5] = dim6;
    dims_[6] = dim7;
    order_ = 7;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * dim7;
    matrix_ = matrix;
}


//overload operator ()

//1D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i) const
{
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in ViewFMatrix 1D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewFMatrix 1D");  // die if >= dim1
        
    return matrix_[(i - 1)];
}

//2D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i, 
                                     size_t j) const
{
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in ViewFMatrix 2D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewFMatrix 2D");  // die if >= dim1
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewFMatrix 2D");  // die if >= dim2
        
    return matrix_[(i - 1) + ((j - 1) * dims_[0])];
}

//3D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i, 
                                     size_t j, 
                                     size_t k) const
{
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in ViewFMatrix 3D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewFMatrix 3D");  // die if >= dim1
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewFMatrix 3D");  // die if >= dim2
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewFMatrix 3D");  // die if >= dim3
        
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])];
}

//4D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i, 
                                     size_t j, 
                                     size_t k, 
                                     size_t l) const
{
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in ViewFMatrix 4D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewFMatrix 4D");  // die if >= dim1
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewFMatrix 4D");  // die if >= dim2
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewFMatrix 4D");  // die if >= dim3
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in ViewFMatrix 4D");  // die if >= dim4
        
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])
                           + ((l - 1) * dims_[0] * dims_[1] * dims_[2])];
}

//5D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i, 
                                     size_t j, 
                                     size_t k, 
                                     size_t l, 
                                     size_t m) const
{
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in ViewFMatrix 5D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewFMatrix 5D");  // die if >= dim1
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewFMatrix 5D");  // die if >= dim2
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewFMatrix 5D");  // die if >= dim3
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in ViewFMatrix 5D");  // die if >= dim4
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in ViewFMatrix 5D");  // die if >= dim5
       
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])
                           + ((l - 1) * dims_[0] * dims_[1] * dims_[2])
                           + ((m - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3])];
}

//6D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i,
                                     size_t j,
                                     size_t k,
                                     size_t l,
                                     size_t m,
                                     size_t n) const
{
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in ViewFMatrix 6D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewFMatrix 6D");  // die if >= dim1
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewFMatrix 6D");  // die if >= dim2
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewFMatrix 6D");  // die if >= dim3
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in ViewFMatrix 6D");  // die if >= dim4
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in ViewFMatrix 6D");  // die if >= dim5
    assert(n >= 1 && n <= dims_[5] && "n is out of bounds in ViewFMatrix 6D");  // die if >= dim6
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])
                           + ((l - 1) * dims_[0] * dims_[1] * dims_[2])
                           + ((m - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3])
                           + ((n - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4])];
}

//6D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i,
                                     size_t j,
                                     size_t k,
                                     size_t l,
                                     size_t m,
                                     size_t n,
                                     size_t o) const
{
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in ViewFMatrix 7D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewFMatrix 7D");  // die if >= dim1
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewFMatrix 7D");  // die if >= dim2
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewFMatrix 7D");  // die if >= dim3
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in ViewFMatrix 7D");  // die if >= dim4
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in ViewFMatrix 7D");  // die if >= dim5
    assert(n >= 1 && n <= dims_[5] && "n is out of bounds in ViewFMatrix 7D");  // die if >= dim6
    assert(o >= 1 && o <= dims_[6] && "o is out of bounds in ViewFMatrix 7D");  // die if >= dim7
    
    return matrix_[(i - 1) + ((j - 1) * dims_[0])
                           + ((k - 1) * dims_[0] * dims_[1])
                           + ((l - 1) * dims_[0] * dims_[1] * dims_[2])
                           + ((m - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3])
                           + ((n - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4])
                           + ((o - 1) * dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5])];
}

// calculate this ViewFMatrix object = math(A,B)
template <typename T>
template <typename M>
void ViewFMatrix<T>::operator=(M do_this_math){
    do_this_math(*this); // pass in this ViewFArray object
}// end of math opperation

template <typename T>
inline size_t ViewFMatrix<T>::dims(size_t i) const {
    i--; // i starts at 1
    assert(i < order_ && "ViewFMatrix order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to ViewFMatrix dims is out of bounds!");
    return dims_[i];
}

template <typename T>
inline size_t ViewFMatrix<T>::order() const {
    return order_;
}

template <typename T>
inline T* ViewFMatrix<T>::pointer() const {
    return matrix_;
}
//-----end ViewFMatrix-----


//5. CArray
// indicies are [0:N-1]
template <typename T>
class CArray {
    
private:
    size_t dims_[7];
    size_t length_; // Length of 1D array
    size_t order_;  // tensor order (rank)
    std::shared_ptr <T []> array_;

public:
    // Default constructor
    CArray ();

    // --- 1D to 7D array ---
    
    CArray (size_t dim0);

    CArray (size_t dim0,
            size_t dim1);

    CArray (size_t dim0,
            size_t dim1,
            size_t dim2);

    CArray (size_t dim0,
            size_t dim1,
            size_t dim2,
            size_t dim3);

    CArray (size_t dim0,
            size_t dim1,
            size_t dim2,
            size_t dim3,
            size_t dim4);

    CArray (size_t dim0,
            size_t dim1,
            size_t dim2,
            size_t dim3,
            size_t dim4,
            size_t dim5);

    CArray (size_t dim0,
            size_t dim1,
            size_t dim2,
            size_t dim3,
            size_t dim4,
            size_t dim5,
            size_t dim6);
    
    CArray (const CArray& temp);
    
    // Overload operator()
    T& operator() (size_t i) const;
    
    T& operator() (size_t i,
                   size_t j) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n,
                   size_t o) const;
    
    // Overload copy assignment operator
    CArray& operator= (const CArray& temp); 

     //return array size
    size_t size() const;

    // return array dims
    size_t dims(size_t i) const;
    
    // return array order (rank)
    size_t order() const;
    
    //return pointer
    T* pointer() const;

    // Deconstructor
    ~CArray ();

}; // End of CArray

//---carray class declarations---

//constructors

//no dim
template <typename T>
CArray<T>::CArray() {
    array_ = NULL;
    length_ = order_ = 0;
}

//1D
template <typename T>
CArray<T>::CArray(size_t dim0)
{
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    array_ = std::shared_ptr <T[]> (new T[length_]);
}

//2D
template <typename T>
CArray<T>::CArray(size_t dim0,
                  size_t dim1)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    order_ = 2;
    length_ = dim0 * dim1;
    array_ = std::shared_ptr <T[]> (new T[length_]);
}

//3D
template <typename T>
CArray<T>::CArray(size_t dim0,
                  size_t dim1,
                  size_t dim2)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = dim0 * dim1 * dim2;
    array_ = std::shared_ptr <T[]> (new T[length_]);
}

//4D
template <typename T>
CArray<T>::CArray(size_t dim0,
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
    array_ = std::shared_ptr <T[]> (new T[length_]);
}

//5D
template <typename T>
CArray<T>::CArray(size_t dim0,
                  size_t dim1,
                  size_t dim2,
                  size_t dim3,
                  size_t dim4) {
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    order_ = 5;
    length_ = dim0 * dim1 * dim2 * dim3 * dim4;
    array_ = std::shared_ptr <T[]> (new T[length_]);
}

//6D
template <typename T>
CArray<T>::CArray(size_t dim0,
                  size_t dim1,
                  size_t dim2,
                  size_t dim3,
                  size_t dim4,
                  size_t dim5) {
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    order_ = 6;
    length_ = dim0 * dim1 * dim2 * dim3 * dim4 * dim5;
    array_ = std::shared_ptr <T[]> (new T[length_]);
}

//7D
template <typename T>
CArray<T>::CArray(size_t dim0,
                  size_t dim1,
                  size_t dim2,
                  size_t dim3,
                  size_t dim4,
                  size_t dim5,
                  size_t dim6) {
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    dims_[6] = dim6;
    order_ = 7;
    length_ = dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6;
    array_ = std::shared_ptr <T[]> (new T[length_]);
}

//Copy constructor

template <typename T>
CArray<T>::CArray(const CArray& temp) {
    
    // Do nothing if the assignment is of the form x = x
    
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for
        
        order_  = temp.order_;
        length_ = temp.length_;
        array_ = temp.array_;
    } // end if
    
} // end constructor


//overload () operator

//1D
template <typename T>
inline T& CArray<T>::operator() (size_t i) const
{
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in CArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in CArray 1D!");

    return array_[i];
}

//2D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j) const
{
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in CArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in CArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in CArray 2D!");
    
    return array_[j + (i *  dims_[1])];
}

//3D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k) const
{
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in CArray 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in CArray 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in Carray 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in CArray 3D!");
    
    return array_[k + (j * dims_[2])
                    + (i * dims_[2] *  dims_[1])];
}

//4D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k,
                                 size_t l) const
{
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in CArray 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in CArray 4D");  // die if >= dim0
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in CArray 4D");  // die if >= dim1
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in CArray 4D");  // die if >= dim2
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in CArray 4D");  // die if >= dim3

    return array_[l + (k * dims_[3])
                    + (j * dims_[3] * dims_[2])
                    + (i * dims_[3] * dims_[2] *  dims_[1])];
}

//5D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k,
                                 size_t l,
                                 size_t m) const
{
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in CArray 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in CArray 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in CArray 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in CArray 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in CArray 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in CArray 5D!");
    
    return array_[m + (l * dims_[4])
                    + (k * dims_[4] * dims_[3])
                    + (j * dims_[4] * dims_[3] * dims_[2])
                    + (i * dims_[4] * dims_[3] * dims_[2] *  dims_[1])];
}

//6D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k,
                                 size_t l,
                                 size_t m,
                                 size_t n) const
{
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in CArray 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in CArray 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in CArray 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in CArray 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in CArray 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in CArray 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in CArray 6D!");
    
    return array_[n + (m * dims_[5])
                    + (l * dims_[5] * dims_[4])
                    + (k * dims_[5] * dims_[4] * dims_[3])
                    + (j * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                    + (i * dims_[5] * dims_[4] * dims_[3] * dims_[2] *  dims_[1])];
}

//7D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k,
                                 size_t l,
                                 size_t m,
                                 size_t n,
                                 size_t o) const
{
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in CArray 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in CArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in CArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in CArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in CArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in CArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in CArray 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in CArray 7D!");
    
    return array_[o + (n * dims_[6])
                    + (m * dims_[6] * dims_[5])
                    + (l * dims_[6] * dims_[5] * dims_[4])
                    + (k * dims_[6] * dims_[5] * dims_[4] * dims_[3])
                    + (j * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                    + (i * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2] *  dims_[1])];
    
}


//overload = operator
template <typename T>
inline CArray<T>& CArray<T>::operator= (const CArray& temp)
{
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_  = temp.order_;
        length_ = temp.length_;
        array_  = temp.array_;
    }
    return *this;
}



//return size
template <typename T>
inline size_t CArray<T>::size() const {
    return length_;
}

template <typename T>
inline size_t CArray<T>::dims(size_t i) const {
    assert(i < order_ && "CArray order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to CArray dims is out of bounds!");
    return dims_[i];
}

template <typename T>
inline size_t CArray<T>::order() const {
    return order_;
}


template <typename T>
inline T* CArray<T>::pointer() const{
    return array_.get();
}

//destructor
template <typename T>
CArray<T>::~CArray() {}

//----endof carray class definitions----


//6. ViewCArray
// indicies are [0:N-1]
template <typename T>
class ViewCArray {

private:
    size_t dims_[7];
    size_t length_; // Length of 1D array
    size_t order_;  // tensor order (rank)
    T * array_;
    
public:
    
    // Default constructor
    ViewCArray ();
    
    //--- 1D to 7D array ---
    ViewCArray(T *array,
               size_t dim0);

    ViewCArray(T *array,
               size_t dim0,
               size_t dim1);
    
    ViewCArray(T *some_array,
               size_t dim0,
               size_t dim1,
               size_t dim2);
    
    ViewCArray(T *some_array,
               size_t dim0,
               size_t dim1,
               size_t dim2,
               size_t dim3);
    
    ViewCArray (T *some_array,
                size_t dim0,
                size_t dim1,
                size_t dim2,
                size_t dim3,
                size_t dim4);

    ViewCArray (T *some_array,
                size_t dim0,
                size_t dim1,
                size_t dim2,
                size_t dim3,
                size_t dim4,
                size_t dim5);
 
    ViewCArray (T *some_array,
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
    
}; // end of ViewCArray

//class definitions

//constructors

//no dim
template <typename T>
ViewCArray<T>::ViewCArray() {
  array_ = NULL;
  length_ = order_ = 0;
}

//1D
template <typename T>
ViewCArray<T>::ViewCArray(T *array,
                          size_t dim0)
{
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    array_ = array;
}

//2D
template <typename T>
ViewCArray<T>::ViewCArray(T *array,
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
ViewCArray<T>::ViewCArray (T *array,
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
ViewCArray<T>::ViewCArray(T *array,
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
ViewCArray<T>::ViewCArray(T *array,
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
ViewCArray<T>::ViewCArray(T *array,
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
ViewCArray<T>::ViewCArray(T *array,
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
inline T& ViewCArray<T>::operator()(size_t i) const
{
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in ViewCArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewCArray 1D!");
    
    return array_[i];
}

/*
//specification for CArray type
//1D
template <typename T>
inline T& ViewCArray<CArray<T>>::operator()(size_t i) const
{
    assert(i < dim1_ && "i is out of bounds in c_array 1D");  // die if >= dim1
    
    return (*this_array_)(i);
}
*/

//2D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i, 
                                    size_t j) const
{
   
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in ViewCArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewCArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewCArray 2D!");
    
    return array_[j + (i *  dims_[1])];
}

//3D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i, 
                                    size_t j, 
                                    size_t k) const
{
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in ViewCArray 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewCArray 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewCarray 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewCArray 3D!");
    
    return array_[k + (j * dims_[2])
                    + (i * dims_[2] *  dims_[1])];
}

//4D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i, 
                                    size_t j, 
                                    size_t k, 
                                    size_t l) const
{
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in ViewCArray 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewCArray 4D");  // die if >= dim0
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewCArray 4D");  // die if >= dim1
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewCArray 4D");  // die if >= dim2
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewCArray 4D");  // die if >= dim3
    
    return array_[l + (k * dims_[3])
                    + (j * dims_[3] * dims_[2])
                    + (i * dims_[3] * dims_[2] *  dims_[1])];
}

//5D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i, 
                                    size_t j, 
                                    size_t k, 
                                    size_t l, 
                                    size_t m) const
{
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in ViewCArray 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewCArray 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewCArray 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewCArray 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewCArray 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewCArray 5D!");
    
    return array_[m + (l * dims_[4])
                    + (k * dims_[4] * dims_[3])
                    + (j * dims_[4] * dims_[3] * dims_[2])
                    + (i * dims_[4] * dims_[3] * dims_[2] *  dims_[1])];
}

//6D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i,
                                    size_t j,
                                    size_t k,
                                    size_t l,
                                    size_t m,
                                    size_t n) const
{
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in ViewCArray 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewCArray 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewCArray 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewCArray 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewCArray 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewCArray 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in ViewCArray 6D!");
    
    return array_[n + (m * dims_[5])
                    + (l * dims_[5] * dims_[4])
                    + (k * dims_[5] * dims_[4] * dims_[3])
                    + (j * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                    + (i * dims_[5] * dims_[4] * dims_[3] * dims_[2] *  dims_[1])];
}

//7D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i,
                                    size_t j,
                                    size_t k,
                                    size_t l,
                                    size_t m,
                                    size_t n,
                                    size_t o) const
{
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in ViewCArray 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewCArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewCArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewCArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewCArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewCArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in ViewCArray 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in ViewCArray 7D!");
    
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
void ViewCArray<T>::operator=(M do_this_math){
    do_this_math(*this); // pass in this ViewFArray object
}// end of math opperation

//return size    
template <typename T>
inline size_t ViewCArray<T>::size() const {
    return length_;
}

template <typename T>
inline size_t ViewCArray<T>::dims(size_t i) const {
    assert(i < order_ && "ViewCArray order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to ViewCArray dims is out of bounds!");
    return dims_[i];
}

template <typename T>
inline size_t ViewCArray<T>::order() const {
    return order_;
}

template <typename T>
inline T* ViewCArray<T>::pointer() const {
    return array_;
}

//---end of ViewCArray class definitions----


//7. CMatrix
template <typename T>
class CMatrix {
        
private:
    size_t dims_[7];
    size_t length_; // Length of 1D array
    size_t order_;  // tensor order (rank)
    std::shared_ptr <T []> matrix_;
            
public:
        
    // default constructor
    CMatrix();

    CMatrix(size_t dim1);

    CMatrix(size_t dim1,
            size_t dim2);

    CMatrix(size_t dim1,
            size_t dim2,
            size_t dim3);

    CMatrix(size_t dim1,
            size_t dim2,
            size_t dim3,
            size_t dim4);

    CMatrix(size_t dim1,
            size_t dim2,
            size_t dim3,
            size_t dim4,
            size_t dim5);

    CMatrix (size_t dim1,
            size_t dim2,
            size_t dim3,
            size_t dim4,
            size_t dim5,
            size_t dim6);

    CMatrix (size_t dim1,
            size_t dim2,
            size_t dim3,
            size_t dim4,
            size_t dim5,
            size_t dim6,
            size_t dim7);

    CMatrix(const CMatrix& temp);
    
    //overload operators to access data
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

    //overload = operator
    CMatrix& operator= (const CMatrix &temp);

    //return array size
    size_t size() const;
    
    // return array dims
    size_t dims(size_t i) const;
    
    // return array order (rank)
    size_t order() const;

    //return pointer
    T* pointer() const;
    
    // deconstructor
    ~CMatrix( );
        
}; // end of CMatrix

// CMatrix class definitions

//constructors

//no dim

//1D
template <typename T>
CMatrix<T>::CMatrix() {
    matrix_ = NULL;
    length_ = 0;
}

//1D
template <typename T>
CMatrix<T>::CMatrix(size_t dim1)
{
    dims_[0] = dim1;
    order_ = 1;
    length_ = dim1;
    matrix_ = std::shared_ptr <T[]> (new T[length_]);
}

//2D
template <typename T>
CMatrix<T>::CMatrix(size_t dim1,
                    size_t dim2)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    order_ = 2;
    length_ = dim1 * dim2;
    matrix_ = std::shared_ptr <T[]> (new T[length_]);
}

//3D
template <typename T>
CMatrix<T>::CMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    order_ = 3;
    length_ = dim1 * dim2 * dim3;
    matrix_ = std::shared_ptr <T[]> (new T[length_]);
}

//4D
template <typename T>
CMatrix<T>::CMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3,
                    size_t dim4)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    order_ = 4;
    length_ = dim1 * dim2 * dim3 * dim4;
    matrix_ = std::shared_ptr <T[]> (new T[length_]);
}   

//5D
template <typename T>
CMatrix<T>::CMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3,
                    size_t dim4,
                    size_t dim5)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    order_ = 5;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5;
    matrix_ = std::shared_ptr <T[]> (new T[length_]);
}

//6D
template <typename T>
CMatrix<T>::CMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3,
                    size_t dim4,
                    size_t dim5,
                    size_t dim6)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    dims_[5] = dim6;
    order_ = 6;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5 * dim6;
    matrix_ = std::shared_ptr <T[]> (new T[length_]);
}

//7D
template <typename T>
CMatrix<T>::CMatrix(size_t dim1,
                    size_t dim2,
                    size_t dim3,
                    size_t dim4,
                    size_t dim5,
                    size_t dim6,
                    size_t dim7)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    dims_[5] = dim6;
    dims_[6] = dim7;
    order_ = 7;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * dim7;
    matrix_ = std::shared_ptr <T[]> (new T[length_]);
}

template <typename T>
CMatrix<T>::CMatrix(const CMatrix& temp) {
    
    // Do nothing if the assignment is of the form x = x
    
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for
        
        order_  = temp.order_;
        length_ = temp.length_;
        matrix_ = temp.matrix_;
    } // end if
    
} // end constructor

//overload () operator

//1D
template <typename T>
T& CMatrix<T>::operator()(size_t i) const
{
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in CMatrix 1D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in CMatrix 1D!");
    
    return matrix_[i-1];
}

//2D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j) const
{
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in CMatrix 2D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in CMatrix 2D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in CMatrix 2D!");
    
    return matrix_[(j-1) + (i-1)*dims_[1]];
}

//3D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k) const
{
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in CMatrix 3D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in CMatrix 3D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in CMatrix 3D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in CMatrix 3D!");
    
    return matrix_[(k-1) + (j-1)*dims_[2]
                         + (i-1)*dims_[2]*dims_[1]];
}

//4D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k,
                          size_t l) const
{
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in CMatrix 4D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in CMatrix 4D");  // die if >= dim0
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in CMatrix 4D");  // die if >= dim1
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in CMatrix 4D");  // die if >= dim2
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in CMatrix 4D");  // die if >= dim3
    
    return matrix_[(l-1) + (k-1)*dims_[3]
                         + (j-1)*dims_[3]*dims_[2]
                         + (i-1)*dims_[3]*dims_[2]*dims_[1]];
}

//5D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k,
                          size_t l,
                          size_t m) const
{
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in CMatrix 5D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in CMatrix 5D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in CMatrix 5D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in CMatrix 5D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in CMatrix 5D!");
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in CMatrix 5D!");
    
    return matrix_[(m-1) + (l-1)*dims_[4]
                         + (k-1)*dims_[4]*dims_[3]
                         + (j-1)*dims_[4]*dims_[3]*dims_[2]
                         + (i-1)*dims_[4]*dims_[3]*dims_[2]*dims_[1]];
}

//6D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k,
                          size_t l,
                          size_t m,
                          size_t n) const
{
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in CMatrix 6D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in CMatrix 6D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in CMatrix 6D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in CMatrix 6D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in CMatrix 6D!");
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in CMatrix 6D!");
    assert(n >= 1 && n <= dims_[5] && "n is out of bounds in CMatrix 6D!");
    
    return matrix_[ (n-1) + (m-1)*dims_[5]
                          + (l-1)*dims_[5]*dims_[4]
                          + (k-1)*dims_[5]*dims_[4]*dims_[3]
                          + (j-1)*dims_[5]*dims_[4]*dims_[3]*dims_[2]
                          + (i-1)*dims_[5]*dims_[4]*dims_[3]*dims_[2]*dims_[1]];
}

//7D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k,
                          size_t l,
                          size_t m,
                          size_t n,
                          size_t o) const
{
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in CMatrix 7D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in CMatrix 7D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in CMatrix 7D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in CMatrix 7D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in CMatrix 7D!");
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in CMatrix 7D!");
    assert(n >= 1 && n <= dims_[5] && "n is out of bounds in CMatrix 7D!");
    assert(o >= 1 && o <= dims_[6] && "o is out of bounds in CMatrix 7D!");
    
    return matrix_[(o-1) + (n-1)*dims_[6]
                         + (m-1)*dims_[6]*dims_[5]
                         + (l-1)*dims_[6]*dims_[5]*dims_[4]
                         + (k-1)*dims_[6]*dims_[5]*dims_[4]*dims_[3]
                         + (j-1)*dims_[6]*dims_[5]*dims_[4]*dims_[3]*dims_[2]
                         + (i-1)*dims_[6]*dims_[5]*dims_[4]*dims_[3]*dims_[2]*dims_[1]];
}

//overload = operator
//THIS = CMatrix<> temp
template <typename T>
CMatrix<T> &CMatrix<T>::operator= (const CMatrix &temp) {
    if(this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_  = temp.order_;
        length_ = temp.length_;
        matrix_ = temp.matrix_;
    }
  return *this;
}

template <typename T>
inline size_t CMatrix<T>::size() const {
    return length_;
}

template <typename T>
inline size_t CMatrix<T>::dims(size_t i) const {
    i--; // i starts at 1
    assert(i < order_ && "CMatrix order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to CMatrix dims is out of bounds!");
    return dims_[i];
}

template <typename T>
inline size_t CMatrix<T>::order() const {
    return order_;
}

template <typename T>
inline T* CMatrix<T>::pointer() const{
    return matrix_.get();
}

// Destructor
template <typename T>
CMatrix<T>::~CMatrix(){}

//----end of CMatrix class definitions----


//8. ViewCMatrix
//  indices [1:N]
template <typename T>
class ViewCMatrix {

private:
    size_t dims_[7];
    size_t length_; // Length of 1D array
    size_t order_;  // tensor order (rank)
     T * matrix_;
		    
public:
		    
    // default constructor
    ViewCMatrix();
		    
		    
    //--- 1D array ---	   	    
    // overloaded constructor
    ViewCMatrix (T *matrix,
                 size_t dim1);
    
    ViewCMatrix (T *matrix,
                 size_t dim1,
                 size_t dim2);

    ViewCMatrix (T *matrix,
		size_t dim1,
		size_t dim2,
		size_t dim3);

    ViewCMatrix (T *matrix,
		size_t dim1,
		size_t dim2,
		size_t dim3,
		size_t dim4);

    ViewCMatrix (T *matrix,
		size_t dim1,
		size_t dim2,
		size_t dim3,
		size_t dim4,
		size_t dim5);

    ViewCMatrix (T *matrix,
		   size_t dim1,
		   size_t dim2,
		   size_t dim3,
		   size_t dim4,
		   size_t dim5,
		   size_t dim6);

    ViewCMatrix (T *matrix,
                 size_t dim1,
                 size_t dim2,
                 size_t dim3,
                 size_t dim4,
                 size_t dim5,
                 size_t dim6,
                 size_t dim7);
    
    T& operator() (size_t i) const;
    
    T& operator() (size_t i,
                   size_t j) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n) const;
    T& operator() (size_t i,
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
    
}; // end of ViewCMatrix

//class definitions

//constructors

//no dim
template <typename T>
ViewCMatrix<T>::ViewCMatrix(){
  matrix_ = NULL;
  length_ = 0;
}

//1D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *matrix,
                            size_t dim1)
{
    dims_[0] = dim1;
    order_ = 1;
    length_ = dim1;
	matrix_ = matrix;
}

//2D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    order_ = 2;
    length_ = dim1 * dim2;
	matrix_ = matrix;
}

//3D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2,
                            size_t dim3)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    order_ = 3;
    length_ = dim1 * dim2 * dim3;
	matrix_ = matrix;
}

//4D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2,
                            size_t dim3,
                            size_t dim4)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    order_ = 4;
    length_ = dim1 * dim2 * dim3 * dim4;
	matrix_ = matrix;
}

//5D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2,
                            size_t dim3,
                            size_t dim4,
                            size_t dim5)
{
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    order_ = 5;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5;
	matrix_ = matrix;
}

//6D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2,
                            size_t dim3,
                            size_t dim4,
                            size_t dim5,
                            size_t dim6) {
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    dims_[5] = dim6;
    order_ = 6;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5 * dim6;
	matrix_ = matrix;
}

//7D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *matrix,
                            size_t dim1,
                            size_t dim2,
                            size_t dim3,
                            size_t dim4,
                            size_t dim5,
                            size_t dim6,
                            size_t dim7) {
    dims_[0] = dim1;
    dims_[1] = dim2;
    dims_[2] = dim3;
    dims_[3] = dim4;
    dims_[4] = dim5;
    dims_[5] = dim6;
    dims_[6] = dim7;
    order_ = 7;
    length_ = dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * dim7;
    matrix = matrix_;
}

//overload () operator

//1D
template <typename T>
T& ViewCMatrix<T>:: operator() (size_t i) const
{
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in ViewCMatrix 1D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewCMatrix 1D!");
    
	return matrix_[i-1];
}

//2D
template <typename T>
T& ViewCMatrix<T>::operator() (size_t i,
                               size_t j) const
{
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in ViewCMatrix 2D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewCMatrix 2D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewCMatrix 2D!");
    
    return matrix_[(j-1) + (i-1)*dims_[1]];
}

//3D
template <typename T>
T& ViewCMatrix<T>::operator () (size_t i,
                                size_t j,
                                size_t k) const
{
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in ViewCMatrix 3D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewCMatrix 3D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewCMatrix 3D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewCMatrix 3D!");
    
    return matrix_[(k-1) + (j-1)*dims_[2]
                         + (i-1)*dims_[2]*dims_[1]];
}

//4D
template <typename T>
T& ViewCMatrix<T>::operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l) const
{
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in ViewCMatrix 4D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewCMatrix 4D");  // die if >= dim0
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewCMatrix 4D");  // die if >= dim1
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewCMatrix 4D");  // die if >= dim2
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in ViewCMatrix 4D");  // die if >= dim3
    
    return matrix_[(l-1) + (k-1)*dims_[3]
                         + (j-1)*dims_[3]*dims_[2]
                         + (i-1)*dims_[3]*dims_[2]*dims_[1]];
}

//5D
template <typename T>
T& ViewCMatrix<T>::operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m) const
{
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in ViewCMatrix 5D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewCMatrix 5D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewCMatrix 5D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewCMatrix 5D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in ViewCMatrix 5D!");
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in ViewCMatrix 5D!");
    
    return matrix_[(m-1) + (l-1)*dims_[4]
                         + (k-1)*dims_[4]*dims_[3]
                         + (j-1)*dims_[4]*dims_[3]*dims_[2]
                         + (i-1)*dims_[4]*dims_[3]*dims_[2]*dims_[1]];
}

//6D
template <typename T>
T& ViewCMatrix<T>::operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m,
                              size_t n) const
{
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in ViewCMatrix 6D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewCMatrix 6D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewCMatrix 6D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewCMatrix 6D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in ViewCMatrix 6D!");
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in ViewCMatrix 6D!");
    assert(n >= 1 && n <= dims_[5] && "n is out of bounds in ViewCMatrix 6D!");
    
    return matrix_[(n-1) + (m-1)*dims_[5]
                         + (l-1)*dims_[5]*dims_[4]
                         + (k-1)*dims_[5]*dims_[4]*dims_[3]
                         + (j-1)*dims_[5]*dims_[4]*dims_[3]*dims_[2]
                         + (i-1)*dims_[5]*dims_[4]*dims_[3]*dims_[2]*dims_[1]];
}

//7D
template <typename T>
T& ViewCMatrix<T>::operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m,
                              size_t n,
                              size_t o) const
{
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in ViewCMatrix 7D!");
    assert(i >= 1 && i <= dims_[0] && "i is out of bounds in ViewCMatrix 7D!");
    assert(j >= 1 && j <= dims_[1] && "j is out of bounds in ViewCMatrix 7D!");
    assert(k >= 1 && k <= dims_[2] && "k is out of bounds in ViewCMatrix 7D!");
    assert(l >= 1 && l <= dims_[3] && "l is out of bounds in ViewCMatrix 7D!");
    assert(m >= 1 && m <= dims_[4] && "m is out of bounds in ViewCMatrix 7D!");
    assert(n >= 1 && n <= dims_[5] && "n is out of bounds in ViewCMatrix 7D!");
    assert(o >= 1 && o <= dims_[6] && "o is out of bounds in ViewCMatrix 7D!");
    
    return matrix_[(o-1) + (n-1)*dims_[6]
                         + (m-1)*dims_[6]*dims_[5]
                         + (l-1)*dims_[6]*dims_[5]*dims_[4]
                         + (k-1)*dims_[6]*dims_[5]*dims_[4]*dims_[3]
                         + (j-1)*dims_[6]*dims_[5]*dims_[4]*dims_[3]*dims_[2]
                         + (i-1)*dims_[6]*dims_[5]*dims_[4]*dims_[3]*dims_[2]*dims_[1]];
}

// calculate this ViewFArray object = math(A,B)
template <typename T>
template <typename M>
void ViewCMatrix<T>::operator=(M do_this_math){
    do_this_math(*this); // pass in this ViewFArray object
}// end of math opperation

template <typename T>
inline size_t ViewCMatrix<T>::size() const {
    return length_;
}

template <typename T>
inline size_t ViewCMatrix<T>::dims(size_t i) const {
    i--; // i starts at 1
    assert(i < order_ && "ViewCMatrix order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to ViewCMatrix dims is out of bounds!");
    return dims_[i];
}

template <typename T>
inline size_t ViewCMatrix<T>::order() const {
    return order_;
}

template <typename T>
inline T* ViewCMatrix<T>::pointer() const {
    return matrix_;
}


//----end of ViewCMatrix class definitions----

//9. RaggedRightArray
template <typename T>
class RaggedRightArray {
private:
    size_t *start_index_;
    T * array_;
    
    size_t dim1_, length_;
    size_t num_saved_; // the number saved in the 1D array
    
public:
    // Default constructor
    RaggedRightArray ();
    
    //--- 2D array access of a ragged right array ---
    
    // Overload constructor for a CArray
    RaggedRightArray (CArray<size_t> &strides_array);
    
    // Overload constructor for a ViewCArray
    RaggedRightArray (ViewCArray<size_t> &strides_array);
    
    // Overloaded constructor for a traditional array
    RaggedRightArray (size_t *strides_array, size_t some_dim1);
    
    // Overload constructor for a RaggedRightArray to
    // support a dynamically built stride_array
    RaggedRightArray (size_t some_dim1, size_t buffer);
    
    // A method to return the stride size
    size_t stride(size_t i) const;
    
    // A method to increase the number of column entries, i.e.,
    // the stride size. Used with the constructor for building
    // the stride_array dynamically.
    // DO NOT USE with the constructures with a strides_array
    void push_back(size_t i);
    
    // Overload operator() to access data as array(i,j)
    // where i=[0:N-1], j=[stride(i)]
    T& operator()(size_t i, size_t j) const;

    // method to return total size
    size_t size() const;

    //return pointer
    T* pointer() const;
    
    //get row starts array
    size_t* get_starts() const;

    RaggedRightArray& operator+= (const size_t i);

    RaggedRightArray& operator= (const RaggedRightArray &temp);

    // Destructor
    ~RaggedRightArray ( );
}; // End of RaggedRightArray

// Default constructor
template <typename T>
RaggedRightArray<T>::RaggedRightArray () {
    array_ = NULL;
    start_index_ = NULL;
    length_ = 0;
}


// Overloaded constructor with CArray
template <typename T>
RaggedRightArray<T>::RaggedRightArray (CArray<size_t> &strides_array){
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i);
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// Overloaded constructor with a view c array
template <typename T>
RaggedRightArray<T>::RaggedRightArray (ViewCArray<size_t> &strides_array) {
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i);
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// Overloaded constructor with a regular cpp array
template <typename T>
RaggedRightArray<T>::RaggedRightArray (size_t *strides_array, size_t dim1){
    // The length of the stride array is some_dim1;
    dim1_ = dim1;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array[i];
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// overloaded constructor for a dynamically built strides_array.
// buffer is the max number of columns needed
template <typename T>
RaggedRightArray<T>::RaggedRightArray (size_t some_dim1, size_t buffer){
    
    dim1_ = some_dim1;
    
    // create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1]();  // note the dim1+1
    //start_index_[0] = 0; // the 1D array starts at 0

    num_saved_ = 0;
    
    length_ = some_dim1*buffer;
    array_ = new T[some_dim1*buffer];
    
} // end constructor

// A method to return the stride size
template <typename T>
inline size_t RaggedRightArray<T>::stride(size_t i) const {
    // Ensure that i is within bounds
    assert(i < dim1_ && "i is greater than dim1_ in RaggedRightArray");

    return start_index_[(i + 1)] - start_index_[i];
}

// A method to increase the stride size, in other words,
// this is used to build the stride array dynamically
// DO NOT USE with constructors that are given a stride array
template <typename T>
void RaggedRightArray<T>::push_back(size_t i){
    num_saved_ ++;
    start_index_[i+1] = num_saved_;
}

// Overload operator() to access data as array(i,j)
// where i=[0:N-1], j=[0:stride(i)]
template <typename T>
inline T& RaggedRightArray<T>::operator()(size_t i, size_t j) const {
    // get the 1D array index
    size_t start = start_index_[i];
    
    // asserts
    assert(i < dim1_ && "i is out of dim1 bounds in RaggedRightArray");  // die if >= dim1
    //assert(j < stride(i) && "j is out of stride bounds in RaggedRightArray");  // die if >= stride
    assert(j+start < length_ && "j+start is out of bounds in RaggedRightArray");  // die if >= 1D array length)
    
    return array_[j + start];
} // End operator()

//return size
template <typename T>
size_t RaggedRightArray<T>::size() const {
    return length_;
}

template <typename T>
RaggedRightArray<T> & RaggedRightArray<T>::operator+= (const size_t i) {
    this->num_saved_ ++;
    this->start_index_[i+1] = num_saved_;
    return *this;
}

//overload = operator
template <typename T>
RaggedRightArray<T> & RaggedRightArray<T>::operator= (const RaggedRightArray &temp) {

    if( this != &temp) {
        dim1_ = temp.dim1_;
        length_ = temp.length_;
        num_saved_ = temp.num_saved_;
        if(start_index_!=NULL)
          delete[] start_index_;
        start_index_ = new size_t[dim1_ + 1];
        for (int j = 0; j < dim1_ + 1; j++) {
            start_index_[j] = temp.start_index_[j];  
        }

        if(array_!=NULL)
          delete[] array_;
        array_ = new T[length_];
        //copy contents
        for(int iter = 0; iter < length_; iter++)
          array_[iter] = temp.array_[iter];
    }
	
    return *this;
}

template <typename T>
inline T* RaggedRightArray<T>::pointer() const{
    return array_;
}

template <typename T>
inline size_t* RaggedRightArray<T>::get_starts() const{
    return start_index_;
}

// Destructor
template <typename T>
RaggedRightArray<T>::~RaggedRightArray () {
    if(array_!=NULL)
      delete[] array_;
    if(start_index_!=NULL)
      delete[] start_index_;
}

//----end of RaggedRightArray class definitions----

//9. RaggedRightArrayofVectors
template <typename T>
class RaggedRightArrayofVectors {
private:
    size_t *start_index_;
    T * array_;
    
    size_t dim1_, length_, vector_dim_;
    size_t num_saved_; // the number saved in the 1D array
    
public:
    // Default constructor
    RaggedRightArrayofVectors ();
    
    //--- 3D array access of a ragged right array storing a vector of size vector_dim_ at each (i,j)---
    
    // Overload constructor for a CArray
    RaggedRightArrayofVectors (CArray<size_t> &strides_array, size_t vector_dim);
    
    // Overload constructor for a ViewCArray
    RaggedRightArrayofVectors (ViewCArray<size_t> &strides_array, size_t vector_dim);
    
    // Overloaded constructor for a traditional array
    RaggedRightArrayofVectors (size_t *strides_array, size_t some_dim1, size_t vector_dim);
    
    // Overload constructor for a RaggedRightArray to
    // support a dynamically built stride_array
    RaggedRightArrayofVectors (size_t some_dim1, size_t buffer, size_t vector_dim);
    
    // A method to return the stride size
    size_t stride(size_t i) const;

    // A method to return the vector dim
    size_t vector_dim() const;
    
    // A method to increase the number of column entries, i.e.,
    // the stride size. Used with the constructor for building
    // the stride_array dynamically.
    // DO NOT USE with the constructures with a strides_array
    void push_back(size_t i);
    
    // Overload operator() to access data as array(i,j)
    // where i=[0:N-1], j=[stride(i)], k=[0,vector_dim_]
    T& operator()(size_t i, size_t j, size_t k) const;

    // method to return total size
    size_t size() const;

    //return pointer
    T* pointer() const;
    
    //get row starts array
    size_t* get_starts() const;

    RaggedRightArrayofVectors& operator+= (const size_t i);

    RaggedRightArrayofVectors& operator= (const RaggedRightArrayofVectors &temp);

    // Destructor
    ~RaggedRightArrayofVectors ( );
}; // End of RaggedRightArray

// Default constructor
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors () {
    array_ = NULL;
    start_index_ = NULL;
    length_ = 0;
}


// Overloaded constructor with CArray
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors (CArray<size_t> &strides_array, size_t vector_dim){
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    vector_dim_ = vector_dim;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i)*vector_dim_;
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// Overloaded constructor with a view c array
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors (ViewCArray<size_t> &strides_array, size_t vector_dim) {
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    vector_dim_ = vector_dim;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i)*vector_dim_;
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// Overloaded constructor with a regular cpp array
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors (size_t *strides_array, size_t dim1, size_t vector_dim){
    // The length of the stride array is some_dim1;
    dim1_ = dim1;
    vector_dim_ = vector_dim;

    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array of vectors and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array[i]*vector_dim_;
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// overloaded constructor for a dynamically built strides_array.
// buffer is the max number of columns needed
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors (size_t some_dim1, size_t buffer, size_t vector_dim){
    
    dim1_ = some_dim1;
    vector_dim_ = vector_dim;

    // create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1]();  // note the dim1+1
    //start_index_[0] = 0; // the 1D array starts at 0

    num_saved_ = 0;
    
    length_ = some_dim1*buffer*vector_dim;
    array_ = new T[some_dim1*buffer];
    
} // end constructor

// A method to return the stride size
template <typename T>
inline size_t RaggedRightArrayofVectors<T>::stride(size_t i) const {
    // Ensure that i is within bounds
    assert(i < dim1_ && "i is greater than dim1_ in RaggedRightArray");

    return (start_index_[(i + 1)] - start_index_[i])/vector_dim_;
}

// A method to increase the stride size, in other words,
// this is used to build the stride array dynamically
// DO NOT USE with constructors that are given a stride array
template <typename T>
void RaggedRightArrayofVectors<T>::push_back(size_t i){
    num_saved_ += vector_dim_;
    start_index_[i+1] = num_saved_;
}

// Overload operator() to access data as array(i,j,k)
// where i=[0:N-1], j=[0:stride(i)], k=[0:vector_dim_]
template <typename T>
inline T& RaggedRightArrayofVectors<T>::operator()(size_t i, size_t j, size_t k) const {
    // get the 1D array index
    size_t start = start_index_[i];
    
    // asserts
    assert(i < dim1_ && "i is out of dim1 bounds in RaggedRightArray");  // die if >= dim1
    //assert(j < stride(i) && "j is out of stride bounds in RaggedRightArray");  // die if >= stride
    assert(j*vector_dim_+start + k < length_ && "j+start is out of bounds in RaggedRightArray");  // die if >= 1D array length)
    
    return array_[j*vector_dim_ + start + k];
} // End operator()

//return size
template <typename T>
size_t RaggedRightArrayofVectors<T>::size() const {
    return length_;
}

template <typename T>
RaggedRightArrayofVectors<T> & RaggedRightArrayofVectors<T>::operator+= (const size_t i) {
    this->num_saved_ += vector_dim_;
    this->start_index_[i+1] = num_saved_;
    return *this;
}

//overload = operator
template <typename T>
RaggedRightArrayofVectors<T> & RaggedRightArrayofVectors<T>::operator= (const RaggedRightArrayofVectors &temp) {

    if( this != &temp) {
        dim1_ = temp.dim1_;
        vector_dim_ = temp.vector_dim_;
        length_ = temp.length_;
        num_saved_ = temp.num_saved_;
        if(start_index_!=NULL)
          delete[] start_index_;
        start_index_ = new size_t[dim1_ + 1];
        for (int j = 0; j < dim1_ + 1; j++) {
            start_index_[j] = temp.start_index_[j];  
        }

        if(array_!=NULL)
          delete[] array_;
        array_ = new T[length_];
        //copy contents
        for(int iter = 0; iter < length_; iter++)
          array_[iter] = temp.array_[iter];
    }
	
    return *this;
}

template <typename T>
inline T* RaggedRightArrayofVectors<T>::pointer() const{
    return array_;
}

template <typename T>
inline size_t* RaggedRightArrayofVectors<T>::get_starts() const{
    return start_index_;
}

// Destructor
template <typename T>
RaggedRightArrayofVectors<T>::~RaggedRightArrayofVectors () {
    if(array_!=NULL)
      delete[] array_;
    if(start_index_!=NULL)
      delete[] start_index_;
}

//----end of RaggedRightArrayofVectors class definitions----

//10. RaggedDownArray
template <typename T>
class RaggedDownArray { 
private:
    size_t *start_index_;
	T * array_;

	size_t dim2_;
    size_t length_;
    size_t num_saved_; // the number saved in the 1D array

public:
    //default constructor
    RaggedDownArray() ;

    //~~~~2D`~~~~
	//overload constructor with CArray
	RaggedDownArray(CArray<size_t> &strides_array);

	//overload with ViewCArray
	RaggedDownArray(ViewCArray <size_t> &strides_array);

	//overload with traditional array
	RaggedDownArray(size_t *strides_array, size_t dome_dim1);

    // Overload constructor for a RaggedDownArray to
    // support a dynamically built stride_array
    RaggedDownArray (size_t some_dim2, size_t buffer);
    
	//method to return stride size
	size_t stride(size_t j);

    // A method to increase the number of column entries, i.e.,
    // the stride size. Used with the constructor for building
    // the stride_array dynamically.
    // DO NOT USE with the constructures with a strides_array
    void push_back(size_t j);
    
	//overload () operator to access data as array (i,j)
	T& operator()(size_t i, size_t j);

    // method to return total size
    size_t size();

    //return pointer
    T* pointer() const;
    
    //get row starts array
    size_t* get_starts() const;

    //overload = operator
    RaggedDownArray& operator= (const RaggedDownArray &temp);

    //destructor
    ~RaggedDownArray();

}; //~~~~~end of RaggedDownArray class declarations~~~~~~~~	

//no dims
template <typename T>
RaggedDownArray<T>::RaggedDownArray() {
    array_ = NULL;
    start_index_ = NULL;
    length_ = 0;
}

//overload constructor with CArray 
template <typename T>
RaggedDownArray<T>::RaggedDownArray( CArray <size_t> &strides_array) {
    // Length of stride array
    //dim2_ = strides_array.size();

    // Create and initialize startding indices
    start_index_ = new size_t[dim2_+1]; //theres a plus 1, because 
    start_index_[0] = 0; //1D array starts at 0

		
	//length of strides
	dim2_ = strides_array.size();

    // Loop to find total length of 1D array
    size_t count = 0;
    for(size_t j = 0; j < dim2_ ; j++) { 
        count += strides_array(j);
        start_index_[j+1] = count;
    } 
    length_ = count;

    array_ = new T[length_];

} // End constructor 

// Overload constructor with ViewCArray
template <typename T>
RaggedDownArray<T>::RaggedDownArray( ViewCArray <size_t> &strides_array) {
    // Length of strides
    //dim2_ = strides_array.size();

    //create array for holding start indices
    start_index_ = new size_t[dim2_+1];
    start_index_[0] = 0;

    size_t count = 0;
    // Loop over to get total length of 1D array
    for(size_t j = 0; j < dim2_ ;j++ ) {
        count += strides_array(j);
        start_index_[j+1] = count;
    }
    length_ = count;	
    array_ = new T[length_];

} // End constructor 

// Overload constructor with regualar array
template <typename T>
RaggedDownArray<T>::RaggedDownArray( size_t *strides_array, size_t dim2){
    // Length of stride array
    dim2_ = dim2;

    // Create and initialize starting index of entries
    start_index_ = new size_t[dim2_+1];
    start_index_[0] = 0;

    // Loop over to find length of 1D array
    // Represent ragged down array and set 1D index
    size_t count = 0;
    for(size_t j = 0; j < dim2_; j++) {
        count += strides_array[j];
        start_index_[j+1] = count;
	}

    length_ = count;	
    array_ = new T[length_];

} //end construnctor

// overloaded constructor for a dynamically built strides_array.
// buffer is the max number of columns needed
template <typename T>
RaggedDownArray<T>::RaggedDownArray (size_t some_dim2, size_t buffer){
    
    dim2_ = some_dim2;
    
    // create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim2_+1]();  // note the dim2+1
    //start_index_[0] = 0; // the 1D array starts at 0
    
    num_saved_ = 0;
    
    length_ = some_dim2*buffer;
    array_ = new T[some_dim2*buffer];
    
} // end constructor

// Check the stride size
template <typename T>
size_t RaggedDownArray<T>::stride(size_t j) {
    assert(j < dim2_ && "j is greater than dim2_ in RaggedDownArray");

    return start_index_[j+1] - start_index_[j];
}

// A method to increase the stride size, in other words,
// this is used to build the stride array dynamically
// DO NOT USE with constructors that are given a stride array
template <typename T>
void RaggedDownArray<T>::push_back(size_t j){
    num_saved_ ++;
    start_index_[j+1] = num_saved_;
}

//return size
template <typename T>
size_t RaggedDownArray<T>::size() {
    return length_;
}

// overload operator () to access data as an array(i,j)
// Note: i = 0:stride(j), j = 0:N-1
template <typename T>
T& RaggedDownArray<T>::operator()(size_t i, size_t j) {
    // Where is the array starting?
    // look at start index
    size_t start = start_index_[j]; 

    // Make sure we are within array bounds
    assert(i < stride(j) && "i is out of bounds in RaggedDownArray");
    assert(j < dim2_ && "j is out of dim2_ bounds in RaggedDownArray");
    assert(i+start < length_ && "i+start is out of bounds in RaggedDownArray");  // die if >= 1D array length)
    
    return array_[i + start];

} // End () operator

//overload = operator
template <typename T>
RaggedDownArray<T> & RaggedDownArray<T>::operator= (const RaggedDownArray &temp) {

    if( this != &temp) {
        dim2_ = temp.dim2_;
        length_ = temp.length_;
        num_saved_ = temp.num_saved_;
        if(start_index_!=NULL)
          delete[] start_index_;

        start_index_ = new size_t[dim2_ + 1];
        for (int j = 0; j < dim2_ + 1; j++) {
            start_index_[j] = temp.start_index_[j];  
        }

        if(array_!=NULL)
          delete[] array_;
        array_ = new T[length_];
        //copy contents
        for(int iter = 0; iter < length_; iter++)
          array_[iter] = temp.array_[iter];
    }
	
    return *this;
}

template <typename T>
inline T* RaggedDownArray<T>::pointer() const{
    return array_;
}


template <typename T>
inline size_t* RaggedDownArray<T>::get_starts() const{
    return start_index_;
}

// Destructor
template <typename T>
RaggedDownArray<T>::~RaggedDownArray() {
  if(array_!=NULL)
    delete[] array_;
  if(start_index_!=NULL)
    delete[] start_index_;

} // End destructor


//----end of RaggedDownArray----


//11. DynamicRaggedRightArray

template <typename T>
class DynamicRaggedRightArray {
private:
    size_t *stride_;
    T * array_;
    
    size_t dim1_;
    size_t dim2_;
    size_t length_;
    
public:
    // Default constructor
    DynamicRaggedRightArray ();
    
    //--- 2D array access of a ragged right array ---
    
    // overload constructor
    DynamicRaggedRightArray (size_t dim1, size_t dim2);
    
    // A method to return or set the stride size
    size_t& stride(size_t i) const;
    
    // A method to return the size
    size_t size() const;

    //return pointer
    T* pointer() const;
    
    // Overload operator() to access data as array(i,j),
    // where i=[0:N-1], j=[stride(i)]
    T& operator()(size_t i, size_t j) const;
    
    // Overload copy assignment operator
    DynamicRaggedRightArray& operator= (const DynamicRaggedRightArray &temp);
    
    // Destructor
    ~DynamicRaggedRightArray ();
};

//nothing
template <typename T>
DynamicRaggedRightArray<T>::DynamicRaggedRightArray () {
    array_ = NULL;
    stride_ = NULL;
    length_ = 0;
}

// Overloaded constructor
template <typename T>
DynamicRaggedRightArray<T>::DynamicRaggedRightArray (size_t dim1, size_t dim2) {
    // The dimensions of the array;
    dim1_  = dim1;
    dim2_  = dim2;
    length_ = dim1*dim2;
    
    // Create memory on the heap for the values
    array_ = new T[dim1*dim2];
    
    // Create memory for the stride size in each row
    stride_ = new size_t[dim1];
    
    // Initialize the stride
    for (int i=0; i<dim1_; i++){
        stride_[i] = 0;
    }
    
    // Start index is always = j + i*dim2
}

// A method to set the stride size for row i
template <typename T>
size_t& DynamicRaggedRightArray<T>::stride(size_t i) const {
    return stride_[i];
}

//return size
template <typename T>
size_t DynamicRaggedRightArray<T>::size() const{
    return length_;
}

// Overload operator() to access data as array(i,j),
// where i=[0:N-1], j=[0:stride(i)]
template <typename T>
inline T& DynamicRaggedRightArray<T>::operator()(size_t i, size_t j) const {
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in DynamicRaggedRight");  // die if >= dim1
    assert(j < dim2_ && "j is out of dim2 bounds in DynamicRaggedRight");  // die if >= dim2
    assert(j < stride_[i] && "j is out of stride bounds in DynamicRaggedRight");  // die if >= stride
    
    return array_[j + i*dim2_];
}

//overload = operator
template <typename T>
inline DynamicRaggedRightArray<T>& DynamicRaggedRightArray<T>::operator= (const DynamicRaggedRightArray &temp)
{
    
    if( this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        length_ = temp.length_;
        if(stride_!=NULL)
          delete[] stride_;
        stride_ = new size_t[dim1_];
        for (int i = 0; i < dim1_; i++) {
            stride_[i] = temp.stride_[i];
        }

        if(array_!=NULL)
          delete[] array_;
        array_ = new T[length_];
        //copy contents
        for(int iter = 0; iter < length_; iter++)
          array_[iter] = temp.array_[iter];
    }
    
    return *this;
}

template <typename T>
inline T* DynamicRaggedRightArray<T>::pointer() const{
    return array_;
}

// Destructor
template <typename T>
DynamicRaggedRightArray<T>::~DynamicRaggedRightArray() {
    if(array_!=NULL)
      delete[] array_;
    if(stride_!=NULL)
      delete[] stride_;
}




//----end DynamicRaggedRightArray class definitions----


//12. DynamicRaggedDownArray

template <typename T>
class DynamicRaggedDownArray {
private:
    size_t *stride_;
    T * array_;
    
    size_t dim1_;
    size_t dim2_;
    size_t length_;
    
public:
    // Default constructor
    DynamicRaggedDownArray ();
    
    //--- 2D array access of a ragged right array ---
    
    // overload constructor
    DynamicRaggedDownArray (size_t dim1, size_t dim2);
    
    // A method to return or set the stride size
    size_t& stride(size_t j) const;
    
    // A method to return the size
    size_t size() const;
    
    // Overload operator() to access data as array(i,j),
    // where i=[stride(j)], j=[0:N-1]
    T& operator()(size_t i, size_t j) const;
    
    // Overload copy assignment operator
    DynamicRaggedDownArray& operator= (const DynamicRaggedDownArray &temp);

    //return pointer
    T* pointer() const;
    
    // Destructor
    ~DynamicRaggedDownArray ();
};

//nothing
template <typename T>
DynamicRaggedDownArray<T>::DynamicRaggedDownArray () {
    array_ = NULL;
    stride_ = NULL;
    length_ = 0;
}

// Overloaded constructor
template <typename T>
DynamicRaggedDownArray<T>::DynamicRaggedDownArray (size_t dim1, size_t dim2) {
    // The dimensions of the array;
    dim1_  = dim1;
    dim2_  = dim2;
    length_ = dim1*dim2;
    
    // Create memory on the heap for the values
    array_ = new T[dim1*dim2];
    
    // Create memory for the stride size in each row
    stride_ = new size_t[dim2];
    
    // Initialize the stride
    for (int j=0; j<dim2_; j++){
        stride_[j] = 0;
    }
    
    // Start index is always = i + j*dim1
}

// A method to set the stride size for column j
template <typename T>
size_t& DynamicRaggedDownArray<T>::stride(size_t j) const {
    return stride_[j];
}

//return size
template <typename T>
size_t DynamicRaggedDownArray<T>::size() const{
    return length_;
}

// overload operator () to access data as an array(i,j)
// Note: i = 0:stride(j), j = 0:N-1

template <typename T>
inline T& DynamicRaggedDownArray<T>::operator()(size_t i, size_t j) const {
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in DynamicRaggedDownArray");  // die if >= dim1
    assert(j < dim2_ && "j is out of dim2 bounds in DynamicRaggedDownArray");  // die if >= dim2
    assert(i < stride_[j] && "i is out of stride bounds in DynamicRaggedDownArray");  // die if >= stride
    
    return array_[i + j*dim1_];
}

//overload = operator
template <typename T>
inline DynamicRaggedDownArray<T>& DynamicRaggedDownArray<T>::operator= (const DynamicRaggedDownArray &temp)
{
    
    if( this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        length_ = temp.length_;
        if(stride_!=NULL)
          delete[] stride_;
        stride_ = new size_t[dim1_];
        for (int j = 0; j < dim2_; j++) {
            stride_[j] = temp.stride_[j];
        }
        if(array_!=NULL)
          delete[] array_;
        array_ = new T[length_];
        //copy contents
        for(int iter = 0; iter < length_; iter++)
          array_[iter] = temp.array_[iter];
    }
    
    return *this;
}

template <typename T>
inline T* DynamicRaggedDownArray<T>::pointer() const{
    return array_;
}

// Destructor
template <typename T>
DynamicRaggedDownArray<T>::~DynamicRaggedDownArray() {
    if(array_!=NULL)
      delete[] array_;
    if(stride_!=NULL)
      delete[] stride_;
}

//----end of DynamicRaggedDownArray class definitions-----




//13. SparseRowArray
template <typename T>
class SparseRowArray {
private:
    size_t *start_index_;
    size_t *column_index_;
    
    T * array_;
    
    size_t dim1_, length_;
    
public:
    // Default constructor
    SparseRowArray ();
    
    //--- 2D array access of a ragged right array ---
    
    // Overload constructor for a CArray
    SparseRowArray (CArray<size_t> &strides_array);
    
    // Overload constructor for a ViewCArray
    SparseRowArray (ViewCArray<size_t> &strides_array);
    
    // Overloaded constructor for a traditional array
    SparseRowArray (size_t *strides_array, size_t some_dim1);
    
    // A method to return the stride size
    size_t stride(size_t i) const;
    
    // A method to return the column index as array.column_index(i,j)
    size_t& column_index(size_t i, size_t j) const;
    
    // A method to access data as array.value(i,j),
    // where i=[0:N-1], j=[stride(i)]
    T& value(size_t i, size_t j) const;

    // A method to return the total size of the array
    size_t size() const;

    //return pointer
    T* pointer() const;

    //get row starts array
    size_t* get_starts() const;
    
    // Destructor
    ~SparseRowArray ();
}; 

//Default Constructor
template <typename T>
SparseRowArray<T>::SparseRowArray (){
    array_ = NULL;
    start_index_ = NULL;
    column_index_ = NULL;
    length_ = 0;
}
// Overloaded constructor
template <typename T>
SparseRowArray<T>::SparseRowArray (CArray<size_t> &strides_array) {
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i);
        start_index_[i+1] = count;
    } // end for i
    
    length_ = count;
    array_ = new T[count];
    column_index_ = new size_t[count];
} 


// Overloaded constructor
template <typename T>
SparseRowArray<T>::SparseRowArray (ViewCArray<size_t> &strides_array) {
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i);
        start_index_[i+1] = count;
    } // end for i
    
    length_ = count;
    array_ = new T[count];
    column_index_ = new size_t[count];
} 

// Overloaded constructor
template <typename T>
SparseRowArray<T>::SparseRowArray (size_t *strides_array, size_t dim1) {
    // The length of the stride array is some_dim1;
    dim1_ = dim1;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array[i];
        start_index_[i+1] = count;
    } // end for i
    
    length_ = count;
    array_ = new T[count];
    column_index_ = new size_t[count];
} 


// A method to return the stride size
template <typename T>
size_t SparseRowArray<T>::stride(size_t i) const {
    return start_index_[i+1] - start_index_[i];
}

// A method to return the column index
template <typename T>
size_t& SparseRowArray<T>::column_index(size_t i, size_t j) const {
    // Get the 1D array index
    size_t start = start_index_[i];
    
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in SparseRowArray");  // die if >= dim1
    assert(j < stride(i) && "j is out of stride bounds in SparseRowArray");  // die if >= stride
    
    return column_index_[j + start];
}

// Access data as array.value(i,j), 
// where i=[0:N-1], j=[0:stride(i)]
template <typename T>
inline T& SparseRowArray<T>::value(size_t i, size_t j) const {
    // Get the 1D array index
    size_t start = start_index_[i];
    
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in sparseRowArray");  // die if >= dim1
    assert(j < stride(i) && "j is out of stride bounds in sparseRowArray");  // die if >= stride
    
    return array_[j + start];
} 

//return size
template <typename T>
size_t SparseRowArray<T>::size() const{
    return length_;
}

template <typename T>
inline T* SparseRowArray<T>::pointer() const{
    return array_;
}

template <typename T>
inline size_t* SparseRowArray<T>::get_starts() const{
    return start_index_;
}

// Destructor
template <typename T>
SparseRowArray<T>::~SparseRowArray() {
    if(array_!=NULL)
      delete[] array_;
    if(start_index_!=NULL)
      delete[] start_index_;
    if(column_index_!=NULL)
      delete[] column_index_;
}

//---- end of SparseRowArray class definitions-----



//14. SparseColArray
template <typename T>
class SparseColArray {

private:
	size_t *start_index_;
	size_t *row_index_;
	T * array_;

	size_t dim2_, length_;

public:

	//default constructor
	SparseColArray ();

	//constructor with CArray
	SparseColArray(CArray<size_t> &strides_array);

	//constructor with ViewCArray
	SparseColArray(ViewCArray<size_t> &strides_array);

	//constructor with regular array
	SparseColArray(size_t *strides_array, size_t some_dim1);

	//method return stride size
	size_t stride(size_t j) const;

	//method return row index ass array.row_index(i,j)
	size_t& row_index(size_t i, size_t j) const;

	//method access data as an array
	T& value(size_t i, size_t j) const;

    // A method to return the total size of the array
    size_t size() const;

    //return pointer
    T* pointer() const;

    //get row starts array
    size_t* get_starts() const;

	//destructor
	~SparseColArray();
};

//Default Constructor
template <typename T>
SparseColArray<T>::SparseColArray (){
    array_ = NULL;
    start_index_ = NULL;
    row_index_ = NULL;
    length_ = 0;
}
//overload constructor with CArray
template <typename T>
SparseColArray<T>::SparseColArray(CArray<size_t> &strides_array) {

	dim2_ = strides_array.size();

	start_index_ = new size_t[dim2_+1];
	start_index_[0] = 0;

	//loop over to find total length of the 1D array
	size_t count = 0;
	for(size_t j = 0; j < dim2_; j++) {
	  count+= strides_array(j);
	  start_index_[j+1] = count;
	}
    
    length_ = count;
	array_ = new T[count];
	row_index_ = new T[count];

} //end constructor with CArray


//overload constructor with ViewCArray
template <typename T>
SparseColArray<T>::SparseColArray(ViewCArray<size_t> &strides_array) {

	dim2_ = strides_array.size();

	//create and initialize starting index of 1D array
	start_index_ = new size_t[dim2_+1];
	start_index_[0] = 0;

	//loop over to find total length of 1D array
	size_t count = 0;
	for(size_t j = 0; j < dim2_ ; j++) {
	  count += strides_array(j);
	  start_index_[j+1] = count;
	}
    
    length_ = count;
	array_ = new T[count];
	row_index_ = new T[count];

} //end constructor

//overload constructor with traditional array
template <typename T>
SparseColArray<T>::SparseColArray(size_t *strides_array, size_t dim2) {

	dim2_ = dim2;

	//create and initialize the starting index 
	start_index_ = new size_t[dim2_ +1];
	start_index_[0] = 0;

	//loop over to find the total length of the 1D array
	size_t count = 0;
	for(size_t j = 0; j < dim2_; j++) {
	  count += strides_array[j];
	  start_index_[j+1] = count;
	}
    
    length_ = count;
	array_ = new T[count];
	row_index_ = new T[count];

} //end constructor

//method to return stride size
template <typename T>
size_t SparseColArray<T>::stride(size_t j) const{
	return start_index_[j+1] - start_index_[j];
}

//acces data ass arrow.row_index(i,j)
// where i = 0:stride(j), j = 0:N-1
template <typename T>
size_t& SparseColArray<T>::row_index(size_t i, size_t j) const {

	//get 1D array index
	size_t start = start_index_[j];

	//asserts to make sure we are in bounds
	assert(i < stride(j) && "i is out of stride bounnds in SparseColArray!");
	assert(j < dim2_ && "j is out of dim1 bounds in SparseColArray");

	return row_index_[i + start];

} //end row index method	


//access values as array.value(i,j)
// where i = 0:stride(j), j = 0:N-1
template <typename T>
T& SparseColArray<T>::value(size_t i, size_t j) const {

	size_t start = start_index_[j];

	//asserts
	assert(i < stride(j) && "i is out of stride boundns in SparseColArray");
	assert(j < dim2_ && "j is out of dim1 bounds in SparseColArray");

	return array_[i + start];

}

//return size
template <typename T>
size_t SparseColArray<T>::size() const{
    return length_;
}

template <typename T>
inline T* SparseColArray<T>::pointer() const{
    return array_;
}

template <typename T>
inline size_t* SparseColArray<T>::get_starts() const{
    return start_index_;
}

//destructor
template <typename T>
SparseColArray<T>::~SparseColArray() {
    if(array_!=NULL)
	  delete [] array_;
    if(start_index_!=NULL)
	  delete [] start_index_;
    if(row_index_!=NULL)
	  delete [] row_index_;
}

//----end SparseColArray----
