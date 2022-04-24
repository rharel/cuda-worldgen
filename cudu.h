/**
 * Copyright (c) 2020 Raoul Harel
 * All rights reserved
 */

#pragma once

#include <array>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "cuda_runtime.h"

#ifdef _DEBUG
    #define CUDU_DEBUG_ASSERT(expression) \
        cudu::abort_if_false((expression), __FILE__, __LINE__, #expression)
#else
    #define CUDU_DEBUG_ASSERT(expression)
#endif

#define CUDU_STATIC_ASSERT(expression) \
    static_assert(expression, #expression)

#define CUDU_CHECK(expression) \
    cudu::abort_on_error((expression), __FILE__, __LINE__, #expression)

#define CUDU_LAUNCH(kernel, workload, ...) \
    (kernel)<<<(workload).nr_blocks, (workload).nr_threads>>>(__VA_ARGS__); \
    CUDU_CHECK(cudaGetLastError()); \
    CUDU_CHECK(cudaDeviceSynchronize());

#define CUDU_LAUNCH_SHARED(kernel, workload, shared_bytes, ...) \
    (kernel)<<<(workload).nr_blocks, (workload).nr_threads, (shared_bytes)>>>(__VA_ARGS__); \
    CUDU_CHECK(cudaGetLastError()); \
    CUDU_CHECK(cudaDeviceSynchronize());

#define CUDU_LAUNCH_BATCHES(kernel, nr_jobs, batch_size, ...) { \
    const cudu::Workload workload = cudu::Workload::for_jobs(batch_size); \
    for (size_t offset = 0; offset < (nr_jobs); offset += (batch_size)) \
    { \
        CUDU_LAUNCH(kernel, workload, offset, __VA_ARGS__); \
    } \
}

#define CUDU_LAUNCH_BATCHES_SHARED(kernel, nr_jobs, batch_size, shared_bytes, ...) { \
    const cudu::Workload workload = cudu::Workload::for_jobs(batch_size); \
    for (size_t offset = 0; offset < (nr_jobs); offset += (batch_size)) \
    { \
        CUDU_LAUNCH_SHARED(kernel, workload, shared_bytes, offset, __VA_ARGS__); \
    } \
}

#define CUDU_THREAD_ID() \
    (blockIdx.x * blockDim.x + threadIdx.x)

#define CUDU_THREAD_TOTAL() \
    (gridDim.x * blockDim.x)

namespace cudu 
{
    void abort_if_false(
        bool status,
        const std::string& file,
        const unsigned line,
        const std::string& message);

    void abort_on_error(
        cudaError_t status,
        const std::string& file,
        const unsigned line,
        const std::string& message);

    struct Workload
    {
        static unsigned max_threads_per_block();
        static Workload for_jobs(unsigned nr_jobs);

        unsigned nr_threads;
        unsigned nr_blocks;
    };

    template <typename T, size_t rank>
    struct Point
    {
        __host__ __device__ Point()
        {
            for (size_t i = 0; i < rank; ++i)
            {
                (*this)[i] = 0;
            }
        }

        __host__ __device__ Point(T x)
        {
            CUDU_STATIC_ASSERT(rank == 1);

            (*this)[0] = x;
        }

        __host__ __device__ Point(T x, T y)
        {
            CUDU_STATIC_ASSERT(rank == 2);

            (*this)[0] = x;
            (*this)[1] = y;
        }

        __host__ __device__ Point(T x, T y, T z)
        {
            CUDU_STATIC_ASSERT(rank == 3);

            (*this)[0] = x;
            (*this)[1] = y;
            (*this)[2] = z;
        }

        __host__ __device__ T& operator[](const size_t i)
        {
            return m_data[i];
        }

        __host__ __device__ const T& operator[](const size_t i) const
        {
            return m_data[i];
        }

        __host__ __device__ T* begin()
        {
            return &(*this)[0];
        }

        __host__ __device__ const T* begin() const
        {
            return &(*this)[0];
        }

        __host__ __device__ T* end()
        {
            return begin() + rank;
        }

        __host__ __device__ const T* end() const
        {
            return begin() + rank;
        }

        template <typename Target_t>
        __host__ __device__ Point<Target_t, rank> as() const
        {
            Point<Target_t, rank> result;
            for (size_t i = 0; i < rank; ++i)
            {
                result[i] = (*this)[i];
            }
            return result;
        }

        T m_data[rank];
    };

    template <typename T>
    using Point1D = Point<T, 1>;
    template <typename T>
    using Point2D = Point<T, 2>;
    template <typename T>
    using Point3D = Point<T, 3>;

    template <size_t rank>
    using Shape = Point<size_t, rank>;
    using Shape1D = Shape<1>;
    using Shape2D = Shape<2>;
    using Shape3D = Shape<3>;

    namespace host
    {
        template <typename T, size_t rank>
        class Array
        {
        public:
            CUDU_STATIC_ASSERT(rank > 0);

            explicit Array(const Shape<rank>& shape) :
                m_shape(shape),
                m_data(new T[size()])
            {
            }

            Array(const Shape<rank>& shape, T fill_value) : Array(shape)
            {
                std::fill(begin(), end(), fill_value);
            }

            Array() : Array(Shape<rank>())
            {
            }

            const Shape<rank>& shape() const
            {
                return m_shape;
            }

            size_t size() const
            {
                return std::accumulate(shape().begin(), shape().end(), 1, std::multiplies<size_t>());
            }

            size_t size_bytes() const
            {
                return size() * sizeof(T);
            }

            T& operator[](const size_t i)
            {
                CUDU_DEBUG_ASSERT(i < size());
                
                return m_data[i];
            }

            const T& operator[](const size_t i) const
            {
                CUDU_DEBUG_ASSERT(i < size());

                return m_data[i];
            }

            T& operator()(const size_t i, const size_t j)
            {
                CUDU_DEBUG_ASSERT(i < shape()[0]);
                CUDU_DEBUG_ASSERT(j < shape()[1]);

                return (*this)[data_index(i, j)];
            }

            const T& operator()(const size_t i, const size_t j) const
            {
                CUDU_DEBUG_ASSERT(i < shape()[0]);
                CUDU_DEBUG_ASSERT(j < shape()[1]);

                return (*this)[data_index(i, j)];
            }

            T& operator()(const size_t i, const size_t j, const size_t k)
            {
                CUDU_DEBUG_ASSERT(i < shape()[0]);
                CUDU_DEBUG_ASSERT(j < shape()[1]);
                CUDU_DEBUG_ASSERT(k < shape()[2]);

                return (*this)[data_index(i, j, k)];
            }

            const T& operator()(const size_t i, const size_t j, const size_t k) const
            {
                CUDU_DEBUG_ASSERT(i < shape()[0]);
                CUDU_DEBUG_ASSERT(j < shape()[1]);
                CUDU_DEBUG_ASSERT(k < shape()[2]);

                return (*this)[data_index(i, j, k)];
            }

            T& operator()(const Point2D<size_t>& p)
            {
                return (*this)(p[0], p[1]);
            }

            const T& operator()(const Point2D<size_t>& p) const
            {
                return (*this)(p[0], p[1]);
            }

            T& operator()(const Point3D<size_t>& p)
            {
                return (*this)(p[0], p[1], p[2]);
            }

            const T& operator()(const Point3D<size_t>& p) const
            {
                return (*this)(p[0], p[1], p[2]);
            }

            T* begin()
            {
                return &(*this)[0];
            }

            const T* begin() const
            {
                return &(*this)[0];
            }

            T* end()
            {
                return begin() + size();
            }

            const T* end() const
            {
                return begin() + size();
            }

        private:
            size_t data_index(const size_t i, const size_t j) const
            {
                return i * shape()[1] + j;
            }

            size_t data_index(const size_t i, const size_t j, const size_t k) const
            {
                return (i * shape()[1] + j) * shape()[2] + k;
            }

            Shape<rank> m_shape;
            std::unique_ptr<T[]> m_data;
        };

        template <typename T>
        using Array1D = Array<T, 1>;
        template <typename T>
        using Array2D = Array<T, 2>;
        template <typename T>
        using Array3D = Array<T, 3>;
    }

    namespace device
    {
        template <typename T>
        class DataBlock
        {
        public:
            static DataBlock from_ptr(const T* data, const size_t size)
            {
                DataBlock block(size);
                block.upload_all(data);
                return block;
            }

            explicit DataBlock(const size_t size = 0) :
                m_size(size),
                m_data(nullptr)
            {
                if (m_size > 0)
                {
                    CUDU_CHECK(cudaMalloc((void**)&m_data, size_bytes()));
                }
            }

            DataBlock(const DataBlock&) = delete;
            DataBlock& operator=(const DataBlock&) = delete;

            DataBlock(DataBlock&& other) noexcept
            {
                *this = std::move(other);
            }

            DataBlock& operator=(DataBlock&& other) noexcept
            {
                std::swap(m_size, other.m_size);
                std::swap(m_data, other.m_data);

                return *this;
            }

            ~DataBlock()
            {
                if (m_size > 0 && m_data != nullptr)
                {
                    CUDU_CHECK(cudaFree(m_data));
                    m_size = 0;
                    m_data = nullptr;
                }
            }

            size_t size() const
            {
                return m_size;
            }

            size_t size_bytes() const
            {
                return m_size * sizeof(T);
            }

            T* ptr()
            {
                return m_data;
            }

            const T* ptr() const
            {
                return m_data;
            }

            void download(T* const host, const size_t offset, const size_t extent) const
            {
                CUDU_DEBUG_ASSERT(offset + extent <= size());
                CUDU_CHECK(cudaMemcpy(
                    host,
                    ptr() + offset,
                    extent * sizeof(T),
                    cudaMemcpyDeviceToHost
                ));
            }

            void download_all(T* const host) const
            {
                download(host, 0, size());
            }

            T download_single(const size_t offset) const
            {
                T value;
                download(&value, offset, 1);
                return value;
            }

            void upload(const T* const host, const size_t offset, const size_t extent)
            {
                CUDU_DEBUG_ASSERT(offset + extent <= size());
                CUDU_CHECK(cudaMemcpy(
                    ptr() + offset,
                    host,
                    extent * sizeof(T),
                    cudaMemcpyHostToDevice
                ));
            }

            void upload_all(const T* const host)
            {
                upload(host, 0, size());
            }

            void upload_single(const T value, const size_t offset)
            {
                upload(&value, offset, 1);
            }

        private:
            size_t m_size = 0;
            T* m_data = nullptr;
        };
    
        template <typename T>
        class DataBlockRef
        {
        public:
            DataBlockRef(DataBlock<T>& data_block) :
                m_size(data_block.size()),
                m_data(data_block.ptr())
            {
            }

            __device__ size_t size() const
            {
                return m_size;
            }

            __device__ T& operator[](const size_t i)
            {
                return m_data[i];
            }

            __device__ const T& operator[](const size_t i) const
            {
                return m_data[i];
            }

            __device__ T* begin()
            {
                return size() == 0 ? nullptr : &(*this)[0];
            }

            __device__ const T* begin() const
            {
                return size() == 0 ? nullptr : &(*this)[0];
            }

            __device__ T* end()
            {
                return begin() + size();
            }

            __device__ const T* end() const
            {
                return begin() + size();
            }

        private:
            size_t m_size;
            T* m_data;
        };
    
        template <typename T>
        class ConstDataBlockRef
        {
        public:
            ConstDataBlockRef(const DataBlock<T>& data_block) :
                m_size(data_block.size()),
                m_data(data_block.ptr())
            {
            }

            __device__ size_t size() const
            {
                return m_size;
            }

            __device__ const T& operator[](const size_t i) const
            {
                return m_data[i];
            }

            __device__ const T* begin() const
            {
                return size() == 0 ? nullptr : &(*this)[0];
            }

            __device__ const T* end() const
            {
                return begin() + size();
            }

        private:
            size_t m_size;
            const T* m_data;
        };
    
        template <typename T, size_t rank>
        class Array
        {
        public:
            CUDU_STATIC_ASSERT(rank > 0);

            static Array<T, 1> from_ptr(const T* data, const size_t size)
            {
                CUDU_STATIC_ASSERT(rank == 1);

                const Shape1D shape(size);
                Array<T, 1> array(shape);
                array.block().upload_all(data);
                return array;
            }

            explicit Array(const Shape<rank>& shape) :
                m_shape(shape),
                m_data(size())
            {
            }

            Array(const Shape<rank>& shape, T fill_value) : Array(shape)
            {
                upload(host::Array<T, rank>(shape, fill_value));
            }

            Array() : Array(Shape<rank>())
            {
            }

            const Shape<rank>& shape() const
            {
                return m_shape;
            }

            DataBlock<T>& block()
            {
                return m_data;
            }

            const DataBlock<T>& block() const
            {
                return m_data;
            }

            size_t size() const
            {
                return std::accumulate(shape().begin(), shape().end(), 1, std::multiplies<size_t>());
            }

            size_t size_bytes() const
            {
                return size() * sizeof(T);
            }

            host::Array<T, rank> download() const
            {
                host::Array<T, rank> result(shape());
                block().download_all(result.begin());
                return result;
            }

            T download_single(const size_t i, const size_t j) const
            {
                CUDU_DEBUG_ASSERT(i < shape()[0]);
                CUDU_DEBUG_ASSERT(j < shape()[1]);

                return block().download_single(block_index(i, j));
            }

            T download_single(const size_t i, const size_t j, const size_t k) const
            {
                CUDU_DEBUG_ASSERT(i < shape()[0]);
                CUDU_DEBUG_ASSERT(j < shape()[1]);
                CUDU_DEBUG_ASSERT(k < shape()[2]);

                return block().download_single(block_index(i, j, k));
            }

            void upload(const host::Array<T, rank>& source)
            {
                (*this) = cudu::device::Array<T, rank>(source.shape());
                block().upload_all(source.begin());
            }

            void upload_single(const size_t i, const size_t j, const T value)
            {
                CUDU_DEBUG_ASSERT(i < shape()[0]);
                CUDU_DEBUG_ASSERT(j < shape()[1]);

                block().upload_single(value, block_index(i, j));
            }

            void upload_single(const size_t i, const size_t j, const size_t k, const T value) const
            {
                CUDU_DEBUG_ASSERT(i < shape()[0]);
                CUDU_DEBUG_ASSERT(j < shape()[1]);
                CUDU_DEBUG_ASSERT(k < shape()[2]);

                block().upload_single(value, block_index(i, j, k));
            }

            void clear()
            {
                *this = Array<T, rank>(Shape<rank>());
            }

        private:
            size_t block_index(const size_t i, const size_t j) const
            {
                return i * m_shape[1] + j;
            }

            size_t block_index(const size_t i, const size_t j, const size_t k) const
            {
                return (i * m_shape[1] + j) * m_shape[2] + k;
            }

            Shape<rank> m_shape = Shape<rank>();
            DataBlock<T> m_data = DataBlock<T>(0);
        };

        template <typename T>
        using Array1D = Array<T, 1>;
        template <typename T>
        using Array2D = Array<T, 2>;
        template <typename T>
        using Array3D = Array<T, 3>;

        template <typename T, size_t rank>
        class ArrayRef
        {
        public:
            static_assert(rank > 0, "arrays require rank > 0");

            ArrayRef(Array<T, rank>& array) : 
                m_shape(array.shape()),
                m_data(array.block())
            {
            }

            __device__ const Shape<rank>& shape() const
            {
                return m_shape;
            }

            __device__ DataBlockRef<T>& block()
            {
                return m_data;
            }

            __device__ const DataBlockRef<T>& block() const
            {
                return m_data;
            }

            __device__ size_t size() const
            {
                return block().size();
            }

            __device__ T& operator[](const size_t i)
            {
                return block()[i];
            }

            __device__ const T& operator[](const size_t i) const
            {
                return block()[i];
            }

            __device__ T& operator()(const size_t i, const size_t j)
            {
                return block()[block_index(i, j)];
            }

            __device__ const T& operator()(const size_t i, const size_t j) const
            {
                return block()[block_index(i, j)];
            }

            __device__ T& operator()(const size_t i, const size_t j, const size_t k)
            {
                return block()[block_index(i, j, k)];
            }

            __device__ const T& operator()(const size_t i, const size_t j, const size_t k) const
            {
                return block()[block_index(i, j, k)];
            }

            __device__ T* begin()
            {
                return block().begin();
            }

            __device__ const T* begin() const
            {
                return block().begin();
            }

            __device__ T* end()
            {
                return block().end();
            }

            __device__ const T* end() const
            {
                return block().end();
            }

        private:
            __device__ size_t block_index(const size_t i, const size_t j) const
            {
                return i * shape()[1] + j;
            }

            __device__ size_t block_index(const size_t i, const size_t j, const size_t k) const
            {
                return (i * shape()[1] + j) * shape()[2] + k;
            }

            Shape<rank> m_shape;
            DataBlockRef<T> m_data;
        };

        template <typename T>
        using ArrayRef1D = ArrayRef<T, 1>;
        template <typename T>
        using ArrayRef2D = ArrayRef<T, 2>;
        template <typename T>
        using ArrayRef3D = ArrayRef<T, 3>;

        template <typename T, size_t rank>
        class ConstArrayRef
        {
        public:
            static_assert(rank > 0, "arrays require rank > 0");

            ConstArrayRef(const Array<T, rank>& array) : 
                m_shape(array.shape()),
                m_data(array.block())
            {
            }

            __device__ const Shape<rank>& shape() const
            {
                return m_shape;
            }

            __device__ const ConstDataBlockRef<T>& block() const
            {
                return m_data;
            }

            __device__ size_t size() const
            {
                return block().size();
            }

            __device__ const T& operator[](const size_t i) const
            {
                return block()[i];
            }

            __device__ const T& operator()(const size_t i, const size_t j) const
            {
                return block()[block_index(i, j)];
            }

            __device__ const T& operator()(const size_t i, const size_t j, const size_t k) const
            {
                return block()[block_index(i, j, k)];
            }

            __device__ const T* begin() const
            {
                return block().begin();
            }

            __device__ const T* end() const
            {
                return block().end();
            }

        private:
            __device__ size_t block_index(const size_t i, const size_t j) const
            {
                return i * shape()[1] + j;
            }

            __device__ size_t block_index(const size_t i, const size_t j, const size_t k) const
            {
                return (i * shape()[1] + j) * shape()[2] + k;
            }

            Shape<rank> m_shape;
            ConstDataBlockRef<T> m_data;
        };

        template <typename T>
        using ConstArrayRef1D = ConstArrayRef<T, 1>;
        template <typename T>
        using ConstArrayRef2D = ConstArrayRef<T, 2>;
        template <typename T>
        using ConstArrayRef3D = ConstArrayRef<T, 3>;
    }
}
