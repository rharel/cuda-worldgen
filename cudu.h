#pragma once

#include <array>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "cuda_runtime.h"

#define CUDU_ASSERT(expression) \
    cudu::abort_on_error((expression), __FILE__, __LINE__, #expression)

#define CUDU_THREAD_ID() \
    (blockIdx.x * blockDim.x + threadIdx.x)

#define CUDU_THREAD_TOTAL() \
    (gridDim.x * blockDim.x)

namespace cudu 
{
    void abort_on_error(
        cudaError_t status,
        const std::string& file,
        const unsigned line,
        const std::string& message);

    struct Workload
    {
        static Workload for_jobs(
            unsigned nr_jobs,
            unsigned nr_threads_max);

        unsigned nr_threads;
        unsigned nr_blocks;
    };

    namespace host
    {
        template <typename T, size_t rank>
        using Point = std::array<T, rank>;
        template <typename T>
        using Point1D = Point<T, 1>;
        template <typename T>
        using Point2D = Point<T, 2>;
        template <typename T>
        using Point3D = Point<T, 3>;

        template <typename T, size_t rank>
        class Array
        {
        public:
            static_assert(rank > 0, "arrays require rank > 0");

            explicit Array(const std::array<size_t, rank>& shape) :
                m_shape(shape),
                m_data(new T[size()])
            {
            }

            const Point<T, rank>& shape() const
            {
                return m_shape;
            }

            size_t shape(size_t dimension_index) const
            {
                return m_shape[dimension_index];
            }

            size_t size() const
            {
                return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<size_t>());
            }

            size_t size_bytes() const
            {
                return size() * sizeof(T);
            }

            T& operator[](const size_t i)
            {
                return m_data[i];
            }

            const T& operator[](const size_t i) const
            {
                return m_data[i];
            }

            T& operator()(const size_t i, const size_t j, std::enable_if_t<rank == 2, int> gate = 0)
            {
                return m_data[data_index(i, j)];
            }

            const T& operator()(const size_t i, const size_t j) const
            {
                return m_data[data_index(i, j)];
            }

            T& operator()(const size_t i, const size_t j, const size_t k)
            {
                return m_data[data_index(i, j, k)];
            }

            const T& operator()(const size_t i, const size_t j, const size_t k) const
            {
                return m_data[data_index(i, j, k)];
            }

            T* begin()
            {
                return ptr();
            }

            const T* begin() const
            {
                return ptr();
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
                return i * m_shape[1] + j;
            }

            size_t data_index(const size_t i, const size_t j, const size_t k) const
            {
                return (i * m_shape[1] + j) * m_shape[2] + k;
            }

            std::array<size_t, rank> m_shape;
            std::unique_ptr<T[]> m_data;
        };

        template <typename T>
        using Array1D = Array<T, 1>;
        template <typename T>
        using Array2D = Array<T, 2>;
        template <typename T>
        using Array3D = Array<T, 3>;
    }

    template <typename T>
    class DataBlock
    {
    public:
        template <typename Container_t>
        static DataBlock from_container(const Container_t& container)
        {
            std::vector<T> vector(container.begin(), container.end());
            DataBlock block(vector.size());
            block.upload_all(vector.data());
            return block;
        }

        explicit DataBlock(const size_t size = 0) : 
            m_size(size),
            m_device_data(nullptr)
        { 
            if (m_size > 0)
            {
                CUDU_ASSERT(cudaMalloc((void**)&m_device_data, size_bytes()));
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
            m_size = other.m_size;
            m_device_data = other.m_device_data;

            other.m_size = 0;
            other.m_device_data = nullptr;

            return *this;
        }

        ~DataBlock()
        {
            if (m_size > 0)
            {
                CUDU_ASSERT(cudaFree(m_device_data));
                m_size = 0;
                m_device_data = nullptr;
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
            return m_device_data;
        }

        const T* ptr() const
        {
            return m_device_data;
        }

        void download(T* const host, const size_t offset, const size_t extent) const 
        {
            CUDU_ASSERT(cudaMemcpy(
                host, 
                m_device_data + offset, 
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
            CUDU_ASSERT(cudaMemcpy(
                m_device_data + offset, 
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
        size_t m_size;
        T* m_device_data;
    };

    template <typename T>
    class DataBlockRef
    {
    public:
        DataBlockRef(DataBlock<T>& data_block) : 
            m_size(data_block.size()), 
            m_device_data(data_block.ptr())
        {
        }

        __device__ size_t size() const
        {
            return m_size;
        }

        __device__ T& operator[](const size_t i)
        {
            return m_device_data[i];
        }

        __device__ const T& operator[](const size_t i) const
        {
            return m_device_data[i];
        }

        __device__ T* begin()
        {
            return size() == 0 ? nullptr : &m_device_data[0];
        }

        __device__ const T* begin() const
        {
            return size() == 0 ? nullptr : &m_device_data[0];
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
        T* m_device_data;
    };

    template <typename T>
    class ConstDataBlockRef
    {
    public:
        ConstDataBlockRef(const DataBlock<T>& data_block) :
            m_size(data_block.size()),
            m_device_data(data_block.ptr())
        {
        }

        __device__ size_t size() const
        {
            return m_size;
        }

        __device__ const T& operator[](const size_t i) const
        {
            return m_device_data[i];
        }

        __device__ const T* begin() const
        {
            return size() == 0 ? nullptr : &m_device_data[0];
        }

        __device__ const T* end() const
        {
            return begin() + size();
        }

    private:
        size_t m_size;
        const T* m_device_data;
    };

    template <typename T, size_t rank>
    class Array
    {
    public:
        static_assert(rank > 0, "arrays require rank > 0");

        template <typename Container_t>
        static Array<T, 1> from_container(const Container_t& container)
        {
            std::vector<T> vector(container.begin(), container.end());
            Array<T, 1> array({vector.size()});
            array.block().upload_all(vector.data());
            return array;
        }

        explicit Array(const std::array<size_t, rank>& shape) :
            m_data_block(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()))
        {
            std::copy(shape.begin(), shape.end(), m_shape);
        }

        DataBlock<T>& block()
        {
            return m_data_block;
        }

        const DataBlock<T>& block() const
        {
            return m_data_block;
        }

        size_t size() const
        {
            return m_data_block.size();
        }

        size_t size_bytes() const
        {
            return m_data_block.size_bytes();
        }

        size_t shape(size_t dimension_index) const
        {
            return m_shape[dimension_index];
        }

        host::Array<T, rank> download() const
        {
            std::array<size_t, rank> shape;
            std::copy(m_shape, m_shape + rank, shape.begin());
            
            host::Array<T, rank> result(shape);
            m_data_block.download_all(result.ptr());

            return result;
        }

        T download_single(const size_t i, const size_t j) const
        {
            return m_data_block.download_single(block_index(i, j));
        }

        T download_single(const size_t i, const size_t j, const size_t k) const
        {
            return m_data_block.download_single(block_index(i, j, k));
        }

        void upload_single(const size_t i, const size_t j, const T value)
        {
            m_data_block.upload_single(value, block_index(i, j));
        }

    private:
        __host__ size_t block_index(const size_t i, const size_t j) const
        {
            return i * m_shape[1] + j;
        }

        __host__ size_t block_index(const size_t i, const size_t j, const size_t k) const
        {
            return (i * m_shape[1] + j) * m_shape[2] + k;
        }

        DataBlock<T> m_data_block;
        size_t m_shape[rank];
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

        ArrayRef(Array<T, rank>& array) : m_data_block(array.block())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                m_shape[i] = array.shape(i);
            }
        }

        __device__ size_t size() const
        {
            return m_data_block.size();
        }

        __device__ size_t shape(size_t dimension_index) const
        {
            return m_shape[dimension_index];
        }

        __device__ T& operator[](const size_t i)
        {
            return m_data_block[i];
        }

        __device__ const T& operator[](const size_t i) const
        {
            return m_data_block[i];
        }

        __device__ T& operator()(const size_t i, const size_t j)
        {
            return m_data_block[block_index(i, j)];
        }

        __device__ const T& operator()(const size_t i, const size_t j) const
        {
            return m_data_block[block_index(i, j)];
        }

        __device__ T& operator()(const size_t i, const size_t j, const size_t k)
        {
            return m_data_block[block_index(i, j, k)];
        }

        __device__ const T& operator()(const size_t i, const size_t j, const size_t k) const
        {
            return m_data_block[block_index(i, j, k)];
        }

        __device__ T* begin()
        {
            return m_data_block.begin();
        }

        __device__ const T* begin() const
        {
            return m_data_block.begin();
        }

        __device__ T* end()
        {
            return m_data_block.end();
        }

        __device__ const T* end() const
        {
            return m_data_block.end();
        }

    private:
        __device__ size_t block_index(const size_t i, const size_t j) const
        {
            return i * m_shape[1] + j;
        }

        __device__ size_t block_index(const size_t i, const size_t j, const size_t k) const
        {
            return (i * m_shape[1] + j) * m_shape[2] + k;
        }

        DataBlockRef<T> m_data_block;
        size_t m_shape[rank];
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

        ConstArrayRef(const Array<T, rank>& array) : m_data_block(array.block())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                m_shape[i] = array.shape(i);
            }
        }

        __device__ size_t size() const
        {
            return m_data_block.size();
        }

        __device__ size_t shape(size_t dimension_index) const
        {
            return m_shape[dimension_index];
        }

        __device__ const T& operator[](const size_t i) const
        {
            return m_data_block[i];
        }

        __device__ const T& operator()(const size_t i, const size_t j) const
        {
            return m_data_block[block_index(i, j)];
        }

        __device__ const T& operator()(const size_t i, const size_t j, const size_t k) const
        {
            return m_data_block[block_index(i, j, k)];
        }

        __device__ const T* begin() const
        {
            return m_data_block.begin();
        }

        __device__ const T* end() const
        {
            return m_data_block.end();
        }

    private:
        __device__ size_t block_index(const size_t i, const size_t j) const
        {
            return i * m_shape[1] + j;
        }

        __device__ size_t block_index(const size_t i, const size_t j, const size_t k) const
        {
            return (i * m_shape[1] + j) * m_shape[2] + k;
        }

        ConstDataBlockRef<T> m_data_block;
        size_t m_shape[rank];
    };

    template <typename T>
    using ConstArrayRef1D = ConstArrayRef<T, 1>;
    template <typename T>
    using ConstArrayRef2D = ConstArrayRef<T, 2>;
    template <typename T>
    using ConstArrayRef3D = ConstArrayRef<T, 3>;
}
