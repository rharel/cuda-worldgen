/**
 * Copyright (c) 2020 Raoul Harel 
 * All rights reserved
 */

#define _USE_MATH_DEFINES

#include <algorithm>
#include <math.h>
#include <random>

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "worldgen.h"

constexpr size_t BATCH_SIZE = 512 * 512;

template <typename T>
__device__ T lerp(T a, T b, T t)
{
    static_assert(
        std::is_floating_point_v<T>,
        "interpolation requires floating point type"
    );
    return (1 - t) * a + t * b;
}

__host__ __device__ float distance_squared_2d(
    float x0, float y0,
    float x1, float y1)
{
    return (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
}

__host__ __device__ float distance_2d(
    float x0, float y0,
    float x1, float y1)
{
    return sqrtf(distance_squared_2d(x0, y0, x1, y1));
}

__device__ void unravel_index(
    const size_t index, 
    const size_t extent,
    size_t& i, 
    size_t& j)
{
    i = index / extent;
    j = index % extent;
}

__device__ size_t wrap(
    const int i, 
    const int extent)
{
    return (i + extent) % extent;
}

__device__ float random_uniform_in_range(
    const float min,
    const float max,
    curandStatePhilox4_32_10& state)
{
    return min + (max - min) * curand_uniform(&state);
}

__global__ void rng_state_setup_kernel(
    const size_t job_offset,
    const unsigned long seed,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> state)
{
    const size_t i = job_offset + CUDU_THREAD_ID();
    
    if (i < state.size())
    {
        curand_init(seed, i, 0, &state[i]);
    }
}

cudu::device::Array2D<curandStatePhilox4_32_10> rng_state(
    const cudu::Shape2D& shape,
    const unsigned long seed)
{
    cudu::device::Array2D<curandStatePhilox4_32_10> result(shape);
    CUDU_LAUNCH_BATCHES(rng_state_setup_kernel, result.size(), BATCH_SIZE, seed, result);
    return result;
}

template <typename T>
__host__ __device__ bool within(
    const cudu::Point2D<T>& pixel,
    const cudu::Shape2D& extents)
{
    return (
        0 <= pixel[0] && pixel[0] < extents[0] &&
        0 <= pixel[1] && pixel[1] < extents[1]
    );
}

__global__ void vicinity_lookup_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<bool> mask,
    const float radius,
    const unsigned nr_samples,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::device::ArrayRef3D<size_t> result)
{
    if (job_offset + CUDU_THREAD_ID() >= mask.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), mask.shape()[1], i, j);
    
    result(i, j, 2) = false;

    if (mask(i, j)) 
    {
        result(i, j, 0) = i;
        result(i, j, 1) = j;
        result(i, j, 2) = true;
        return;
    }

    float nearest_sample_distance = 1e30;

    const float magnitude_step = radius / nr_samples;
    const float angle_step = 2 * M_PI / nr_samples;

    for (size_t x = 0; x < nr_samples; ++x)
    {
        const float magnitude = random_uniform_in_range(
            (x + 0) * magnitude_step, 
            (x + 1) * magnitude_step, 
            rng_state(i, j)
        );
        for (size_t y = 0; y < nr_samples; ++y)
        {
            const float angle = random_uniform_in_range(
                (y + 0) * angle_step,
                (y + 1) * angle_step,
                rng_state(i, j)
            );

            const cudu::Point2D<float> sample(
                i + magnitude * cosf(angle),
                j + magnitude * sinf(angle)
            );

            if (within(sample, mask.shape()) &&
                mask(sample[0], sample[1]) &&
                distance_squared_2d(i, j, sample[0], sample[1]) < nearest_sample_distance)
            {
                nearest_sample_distance = distance_squared_2d(i, j, sample[0], sample[1]);
                result(i, j, 0) = sample[0];
                result(i, j, 1) = sample[1];
                result(i, j, 2) = true;
            }
        }
        if (result(i, j, 2))
        {
            return;
        }
    }
}

cudu::device::Array3D<size_t> vicinity_lookup(
    const cudu::device::Array2D<bool>& mask,
    const worldgen::VicinityLookupParams& lookup_params)
{
    cudu::device::Array2D<curandStatePhilox4_32_10> device_rng = rng_state(mask.shape(), lookup_params.rng_seed);
    cudu::device::Array3D<size_t> result(cudu::Shape3D(mask.shape()[0], mask.shape()[1], 3));
    CUDU_LAUNCH_BATCHES(
        vicinity_lookup_kernel,
        mask.size(),
        BATCH_SIZE,
        mask,
        lookup_params.radius,
        lookup_params.nr_samples,
        device_rng,
        result
    );
    return result;
}

__global__ void altitude_diamond_kernel(
    const size_t job_offset,
    const float noise,
    const size_t stride,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::device::ArrayRef2D<float> result)
{
    size_t i; 
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), result.shape()[1] / stride, i, j);

    const size_t offset = stride / 2;
    i = offset + i * stride;
    j = offset + j * stride;
    
    if (i >= result.shape()[0] ||
        j >= result.shape()[1]) {
        return;
    }

    const float noise_factor = random_uniform_in_range(-noise, +noise, rng_state[CUDU_THREAD_ID() % rng_state.size()]);

    result(i, j) = (
        result(i - offset, j - offset) +
        result(i - offset, j + offset) +
        result(i + offset, j - offset) +
        result(i + offset, j + offset)
    ) / 4 + noise_factor;
}

__global__ void altitude_square_kernel(
    const size_t job_offset,
    const size_t nr_squares,
    const float noise,
    const size_t stride,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::device::ArrayRef2D<float> result)
{
    if (job_offset + CUDU_THREAD_ID() >= nr_squares)
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(
        (job_offset + CUDU_THREAD_ID()) % (nr_squares / 2), 
        result.shape()[1] / stride, 
        i, j
    );

    const size_t offset = stride / 2;
    i = i * stride;
    j = offset + j * stride;

    if (job_offset + CUDU_THREAD_ID() >= nr_squares / 2)
    {
        const size_t temp = i;
        i = j;
        j = temp;
    }

    const size_t nr_rows = result.shape()[0];
    const size_t nr_cols = result.shape()[1];

    const float noise_factor = random_uniform_in_range(-noise, +noise, rng_state[CUDU_THREAD_ID() % rng_state.size()]);

    result(i, j) = (
        result(wrap(int(i) - int(stride) / 2, nr_rows), j) +
        result(wrap(int(i) + int(stride) / 2, nr_rows), j) +
        result(i, wrap(int(j) - int(stride) / 2, nr_cols)) +
        result(i, wrap(int(j) + int(stride) / 2, nr_cols))
    ) / 4 + noise_factor;
}

cudu::device::Array2D<float> worldgen::altitude(
    const uint32_t max_level_of_detail,
    const float noise_initial,
    const float noise_scale_factor,
    const float bias,
    const cudu::device::Array2D<float>& bias_map,
    const float bias_map_shift,
    const float bias_map_stretch,
    const uint64_t rng_seed)
{
    const size_t edge_size = std::pow(2, max_level_of_detail) + 1;
    const cudu::Shape2D shape{ edge_size, edge_size };
    cudu::device::Array2D<float> result(shape);
    cudu::device::Array2D<curandStatePhilox4_32_10> device_rng = rng_state(
        cudu::Shape2D(cudu::Workload::max_threads_per_block(), 1), 
        rng_seed
    );
    
    // Initialize corners.
    std::default_random_engine rng(rng_seed);
    std::uniform_real_distribution<float> distrib(-noise_initial / 2,  +noise_initial / 2);

    result.upload_single(0, 0, distrib(rng) + bias);
    result.upload_single(0, edge_size - 1, distrib(rng) + bias);
    result.upload_single(edge_size - 1, 0, distrib(rng) + bias);
    result.upload_single(edge_size - 1, edge_size - 1, distrib(rng) + bias);
    
    float noise = noise_initial;
    
    for (size_t lod = 0; lod < max_level_of_detail; ++lod) {
        
        noise *= noise_scale_factor;

        const size_t stride = (edge_size - 1) >> lod;

        // Diamond step.
        const unsigned nr_diamonds = std::pow(4, lod);
        CUDU_LAUNCH_BATCHES(altitude_diamond_kernel, nr_diamonds, BATCH_SIZE, noise, stride, device_rng, result);

        // Square step.
        const unsigned nr_squares = 2 * std::sqrt(nr_diamonds) * (std::sqrt(nr_diamonds) + 1);
        CUDU_LAUNCH_BATCHES(altitude_square_kernel, nr_squares, BATCH_SIZE, nr_squares, noise, stride, device_rng, result);
    }
    return result;
}

__global__ void cell_distance_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef1D<worldgen::MaterialCell> cells,
    cudu::device::ArrayRef3D<float> cell_distance)
{
    if (job_offset + CUDU_THREAD_ID() >= 
        cell_distance.shape()[0] * cell_distance.shape()[1])
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), cell_distance.shape()[1], i, j);

    extern __shared__ worldgen::MaterialCell cells_local[];

    const size_t iThread = threadIdx.x;

    if (iThread < cells.size())
    {
        cells_local[iThread] = cells[iThread];
    }
    __syncthreads();

    for (size_t k = 1; k < cells.size(); ++k)
    {
        cell_distance(i, j, k) = distance_2d(i, j, cells_local[k].row, cells_local[k].col);
    }
}

__global__ void material_mixture_kernal(
    const size_t job_offset,
    const cudu::device::ConstArrayRef1D<worldgen::MaterialCell> cells,
    const cudu::device::ConstArrayRef3D<float> cell_distance,
    const unsigned blur,
    cudu::device::ArrayRef3D<float> mixtures)
{
    if (job_offset + CUDU_THREAD_ID() >= mixtures.shape()[0] * mixtures.shape()[1])
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), mixtures.shape()[1], i, j);

    float distance_sum = 0;

    for (size_t m = 0; m < mixtures.shape()[2]; ++m)
    {
        mixtures(i, j, m) = 0;
    }
    for (size_t c = 0; c < cell_distance.shape()[2]; ++c)
    {
        const size_t m = cells[c].material_index;
        mixtures(i, j, m) += powf(cell_distance(i, j, c), blur);
        distance_sum += powf(cell_distance(i, j, c), blur);
    }
    for (size_t m = 0; m < mixtures.shape()[2]; ++m)
    {
        mixtures(i, j, m) = 1 - mixtures(i, j, m) / distance_sum;
    }
}

cudu::device::Array3D<float> worldgen::material(
    const cudu::Shape2D& shape,
    const std::vector<SurfaceMaterial>& materials,
    const unsigned cell_count,
    const unsigned cell_blur,
    const uint64_t rng_seed)
{
    if (materials.size() == 1)
    {
        return cudu::device::Array3D<float>({ shape[0], shape[1], 1 }, 1);
    }

    // Compute cell centers.
    
    std::vector<double> material_likelihoods(materials.size());
    
    for (size_t i = 0; i < materials.size(); ++i) 
    {
        material_likelihoods[i] = materials[i].likelihood;
    }

    std::default_random_engine rng(rng_seed);
    std::discrete_distribution<size_t> material_distrib(
        material_likelihoods.begin(), 
        material_likelihoods.end()
    );
    std::uniform_int_distribution<size_t> row_distrib(0, shape[0]);
    std::uniform_int_distribution<size_t> col_distrib(0, shape[1]);

    std::vector<worldgen::MaterialCell> cells_vec(cell_count);

    for (size_t i = 0; i < cells_vec.size(); ++i)
    {
        cells_vec[i].material_index = material_distrib(rng);
        cells_vec[i].row = row_distrib(rng);
        cells_vec[i].col = col_distrib(rng);
    }

    auto cells = cudu::device::Array1D<worldgen::MaterialCell>::from_ptr(
        cells_vec.data(), 
        cells_vec.size()
    );

    // Compute nearest cells to each pixel.

    cudu::device::Array3D<float> cell_distance({shape[0], shape[1], cells.size()});
    CUDU_LAUNCH_BATCHES_SHARED(cell_distance_kernel, shape[0] * shape[1], BATCH_SIZE, cells.size_bytes(), cells, cell_distance);

    // Compute material mixtures based on cell distance.

    cudu::device::Array3D<float> mixtures({shape[0], shape[1], materials.size()});
    CUDU_LAUNCH_BATCHES(material_mixture_kernal, shape[0] * shape[1], BATCH_SIZE, cells, cell_distance, cell_blur, mixtures);

    return mixtures;
}

__device__ float latitude_from_index(
    const float i,
    const float extent)
{
    return (M_PI / 2) * fabsf(i - 0.5 * extent) / (0.5 * extent);
}

__device__ float degrees(const float radians)
{
    return radians * (180 / M_PI);
}

__global__ void temperature_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> altitude,
    const float latitude_factor,
    const float altitude_factor,
    const float noise,
    const float bias,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::device::ArrayRef2D<float> temperature)
{
    if (job_offset + CUDU_THREAD_ID() >= temperature.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), temperature.shape()[1], i, j);

    const float latitude = latitude_from_index(i, temperature.shape()[0]);
    const float relative_equator_offset = 1 - fabsf(cosf(latitude));
    const float latitude_effect = -latitude_factor * relative_equator_offset;
    const float altitude_effect = -fmaxf(altitude(i, j), 0) * altitude_factor;
    const float noise_factor = random_uniform_in_range(-noise, +noise, rng_state(i, j));
    
    temperature(i, j) = bias + latitude_effect + altitude_effect + noise_factor;
}

cudu::device::Array2D<float> worldgen::temperature(
    const cudu::device::Array2D<float>& altitude,
    const float latitude_factor,
    const float altitude_factor,
    const float noise,
    const float bias,
    const uint64_t rng_seed)
{
    cudu::device::Array2D<curandStatePhilox4_32_10> device_rng = rng_state(altitude.shape(), rng_seed);
    cudu::device::Array2D<float> result(altitude.shape());
    
    CUDU_LAUNCH_BATCHES(
        temperature_kernel,
        result.size(),
        BATCH_SIZE,
        altitude,
        latitude_factor,
        altitude_factor,
        noise,
        bias,
        device_rng,
        result
    );

    return result;
}

__global__ void ocean_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> altitude,
    const cudu::device::ConstArrayRef2D<float> temperature,
    const float boil_temperature,
    cudu::device::ArrayRef2D<bool> ocean_mask)
{
    if (job_offset + CUDU_THREAD_ID() >= ocean_mask.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), ocean_mask.shape()[1], i, j);

    ocean_mask(i, j) = (
        altitude(i, j) <= 0 && 
        temperature(i, j) < boil_temperature
    );
}

cudu::device::Array2D<bool> worldgen::ocean_mask(
    const cudu::device::Array2D<float>& altitude,
    const cudu::device::Array2D<float>& temperature,
    const float boil_temperature)
{
    cudu::device::Array2D<bool> result(altitude.shape());
    CUDU_LAUNCH_BATCHES(ocean_kernel, result.size(), BATCH_SIZE, altitude, temperature, boil_temperature, result);
    return result;
}

__global__ void ice_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> temperature,
    const cudu::device::ConstArrayRef2D<bool> ocean_mask,
    const float ocean_freeze_temperature,
    cudu::device::ArrayRef2D<bool> ice_mask)
{
    if (job_offset + CUDU_THREAD_ID() >= ice_mask.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), ice_mask.shape()[1], i, j);

    ice_mask(i, j) = (
        ocean_mask(i, j) ?
        temperature(i, j) <= ocean_freeze_temperature : 
        temperature(i, j) <= 0
    );
}

cudu::device::Array2D<bool> worldgen::ice_mask(
    const cudu::device::Array2D<float>& temperature,
    const cudu::device::Array2D<bool>& ocean_mask,
    const float ocean_freeze_temperature)
{
    cudu::device::Array2D<bool> result(temperature.shape());
    CUDU_LAUNCH_BATCHES(ice_kernel, result.size(), BATCH_SIZE, temperature, ocean_mask, ocean_freeze_temperature, result);
    return result;
}

__device__ float range_based_weight(
    const float value,
    const worldgen::Range& range)
{
    return 1 - fminf(range.distance_relative_to(value), 1);
}

__global__ void precipitation_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> temperature,
    const cudu::device::ConstArrayRef2D<float> ocean_distance,
    const float ocean_distance_factor,
    const cudu::device::ConstArrayRef1D<worldgen::ClimateZone> climate_zones,
    cudu::device::ArrayRef2D<float> precipitation)
{
    if (job_offset + CUDU_THREAD_ID() >= precipitation.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), precipitation.shape()[1], i, j);

    const float latitude = degrees(latitude_from_index(i, precipitation.shape()[0]));
    
    precipitation(i, j) = 0;
    float weight_sum = 0;

    for (const worldgen::ClimateZone& zone: climate_zones)
    {
        const float temperature_match = range_based_weight(
            temperature(i, j), 
            zone.temperature_range
        );
        const float contribution = lerp(
            zone.precipitation_range.min,
            zone.precipitation_range.max,
            temperature_match
        );
        const float weight = 1 - fminf(zone.latitude_range.distance_relative_to(latitude) / 0.5, 1);
        precipitation(i, j) += weight * contribution;
        weight_sum += weight;
    }

    precipitation(i, j) /= weight_sum;

    const float ocean_factor = fminf(ocean_distance(i, j) * ocean_distance_factor, 1);

    precipitation(i, j) *= (1 - ocean_factor);
}

__global__ void distance_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef3D<size_t> vicinity,
    cudu::device::ArrayRef2D<float> distance,
    cudu::device::ArrayRef2D<bool> mask)
{
    if (job_offset + CUDU_THREAD_ID() >= distance.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), distance.shape()[1], i, j);

    distance(i, j) = (
        vicinity(i, j, 2) ?
        distance_2d(i, j, vicinity(i, j, 0), vicinity(i, j, 1)) :
        1e30
    );
    mask(i, j) = vicinity(i, j, 2);
}

__global__ void smoothing_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> data,
    const cudu::device::ConstArrayRef2D<bool> mask,
    const float variance,
    cudu::device::ArrayRef2D<float> result)
{
    if (job_offset + CUDU_THREAD_ID() >= data.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), data.shape()[1], i, j);

    const size_t kernel_size = ceilf(6 * sqrtf(variance));

    float convolution = 0;

    for (size_t k0 = 0; k0 < kernel_size; ++k0)
    {
        for (size_t k1 = 0; k1 < kernel_size; ++k1)
        {
            const cudu::Point2D<float> sample(
                i - kernel_size / 2 + k0,
                j - kernel_size / 2 + k1
            );
            if (within(sample, data.shape()) && mask(sample[0], sample[1]))
            {
                const float x = float(i) - float(sample[0]);
                const float y = float(j) - float(sample[1]);
                const float weight = 1.0f / sqrtf(2 * M_PI * variance) * exp(-(x * x + y * y) / (2 * variance));
                convolution += weight * data(sample[0], sample[1]);
            }
        }
    }
    result(i, j) = convolution;
}

cudu::device::Array2D<float> worldgen::precipitation(
    const cudu::device::Array2D<float>& temperature,
    const cudu::device::Array2D<bool>& ocean_mask,
    const std::vector<ClimateZone>& climate_zones,
    const VicinityLookupParams& ocean_lookup_params,
    const float ocean_distance_smoothing,
    const float ocean_distance_factor)
{
    if (climate_zones.empty())
    {
        return cudu::device::Array2D<float>(temperature.shape(), 0);
    }

    const auto zones = cudu::device::Array1D<ClimateZone>::from_ptr(climate_zones.data(), climate_zones.size());

    cudu::device::Array3D<size_t> ocean_vicinity = vicinity_lookup(ocean_mask, ocean_lookup_params);
    cudu::device::Array2D<float> ocean_distance(ocean_mask.shape());
    cudu::device::Array2D<bool> ocean_distance_mask(ocean_distance.shape());

    CUDU_LAUNCH_BATCHES(
        distance_kernel, 
        temperature.size(),
        BATCH_SIZE, 
        ocean_vicinity, 
        ocean_distance, 
        ocean_distance_mask
    );

    ocean_vicinity.clear();

    cudu::device::Array2D<float> ocean_distance_smooth(ocean_distance.shape());
    
    CUDU_LAUNCH_BATCHES(
        smoothing_kernel, 
        temperature.size(),
        BATCH_SIZE,
        ocean_distance, 
        ocean_distance_mask, 
        ocean_distance_smoothing, 
        ocean_distance_smooth
    );

    ocean_distance.clear();
    ocean_distance_mask.clear();

    cudu::device::Array2D<float> result(temperature.shape());

    CUDU_LAUNCH_BATCHES(
        precipitation_kernel, 
        result.size(),
        BATCH_SIZE,
        temperature,
        ocean_distance_smooth,
        ocean_distance_factor,
        zones,
        result
    );
    return result;
}

__global__ void max_in_block_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> data,
    const size_t block_size,
    cudu::device::ArrayRef3D<size_t> result)
{
    if (job_offset + CUDU_THREAD_ID() >= result.shape()[0] * result.shape()[1])
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), result.shape()[1], i, j);

    const size_t r_min = fminf((i + 0) * block_size, data.shape()[0] - 1);
    const size_t r_max = fminf((i + 1) * block_size, data.shape()[0]);

    const size_t c_min = fminf((j + 0) * block_size, data.shape()[1] - 1);
    const size_t c_max = fminf((j + 1) * block_size, data.shape()[1]);

    float max = data(r_min, c_min);
    size_t max_coords[2] = {r_min, c_min};

    for (size_t r = r_min; r < r_max; ++r)
    {
        for (size_t c = c_min; c < c_max; ++c)
        {
            if (data(r, c) > max)
            {
                max = data(r, c);
                max_coords[0] = r;
                max_coords[1] = c;
            }
        }
    }
    result(i, j, 0) = max_coords[0];
    result(i, j, 1) = max_coords[1];
}

cudu::device::Array3D<size_t> max_in_block(
    const cudu::device::Array2D<float>& data,
    const float block_size)
{
    cudu::device::Array3D<size_t> result({
        size_t(std::ceil(data.shape()[0] / (block_size * data.shape()[0]))), 
        size_t(std::ceil(data.shape()[1] / (block_size * data.shape()[1]))), 
        2
    });
    CUDU_LAUNCH_BATCHES(max_in_block_kernel, result.size() / 2, BATCH_SIZE, data, block_size * data.shape()[0], result);
    return result;
}

void run_river(
    const cudu::Point2D<size_t>& source,
    const cudu::Point2D<size_t>& sink,
    const cudu::host::Array2D<float>& altitude,
    const cudu::host::Array2D<bool>& ocean_mask,
    const cudu::host::Array2D<bool>& ice_mask,
    cudu::host::Array2D<float>& river_depth,
    cudu::host::Array2D<bool>& visited)
{
    constexpr size_t MAX_TRAIL_SIZE = 10000;
    
    if (distance_2d(source[0], source[1], sink[0], sink[1]) > MAX_TRAIL_SIZE)
    {
        return;
    }

    std::vector<cudu::Point2D<size_t>> trail;
    cudu::Point2D<size_t> head = source;
    
    while (trail.size() < MAX_TRAIL_SIZE)
    {
        if (ice_mask(head) || visited(head))
        {
            break;
        }
        if (ocean_mask(head) || river_depth(head) != 0)
        {
            trail.push_back(head);
            break;
        }

        float next_altitude = std::numeric_limits<float>::max();
        cudu::Point2D<size_t> next;

        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                const cudu::Point2D<float> candidate{ 
                    float(head[0] + dx), 
                    float(head[1] + dy) 
                };
                
                if (abs(dx) + abs(dy) != 0 && within(candidate, river_depth.shape()) &&
                    distance_squared_2d(candidate[0], candidate[1], sink[0], sink[1]) <= 
                    distance_squared_2d(head[0], head[1], sink[0], sink[1]) &&
                    altitude(candidate.as<size_t>()) < next_altitude)
                {
                    next_altitude = altitude(candidate.as<size_t>());
                    next = candidate.as<size_t>();
                }
            }
        }
        if (next_altitude == std::numeric_limits<float>::max())
        {
            break;
        }
        river_depth(head) = 1;
        trail.push_back(head);
        visited(head) = true;
        head = next;
    }
    for (auto pixel : trail)
    {
        visited(pixel) = false;
    }
    if (trail.size() < 2)
    {
        for (auto pixel : trail)
        {
            river_depth(pixel) = 0;
        }
        return;
    }
    
    bool done = false;
    unsigned nr_steps = 0;
    constexpr unsigned MAX_NR_STEPS = 1000;

    while (!done && nr_steps < MAX_NR_STEPS)
    {
        done = true;
        for (size_t i = 0; i < trail.size() - 2; ++i) 
        {
            const auto curr = trail[i];
            const auto next = trail[i + 1];
            const auto potential = [&altitude, &river_depth](const cudu::Shape2D& pixel) {
                return altitude(pixel) + river_depth(pixel);
            };
            if (potential(curr) < potential(next)) 
            {
                river_depth(curr) = potential(next) - altitude(curr);
                done = false;
            }
        }
        nr_steps += 1;
    }
}

cudu::device::Array2D<float> worldgen::rivers(
    const cudu::device::Array2D<float>& altitude,
    const cudu::device::Array2D<bool>& ocean_mask,
    const cudu::device::Array2D<bool>& ice_mask,
    const float block_size,
    const VicinityLookupParams& ocean_lookup_params)
{
    const cudu::device::Array3D<size_t> sources = max_in_block(altitude, block_size);
    const cudu::device::Array3D<size_t> sinks = vicinity_lookup(ocean_mask, ocean_lookup_params);

    const cudu::host::Array3D<size_t> h_sources = sources.download();
    const cudu::host::Array3D<size_t> h_sinks = sinks.download();

    const cudu::host::Array2D<float> h_altitude = altitude.download();
    const cudu::host::Array2D<bool> h_ocean_mask = ocean_mask.download();
    const cudu::host::Array2D<bool> h_ice_mask = ice_mask.download();
    
    cudu::host::Array2D<float> h_river_depth(altitude.shape(), 0);
    cudu::host::Array2D<bool> h_visited(h_river_depth.shape(), false);

    for (size_t i = 0; i < sources.shape()[0]; ++i)
    {
        #pragma omp parallel for default(none) shared(h_sources, h_sinks, h_altitude, h_ocean_mask, h_ice_mask, h_river_depth, h_visited)
        for (int j = 0; j < sources.shape()[1]; ++j)
        {
            const cudu::Point2D<size_t> source = {
                h_sources(i, j, 0),
                h_sources(i, j, 1)
            };

            const bool sink_exists = h_sinks(source[0], source[1], 2);

            if (sink_exists)
            {
                const cudu::Point2D<size_t> sink = {
                    h_sinks(source[0], source[1], 0),
                    h_sinks(source[0], source[1], 1)
                };
                run_river(source, sink, h_altitude, h_ocean_mask, h_ice_mask, h_river_depth, h_visited);
            }
        }
    }
    cudu::device::Array2D<float> result(h_river_depth.shape());
    result.block().upload_all(h_river_depth.begin());
    return result;
}

template <typename T>
__global__ void max_kernel(
    const T* data, 
    const size_t size,
    T* output)
{
    extern __shared__ T workspace[];
    
    const size_t iThread = threadIdx.x;
    workspace[iThread] = (
        CUDU_THREAD_ID() < size ?
        data[CUDU_THREAD_ID()] :
        1e-20
    );
    __syncthreads();
    
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (iThread < stride)
        {
            workspace[iThread] = fmaxf(workspace[iThread], workspace[iThread + stride]);
        }
        __syncthreads();
    }
    
    if (iThread == 0)
    {
        output[blockIdx.x] = workspace[0];
    }
}

template <typename T>
T max(const cudu::device::DataBlock<T>& data)
{
    if (data.size() == 1)
    {
        return data.download_single(0);
    }

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    const unsigned max_nr_threads = device_props.maxThreadsPerBlock;
    
    size_t size = data.size();
    unsigned nr_blocks = std::max<unsigned>(std::ceil(float(size) / max_nr_threads), 1);
    const unsigned shared_memory_bytes = max_nr_threads * sizeof(T);
    
    cudu::device::DataBlock<T> buffer_1(nr_blocks);
    cudu::device::DataBlock<T> buffer_2(nr_blocks);
    
    const cudu::device::DataBlock<T>* input = &data;
    cudu::device::DataBlock<T>* output = &buffer_1;

    while (size > 1)
    {
        max_kernel<<<nr_blocks, max_nr_threads, shared_memory_bytes>>>(
            input->ptr(), 
            size, 
            output->ptr()
        );
        CUDU_CHECK(cudaDeviceSynchronize());

        size = nr_blocks;
        nr_blocks = std::max<unsigned>(std::ceil(float(size) / max_nr_threads), 1);

        if (size > 1)
        {
            input = output;
            output = output == &buffer_1 ? &buffer_2 : &buffer_1;
        }
    }
    
    T result = output->download_single(0);

    return result;
}

template <typename T>
__global__ void min_kernel(
    const T* data,
    const size_t size,
    T* output)
{
    extern __shared__ T workspace[];

    const size_t iThread = threadIdx.x;
    workspace[iThread] = (
        CUDU_THREAD_ID() < size ?
        data[CUDU_THREAD_ID()] :
        1e+20
    );
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (iThread < stride)
        {
            workspace[iThread] = fminf(workspace[iThread], workspace[iThread + stride]);
        }
        __syncthreads();
    }

    if (iThread == 0)
    {
        output[blockIdx.x] = workspace[0];
    }
}

template <typename T>
T min(const cudu::device::DataBlock<T>& data)
{
    if (data.size() == 1)
    {
        return data.download_single(0);
    }

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    const unsigned max_nr_threads = device_props.maxThreadsPerBlock;

    size_t size = data.size();
    unsigned nr_blocks = std::max<unsigned>(std::ceil(float(size) / max_nr_threads), 1);
    const unsigned shared_memory_bytes = max_nr_threads * sizeof(T);

    cudu::device::DataBlock<T> buffer_1(nr_blocks);
    cudu::device::DataBlock<T> buffer_2(nr_blocks);

    const cudu::device::DataBlock<T>* input = &data;
    cudu::device::DataBlock<T>* output = &buffer_1;

    while (size > 1)
    {
        min_kernel<<<nr_blocks, max_nr_threads, shared_memory_bytes>>>(
            input->ptr(),
            size,
            output->ptr()
        );
        CUDU_CHECK(cudaDeviceSynchronize());

        size = nr_blocks;
        nr_blocks = std::max<unsigned>(std::ceil(float(size) / max_nr_threads), 1);

        if (size > 1)
        {
            input = output;
            output = output == &buffer_1 ? &buffer_2 : &buffer_1;
        }
    }

    T result = output->download_single(0);

    return result;
}

template <typename T>
__device__ T lerp(
    T x0, T y0, 
    T x1, T y1, 
    T x)
{
    static_assert(
        std::is_floating_point_v<T>, 
        "linear interpolation requires floating point type"
    );
    const T t = (x - x0) / (x1 - x0);
    return lerp(y0, y1, t);
}

__device__ worldgen::Rgb lerp_rgb(
    const worldgen::Rgb& from,
    const worldgen::Rgb& to,
    const double t)
{
    return worldgen::Rgb{
        unsigned char(lroundf(lerp<double>(from.r, to.r, t))),
        unsigned char(lroundf(lerp<double>(from.g, to.g, t))),
        unsigned char(lroundf(lerp<double>(from.b, to.b, t)))
    };
}

__device__ worldgen::Rgb ease_out_rgb(
    const worldgen::Rgb& from,
    const worldgen::Rgb& to,
    const double t,
    const double power)
{
    return lerp_rgb(from, to, 1 - pow(1 - t, power));
}

__global__ void image_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> altitude,
    const float altitude_min,
    const float altitude_max,
    const cudu::device::ConstArrayRef3D<float> material,
    const cudu::device::ConstArrayRef2D<float> temperature,
    const cudu::device::ConstArrayRef2D<bool> ocean_mask,
    const cudu::device::ConstArrayRef2D<bool> ice_mask,
    const cudu::device::ConstArrayRef2D<float> precipitation,
    const cudu::device::ConstArrayRef2D<float> river_depth,
    const cudu::device::ConstArrayRef1D<worldgen::ClimateZone> climate_zones,
    const worldgen::RenderingOptions rendering_options,
    cudu::device::ArrayRef3D<unsigned char> image_bgr)
{
    if (job_offset + CUDU_THREAD_ID() >= altitude.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(job_offset + CUDU_THREAD_ID(), image_bgr.shape()[1], i, j);

    // Debug

    /*const worldgen::Rgb rainfall = lerp_rgb(
        worldgen::Rgb{ 255, 255, 255 },
        worldgen::Rgb{ 0, 0, 255 },
        fminf(precipitation(i, j) / 3000, 1)
    );
    image_bgr(i, j, 0) = rainfall.b;
    image_bgr(i, j, 1) = rainfall.g;
    image_bgr(i, j, 2) = rainfall.r;
    return;*/

    // Compute initial color based on material composition.

    worldgen::Rgb ground_color{0, 0, 0};

    for (size_t k = 0; k < material.shape()[2]; ++k)
    {
        const float weight = material(i, j, k);
        const worldgen::MaterialVisual material_visual = rendering_options.materials[k];
        const worldgen::Rgb material_color = lerp_rgb(
            material_visual.low_altitude,
            material_visual.high_altitude,
            altitude(i, j) / altitude_max
        );
        ground_color.r += weight * material_color.r;
        ground_color.g += weight * material_color.g;
        ground_color.b += weight * material_color.b;
    }

    // Identify dominant climate zone (if any).

    const float latitude = degrees(latitude_from_index(i, altitude.shape()[0]));
    worldgen::ClimateZone climate_zone[2];
    float climate_zone_score[2] = { 0, 0 };  // Invariant: score[0] >= score[1]

    for (const worldgen::ClimateZone& candidate_zone : climate_zones)
    {
        const float latitude_distance = candidate_zone.latitude_range.distance_relative_to(latitude);
        const float latitude_match = 1 - fminf(latitude_distance, 1);

        const float precipitation_distance = candidate_zone.precipitation_range.distance_relative_to(precipitation(i, j));
        const float precipitation_match = 1 - fminf(precipitation_distance, 1);

        const float temperature_distance = candidate_zone.temperature_range.distance_relative_to(temperature(i, j));
        const float temperature_match = 1 - fminf(temperature_distance, 1);

        const float candidate_zone_score = latitude_match * precipitation_match * temperature_match;

        if (candidate_zone_score > climate_zone_score[0])
        {
            climate_zone_score[1] = climate_zone_score[0];
            climate_zone[1] = climate_zone[0];

            climate_zone_score[0] = candidate_zone_score;
            climate_zone[0] = candidate_zone;
        }
        else if (candidate_zone_score > climate_zone_score[1])
        {
            climate_zone_score[1] = candidate_zone_score;
            climate_zone[1] = candidate_zone;
        }
    }

    worldgen::Rgb surface_color = ground_color;

    if (climate_zone_score[0] > 0)
    {
        const float latitude_distance[2] = {
            climate_zone[0].latitude_range.distance_relative_to(latitude),
            climate_zone[1].latitude_range.distance_relative_to(latitude)
        };
        const float latitude_match[2] = {
            1 - fminf(latitude_distance[0], 1),
            1 - fminf(latitude_distance[1], 1)
        };

        const float precipitation_distance[2] = {
            climate_zone[0].precipitation_range.distance_relative_to(precipitation(i, j)),
            climate_zone[1].precipitation_range.distance_relative_to(precipitation(i, j)) 
        };
        const float precipitation_match[2] = {
            1 - fminf(precipitation_distance[0], 1),
            1 - fminf(precipitation_distance[1], 1)
        };
        
        const float temperature_distance[2] = {
            climate_zone[0].temperature_range.distance_relative_to(temperature(i, j)),
            climate_zone[1].temperature_range.distance_relative_to(temperature(i, j)) 
        };
        const float temperature_match[2] = {
            1 - fminf(temperature_distance[0], 1),
            1 - fminf(temperature_distance[1], 1) 
        };

        const float climate_zone_density[2] = {
            /*latitude_match[0] * */ precipitation_match[0] * temperature_match[0],
            /*latitude_match[1] * */ precipitation_match[1] * temperature_match[1]
        };

        const worldgen::Rgb climate_zone_colors[2] = {
            lerp_rgb(
                climate_zone[0].visual_sparse,
                climate_zone[0].visual_dense,
                climate_zone_density[0]
            ),
            lerp_rgb(
                climate_zone[1].visual_sparse,
                climate_zone[1].visual_dense,
                climate_zone_density[1]
            )
        };
        const worldgen::Rgb climate_zone_color = lerp_rgb(
            climate_zone_colors[0],
            climate_zone_colors[1],
            climate_zone_score[1] / (climate_zone_score[0] + climate_zone_score[1])
        );
        surface_color = ease_out_rgb(
            ground_color,
            climate_zone_color,
            climate_zone_density[0],
            2
        );
    }

    image_bgr(i, j, 0) = surface_color.b;
    image_bgr(i, j, 1) = surface_color.g;
    image_bgr(i, j, 2) = surface_color.r;

    if (ice_mask(i, j))
    {
        const float temperature_fraction = fminf(
            fabsf(temperature(i, j) / rendering_options.ice_temperature_limit),
            1
        );
        const worldgen::Rgb ice_color = lerp_rgb(
            rendering_options.ice_thin,
            rendering_options.ice_thick,
            temperature_fraction
        );
        const worldgen::Rgb pixel_color = ease_out_rgb(
            ground_color,
            ice_color,
            temperature_fraction,
            4
        );

        image_bgr(i, j, 0) = pixel_color.b;
        image_bgr(i, j, 1) = pixel_color.g;
        image_bgr(i, j, 2) = pixel_color.r;

        return;
    }

    if (ocean_mask(i, j))
    {
        const float depth_fraction = fminf(
            fabsf(altitude(i, j) / rendering_options.water_depth_limit), 
            1
        );
        const worldgen::Rgb ocean_color = lerp_rgb(
            rendering_options.water_shallow,
            rendering_options.water_deep,
            depth_fraction
        );

        image_bgr(i, j, 0) = ocean_color.b;
        image_bgr(i, j, 1) = ocean_color.g;
        image_bgr(i, j, 2) = ocean_color.r;
        
        return;
    }

    if (river_depth(i, j) > 0)
    {
        const float depth_fraction = fminf(
            fabsf(river_depth(i, j) / rendering_options.river_depth_limit),
            1
        );
        const worldgen::Rgb river_color = lerp_rgb(
            rendering_options.river_shallow,
            rendering_options.river_deep,
            depth_fraction
        );

        image_bgr(i, j, 0) = river_color.b;
        image_bgr(i, j, 1) = river_color.g;
        image_bgr(i, j, 2) = river_color.r;

        return;
    }
}

cudu::device::Array3D<unsigned char> worldgen::image(
    const cudu::device::Array2D<float>& altitude,
    const cudu::device::Array3D<float>& material,
    const cudu::device::Array2D<float>& temperature,
    const cudu::device::Array2D<bool>& ocean_mask,
    const cudu::device::Array2D<bool>& ice_mask,
    const cudu::device::Array2D<float>& precipitation,
    const cudu::device::Array2D<float>& river_depth,
    const std::vector<worldgen::ClimateZone>& climate_zones,
    const RenderingOptions& rendering_options)
{
    const float altitude_max = max(altitude.block());
    const float altitude_min = min(altitude.block());

    static worldgen::ClimateZone EMPTY_ZONE = worldgen::ClimateZone{
        Range{9999.f, 9999.f},
        Range{9999.f, 9999.f},
        Range{0.f, 0.f},
        Rgb{0, 0, 0},
        Rgb{0, 0, 0}
    };
    const auto zones = (
        climate_zones.empty() ?
        cudu::device::Array1D<worldgen::ClimateZone>::from_ptr(&EMPTY_ZONE, 1) :
        cudu::device::Array1D<worldgen::ClimateZone>::from_ptr(climate_zones.data(), climate_zones.size())
    );

    cudu::device::Array3D<unsigned char> image_bgr({ altitude.shape()[0], altitude.shape()[1], 3});

    CUDU_LAUNCH_BATCHES(
        image_kernel,
        altitude.size(),
        BATCH_SIZE,
        altitude,
        altitude_min,
        altitude_max,
        material,
        temperature,
        ocean_mask,
        ice_mask,
        precipitation,
        river_depth,
        zones,
        rendering_options,
        image_bgr
    );

    return image_bgr;
}
