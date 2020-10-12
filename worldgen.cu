#define _USE_MATH_DEFINES

#include <algorithm>
#include <math.h>
#include <random>

#include <curand_kernel.h>

#include "worldgen.h"

template <typename T>
__device__ T lerp(T a, T b, T t)
{
    static_assert(
        std::is_floating_point_v<T>,
        "interpolation requires floating point type"
    );
    return (1 - t) * a + t * b;
}

template <typename T>
__device__ T ease_in(T a, T b, T t, T power)
{
    static_assert(
        std::is_floating_point_v<T>,
        "interpolation requires floating point type"
    );
    return lerp(a, b, pow(t, power));
}

template <typename T>
__device__ T ease_out(T a, T b, T t, T power)
{
    static_assert(
        std::is_floating_point_v<T>,
        "interpolation requires floating point type"
    );
    return lerp(a, b, 1 - pow(1 - t, power));
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

__global__ void rng_setup_kernel(
    const unsigned long seed,
    cudu::ArrayRef2D<curandStatePhilox4_32_10> state)
{
    curand_init(seed, CUDU_THREAD_ID(), 0, &state[CUDU_THREAD_ID()]);
}

template <typename State_t>
__device__ float random_uniform_in_range(
    const float min,
    const float max,
    State_t& state)
{
    return min + (max - min) * curand_uniform(&state);
}

cudu::Array2D<curandStatePhilox4_32_10> new_rng_array(
    const std::array<size_t, 2>& size,
    const unsigned long seed,
    const unsigned max_nr_threads)
{
    cudu::Array2D<curandStatePhilox4_32_10> state(size);

    const auto workload = cudu::Workload::for_jobs(size[0] * size[1], max_nr_threads);
    rng_setup_kernel<<<workload.nr_blocks, workload.nr_threads>>>(seed, state);
    CUDU_ASSERT(cudaDeviceSynchronize());

    return state;
}

__host__ __device__ bool within(
    const float pixel[2],
    const float extents[2])
{
    return (
        0 <= pixel[0] && pixel[0] < extents[0] &&
        0 <= pixel[1] && pixel[1] < extents[1]
    );
}

__global__ void vicinity_lookup_kernel(
    const cudu::ConstArrayRef2D<bool> mask,
    const float radius_max,
    const float radius_step,
    const size_t nr_samples,
    cudu::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::ArrayRef3D<size_t> result)
{
    if (CUDU_THREAD_ID() >= mask.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(CUDU_THREAD_ID(), mask.shape(1), i, j);
    
    result(i, j, 2) = false;

    if (mask(i, j)) 
    {
        result(i, j, 0) = i;
        result(i, j, 1) = j;
        result(i, j, 2) = true;
        return;
    }

    for (float radius = 0; radius < radius_max; radius += radius_step)
    {
        for (size_t sample = 0; sample < nr_samples; ++sample)
        {
            const float arc_size = 2 * M_PI / nr_samples;
            const float arc_begin = sample * arc_size;
            const float arc_end = (i + 1) * arc_size;
            const float angle = random_uniform_in_range(arc_begin, arc_end, rng_state(i, j));
            const float magnitude = random_uniform_in_range(radius, radius + radius_step, rng_state(i, j));
            const float coords[2] = {
                i + magnitude * cosf(angle),
                j + magnitude * sinf(angle)
            };
            const float extents[2] = {
                mask.shape(0),
                mask.shape(1)
            };
            if (within(coords, extents) && mask(coords[0], coords[1])) {
                result(i, j, 0) = coords[0];
                result(i, j, 1) = coords[1];
                result(i, j, 2) = true;
                return;
            }
        }
    }
}

cudu::Array3D<size_t> vicinity_lookup(
    const cudu::Array2D<bool>& mask,
    const worldgen::VicinityLookupParams& lookup_params,
    const unsigned max_nr_threads)
{
    cudu::Array2D<curandStatePhilox4_32_10> rng_state = new_rng_array(
        {mask.shape(0), mask.shape(1)},
        lookup_params.rng_seed,
        max_nr_threads
    );
    cudu::Array3D<size_t> result({
        mask.shape(0),
        mask.shape(1),
        3
    });
    const auto workload = cudu::Workload::for_jobs(mask.size(), max_nr_threads);

    vicinity_lookup_kernel<<<workload.nr_blocks, workload.nr_threads>>>(
        mask,
        lookup_params.radius,
        lookup_params.radius_step,
        lookup_params.nr_samples,
        rng_state,
        result
    );
    CUDU_ASSERT(cudaDeviceSynchronize());

    return result;
}

__global__ void surface_altitude_diamond_kernel(
    const float noise,
    const size_t stride,
    cudu::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::ArrayRef2D<float> result)
{
    size_t i; 
    size_t j;
    unravel_index(CUDU_THREAD_ID(), result.shape(1) / stride, i, j);
    
    if (i >= result.shape(0) ||
        j >= result.shape(1)) {
        return;
    }

    const size_t offset = stride / 2;
    i = offset + i * stride;
    j = offset + j * stride;
    
    const float noise_factor = random_uniform_in_range(-noise, +noise, rng_state(i, j));

    result(i, j) = (
        result(i - offset, j - offset) +
        result(i - offset, j + offset) +
        result(i + offset, j - offset) +
        result(i + offset, j + offset)
    ) / 4 + noise_factor;
}

__global__ void surface_altitude_square_kernel(
    const float noise,
    const size_t stride,
    cudu::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::ArrayRef2D<float> result)
{
    size_t i;
    size_t j;
    unravel_index(
        CUDU_THREAD_ID() % (CUDU_THREAD_TOTAL() / 2), 
        result.shape(1) / stride, 
        i, j
    );

    if (i >= result.shape(0) ||
        j >= result.shape(1)) {
        return;
    }

    const size_t offset = stride / 2;
    i = i * stride;
    j = offset + j * stride;

    if (CUDU_THREAD_ID() >= CUDU_THREAD_TOTAL() / 2)
    {
        const size_t temp = i;
        i = j;
        j = temp;
    }
    
    const size_t r = result.shape(0);
    const size_t c = result.shape(1);

    const float noise_factor = random_uniform_in_range(-noise, +noise, rng_state(i, j));

    result(i, j) = (
        result(wrap(int(i) - int(stride) / 2, r), j) +
        result(wrap(int(i) + int(stride) / 2, r), j) +
        result(i, wrap(int(j) - int(stride) / 2, c)) +
        result(i, wrap(int(j) + int(stride) / 2, c))
    ) / 4 + noise_factor;
}

cudu::Array2D<float> worldgen::surface_altitude(
    const uint32_t max_level_of_detail,
    const float noise_initial,
    const float noise_scale_factor,
    const float bias,
    const uint64_t rng_seed,
    const unsigned max_nr_threads)
{
    const size_t edge_size = (size_t(1) << max_level_of_detail) + 1;

    cudu::Array2D<curandStatePhilox4_32_10> rng_state(
        new_rng_array({edge_size, edge_size}, rng_seed, max_nr_threads)
    );

    cudu::Array2D<float> altitude({edge_size, edge_size});
    
    // Initialize corners.
    std::default_random_engine rng(rng_seed);
    std::uniform_real_distribution<float> distrib(
        -noise_initial / 2, 
        +noise_initial / 2
    );
    altitude.upload_single(0, 0, distrib(rng) + bias);
    altitude.upload_single(0, edge_size - 1, distrib(rng) + bias);
    altitude.upload_single(edge_size - 1, 0, distrib(rng) + bias);
    altitude.upload_single(edge_size - 1, edge_size - 1, distrib(rng) + bias);
    
    float noise = noise_initial;
    
    for (size_t lod = 0; lod < max_level_of_detail; ++lod) {
        
        noise *= noise_scale_factor;

        const size_t stride = (edge_size - 1) >> lod;

        // Diamond step.
        const unsigned nr_diamonds = std::pow(4, lod);
        const auto workload_diamonds = cudu::Workload::for_jobs(nr_diamonds, max_nr_threads);
        surface_altitude_diamond_kernel<<<
            workload_diamonds.nr_blocks, 
            workload_diamonds.nr_threads>>>(noise, stride, rng_state, altitude);
        CUDU_ASSERT(cudaDeviceSynchronize());
        
        // Square step.
        const unsigned nr_squares = 2 * std::sqrt(nr_diamonds) * (std::sqrt(nr_diamonds) + 1);
        const auto workload_squares = cudu::Workload::for_jobs(nr_squares, max_nr_threads);
        surface_altitude_square_kernel<<<
            workload_squares.nr_blocks, 
            workload_squares.nr_threads>>>(noise, stride, rng_state, altitude);
        CUDU_ASSERT(cudaDeviceSynchronize());
    }

    return altitude;
}

__host__ __device__ float distance_squared(
    float x0, float y0,
    float x1, float y1)
{
    return (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
}

__global__ void cell_distance_kernel(
    const cudu::ConstArrayRef1D<worldgen::MaterialCell> cells,
    cudu::ArrayRef3D<size_t> cell_distance)
{
    if (CUDU_THREAD_ID() >= cell_distance.shape(0) * cell_distance.shape(1))
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(CUDU_THREAD_ID(), cell_distance.shape(1), i, j);

    extern __shared__ worldgen::MaterialCell cells_local[];

    const size_t iThread = threadIdx.x;

    if (iThread < cells.size())
    {
        cells_local[iThread] = cells[iThread];
    }
    __syncthreads();

    for (size_t k = 1; k < cells.size(); ++k)
    {
        cell_distance(i, j, k) = distance_squared(i, j, cells_local[k].row, cells_local[k].col);
    }
}

__global__ void material_mixture_kernal(
    const cudu::ConstArrayRef1D<worldgen::MaterialCell> cells,
    const cudu::ConstArrayRef3D<size_t> cell_distance,
    const float blur,
    cudu::ArrayRef3D<float> mixtures)
{
    if (CUDU_THREAD_ID() >= mixtures.shape(0) * mixtures.shape(1))
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(CUDU_THREAD_ID(), mixtures.shape(1), i, j);

    float distance_sum = 0;

    for (size_t m = 0; m < mixtures.shape(2); ++m)
    {
        mixtures(i, j, m) = 0;
    }
    for (size_t c = 0; c < cell_distance.shape(2); ++c)
    {
        const size_t m = cells[c].material_index;
        mixtures(i, j, m) += powf(cell_distance(i, j, c), blur);
        distance_sum += powf(cell_distance(i, j, c), blur);
    }
    for (size_t m = 0; m < mixtures.shape(2); ++m)
    {
        mixtures(i, j, m) = 1 - mixtures(i, j, m) / distance_sum;
    }
}

cudu::Array3D<float> worldgen::surface_material(
    const std::array<size_t, 2>& shape,
    const std::vector<SurfaceMaterial>& materials,
    const unsigned cell_count,
    const float cell_blur,
    const uint64_t rng_seed,
    const unsigned max_nr_threads)
{
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

    auto cells = cudu::Array1D<worldgen::MaterialCell>::from_container(cells_vec);

    // Compute nearest cells to each pixel.

    cudu::Array3D<size_t> cell_distance({shape[0], shape[1], cells.size()});

    const auto workload = cudu::Workload::for_jobs(
        cell_distance.size(),
        max_nr_threads
    );
    cell_distance_kernel<<<
        workload.nr_blocks,
        workload.nr_threads,
        cells.size_bytes()>>>(cells, cell_distance);
    CUDU_ASSERT(cudaDeviceSynchronize());

    // Compute material mixtures based on cell distance.

    cudu::Array3D<float> mixtures({shape[0], shape[1], materials.size()});

    material_mixture_kernal<<<
        workload.nr_blocks,
        workload.nr_threads>>>(cells, cell_distance, cell_blur, mixtures);
    CUDU_ASSERT(cudaDeviceSynchronize());

    return mixtures;
}

__device__ float latitude(
    const float i,
    const float extent)
{
    return (M_PI / 2) * fabsf(i - 0.5 * extent) / (0.5 * extent);
}

__global__ void temperature_kernel(
    const cudu::ConstArrayRef2D<float> surface_altitude,
    const float latitude_factor,
    const float altitude_factor,
    const float noise,
    const float bias,
    cudu::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::ArrayRef2D<float> temperature)
{
    if (CUDU_THREAD_ID() >= temperature.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(CUDU_THREAD_ID(), temperature.shape(1), i, j);

    const float pixel_latitude = latitude(i, temperature.shape(0));
    const float relative_equator_offset = 1 - fabsf(cosf(pixel_latitude));
    const float latitude_effect = -latitude_factor * relative_equator_offset;
    const float altitude_effect = -fmaxf(surface_altitude(i, j), 0) * altitude_factor;
    const float noise_factor = random_uniform_in_range(-noise, +noise, rng_state(i, j));
    
    temperature(i, j) = bias + latitude_effect + altitude_effect + noise_factor;
}

cudu::Array2D<float> worldgen::temperature(
    const cudu::Array2D<float>& surface_altitude,
    const float latitude_factor,
    const float altitude_factor,
    const float noise,
    const float bias,
    const uint64_t rng_seed,
    const unsigned max_nr_threads)
{
    cudu::Array2D<curandStatePhilox4_32_10> rng_state = new_rng_array(
        { surface_altitude.shape(0), surface_altitude.shape(1) },
        rng_seed,
        max_nr_threads
    );
    cudu::Array2D<float> surface_temperature({ 
        surface_altitude.shape(0), 
        surface_altitude.shape(1) 
    });
    
    const auto workload = cudu::Workload::for_jobs(
        surface_temperature.size(),
        max_nr_threads
    );
    temperature_kernel<<<workload.nr_blocks, workload.nr_threads>>>(
        surface_altitude, 
        latitude_factor, 
        altitude_factor, 
        noise, 
        bias,
        rng_state,
        surface_temperature
    );
    CUDU_ASSERT(cudaDeviceSynchronize());

    return surface_temperature;
}

__global__ void ocean_kernel(
    const cudu::ConstArrayRef2D<float> surface_altitude,
    const cudu::ConstArrayRef2D<float> temperature,
    const float boil_temperature,
    cudu::ArrayRef2D<bool> ocean)
{
    if (CUDU_THREAD_ID() >= ocean.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(CUDU_THREAD_ID(), ocean.shape(1), i, j);

    ocean(i, j) = (
        surface_altitude(i, j) <= 0 && 
        temperature(i, j) < boil_temperature
    );
}

cudu::Array2D<bool> worldgen::ocean(
    const cudu::Array2D<float>& surface_altitude,
    const cudu::Array2D<float>& temperature,
    const float boil_temperature,
    const unsigned max_nr_threads)
{
    cudu::Array2D<bool> mask({
        surface_altitude.shape(0),
        surface_altitude.shape(1)
    });
    
    const auto workload = cudu::Workload::for_jobs(
        mask.size(),
        max_nr_threads
    );
    ocean_kernel<<<workload.nr_blocks, workload.nr_threads>>>(
        surface_altitude, 
        temperature,
        boil_temperature,
        mask
    );
    CUDU_ASSERT(cudaDeviceSynchronize());

    // TODO: minimum size

    return mask;
}

__global__ void ice_kernel(
    const cudu::ConstArrayRef2D<float> temperature,
    const cudu::ConstArrayRef2D<bool> ocean_mask,
    const float ocean_freeze_temperature,
    cudu::ArrayRef2D<bool> ice_mask)
{
    if (CUDU_THREAD_ID() >= ice_mask.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(CUDU_THREAD_ID(), ice_mask.shape(1), i, j);

    ice_mask(i, j) = (
        ocean_mask(i, j) ?
        temperature(i, j) <= ocean_freeze_temperature : 
        temperature(i, j) <= 0
    );
}

cudu::Array2D<bool> worldgen::ice(
    const cudu::Array2D<float>& temperature,
    const cudu::Array2D<bool>& ocean_mask,
    const float ocean_freeze_temperature,
    const unsigned max_nr_threads)
{
    cudu::Array2D<bool> mask({
        temperature.shape(0),
        temperature.shape(1)
    });

    const auto workload = cudu::Workload::for_jobs(
        mask.size(),
        max_nr_threads
    );
    ice_kernel<<<workload.nr_blocks, workload.nr_threads>>>(
        temperature,
        ocean_mask,
        ocean_freeze_temperature,
        mask
    );
    CUDU_ASSERT(cudaDeviceSynchronize());

    return mask;
}

__device__ float range_based_weight(
    const float value,
    const worldgen::Range& range)
{
    const float distance = (
        value < range.min ? range.min - value :
        value > range.max ? value - range.max :
        0
    );
    
    const float relative_distance = distance / range.size();

    return 1 - relative_distance;
}

__global__ void precipitation_kernel(
    const cudu::ConstArrayRef2D<float> temperature,
    const cudu::ConstArrayRef3D<size_t> ocean_vicinity,
    const float ocean_distance_factor,
    const cudu::ConstArrayRef1D<worldgen::ClimateZone> climate_zones,
    const float climate_zone_weight_min,
    cudu::ArrayRef2D<float> precipitation)
{
    if (CUDU_THREAD_ID() >= precipitation.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(CUDU_THREAD_ID(), precipitation.shape(1), i, j);

    const float pixel_latitude = latitude(i, precipitation.shape(0));
    
    float weight_sum = 0;

    for (const worldgen::ClimateZone& zone: climate_zones)
    {
        const float weight = (
            fmaxf(range_based_weight(pixel_latitude, zone.latitude_range), climate_zone_weight_min) *
            fmaxf(range_based_weight(temperature(i, j), zone.temperature_range), climate_zone_weight_min)
        );
        weight_sum += weight;
        precipitation(i, j) += weight * zone.precipitation_range.center();
    }

    precipitation(i, j) /= weight_sum;

    const float ocean_distance = (
        ocean_vicinity(i, j, 2) ?
        sqrtf(distance_squared(i, j, ocean_vicinity(i, j, 0), ocean_vicinity(i, j, 1))) :
        100 * fmaxf(precipitation.shape(0), precipitation.shape(1))
    );
    const float ocean_factor = fminf(ocean_distance * ocean_distance_factor, 1);

    precipitation(i, j) *= (1 - ocean_factor);
}

cudu::Array2D<float> worldgen::precipitation(
    const cudu::Array2D<float>& temperature,
    const cudu::Array2D<bool>& ocean_mask,
    const std::vector<ClimateZone>& climate_zones,
    const float climate_zone_weight_min,
    const VicinityLookupParams& ocean_lookup_params,
    const float ocean_distance_factor,
    const unsigned max_nr_threads)
{
    const auto zones = cudu::Array1D<ClimateZone>::from_container(climate_zones);

    cudu::Array3D<size_t> ocean_vicinity = vicinity_lookup(
        ocean_mask,
        ocean_lookup_params,
        max_nr_threads
    );
    cudu::Array2D<float> result({temperature.shape(0), temperature.shape(1)});
    
    const auto workload = cudu::Workload::for_jobs(result.size(), max_nr_threads);

    precipitation_kernel<<<workload.nr_blocks, workload.nr_threads>>>(
        temperature,
        ocean_vicinity,
        ocean_distance_factor,
        zones,
        climate_zone_weight_min,
        result
    );
    CUDU_ASSERT(cudaDeviceSynchronize());

    return result;
}

__global__ void max_in_block_kernel(
    const cudu::ConstArrayRef2D<float> data,
    const size_t block_size,
    cudu::ArrayRef3D<size_t> result)
{
    if (CUDU_THREAD_ID() >= result.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(CUDU_THREAD_ID(), result.shape(1), i, j);

    const size_t r_min = i * block_size;
    const size_t r_max = fminf((i + 1) * block_size, data.shape(1));

    const size_t c_min = j * block_size;
    const size_t c_max = fminf((j + 1) * block_size, data.shape(0));

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

cudu::Array3D<size_t> max_in_block(
    const cudu::Array2D<float>& data,
    const size_t block_size,
    const unsigned max_nr_threads)
{
    cudu::Array3D<size_t> result({
        size_t(std::ceil(data.shape(0) / float(block_size))), 
        size_t(std::ceil(data.shape(1) / float(block_size))), 
        2
    });

    const auto workload = cudu::Workload::for_jobs(result.shape(0) * result.shape(1), max_nr_threads);
    max_in_block_kernel<<<workload.nr_blocks, workload.nr_threads>>>(data, block_size, result);
    CUDU_ASSERT(cudaDeviceSynchronize());

    return result;
}

void run_river(
    const std::array<size_t, 2>& source,
    const std::array<size_t, 2>& sink,
    const cudu::host::Array2D<float>& altitude,
    const cudu::host::Array2D<bool>& ocean_mask,
    const cudu::host::Array2D<bool>& ice_mask,
    cudu::host::Array2D<float>& river_depth,
    cudu::host::Array2D<bool>& visited)
{
    std::vector<std::array<size_t, 2>> trail;
    std::array<size_t, 2> head = source;

    while (trail.size() < 10000)
    {
        if (ice_mask(head[0], head[1]) || visited(head[0], head[1]))
        {
            break;
        }
        if (ocean_mask(head[0], head[1]) || river_depth(head[0], head[1]) != 0)
        {
            trail.push_back(head);
            break;
        }


        std::array<size_t, 2> best_candidate;
        float best_altitude = std::numeric_limits<float>::max();

        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                const float extent[2] = { river_depth.shape(0), river_depth.shape(1) };
                std::array<float, 2> candidate = {head[0] + dx, head[1] + dy};
                if (abs(dx) + abs(dy) != 0 && within(candidate.data(), extent) &&
                    distance_squared(candidate[0], candidate[1], sink[0], sink[1]) <= 
                    distance_squared(head[0], head[1], sink[0], sink[1]) &&
                    altitude(candidate[0], candidate[1]) < best_altitude)
                {
                    best_candidate[0] = candidate[0];
                    best_candidate[1] = candidate[1];
                    best_altitude = altitude(candidate[0], candidate[1]);
                }
            }
        }
        if (best_altitude = std::numeric_limits<float>::max())
        {
            break;
        }
        river_depth(head[0], head[1]) = 1;
        trail.push_back(head);
        visited(head[0], head[1]) = true;
        head = best_candidate;
    }
    if (trail.size() < 2)
    {
        for (auto pixel : trail)
        {
            river_depth(pixel[0], pixel[1]) = 0;
        }
    }
    for (auto pixel : trail)
    {
        visited(pixel[0], pixel[1]) = false;
    }
    if (trail.size() < 2)
    {
        return;
    }
    bool done = false;
    size_t nr_steps = 0;
    while (!done && nr_steps < 1000) 
    {
        done = true;
        for (size_t i = 0; i < trail.size() - 2; ++i) 
        {
            const auto curr = trail[i];
            const auto next = trail[i + 1];
            static const auto potential = [&altitude, &river_depth](std::array<size_t, 2> pixel) {
                return altitude(pixel[0], pixel[1]) + river_depth(pixel[0], pixel[1]);
            };
            if (potential(curr) < potential(next)) 
            {
                river_depth(curr[0], curr[1])= potential(next) - altitude(curr[0], curr[1]);
                done = false;
            }
        }
        nr_steps += 1;
    }
}

cudu::Array2D<float> worldgen::rivers(
    const cudu::Array2D<float>& surface_altitude,
    const cudu::Array2D<bool>& ocean_mask,
    const cudu::Array2D<bool>& ice_mask,
    const size_t block_size,
    const VicinityLookupParams& ocean_lookup_params,
    const unsigned max_nr_threads)
{
    const cudu::Array3D<size_t> sources = max_in_block(surface_altitude, block_size, max_nr_threads);
    const cudu::Array3D<size_t> sinks = vicinity_lookup(ocean_mask, ocean_lookup_params, max_nr_threads);

    std::vector<size_t> sources_vec(sources.size());
    sources.block().download_all(sources_vec.data());

    const cudu::host::Array2D<float> h_altitude = surface_altitude.download();
    const cudu::host::Array2D<bool> h_ocean_mask = ocean_mask.download();
    const cudu::host::Array2D<bool> h_ice_mask = ice_mask.download();
    
    cudu::host::Array2D<float> h_river_depth({h_altitude.shape(0), h_altitude.shape(1)});
    cudu::host::Array2D<bool> h_visited({h_altitude.shape(0), h_altitude.shape(1)});

    for (size_t iSource = 0; iSource < sources.size() / 2; ++iSource)
    {
        const size_t source[2] = {
            sources_vec[2 * iSource + 0],
            sources_vec[2 * iSource + 1]
        };
        
        const bool sink_exists = sinks.download_single(source[0], source[1], 2);

        if (sink_exists)
        {
            const size_t sink[2] = {
                sinks.download_single(source[0], source[1], 0),
                sinks.download_single(source[0], source[1], 1)
            };
            run_river({ source[0], source[1] }, { sink[0], sink[1] }, h_altitude, h_ocean_mask, h_ice_mask, h_river_depth, h_visited);
        }
    }
    cudu::Array2D<float> result({ h_river_depth.shape(0), h_river_depth.shape(1) });
    result.block().upload_all(h_river_depth.ptr());
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
T max(
    const cudu::DataBlock<T>& data, 
    const unsigned max_nr_threads)
{
    if (data.size() == 1)
    {
        return data.download_single(0);
    }

    size_t size = data.size();
    unsigned nr_blocks = std::max<unsigned>(std::ceil(float(size) / max_nr_threads), 1);
    const unsigned shared_memory_bytes = max_nr_threads * sizeof(T);
    
    cudu::DataBlock<T> buffer_1(nr_blocks);
    cudu::DataBlock<T> buffer_2(nr_blocks);
    
    const cudu::DataBlock<T>* input = &data;
    cudu::DataBlock<T>* output = &buffer_1;

    while (size > 1)
    {
        max_kernel<<<nr_blocks, max_nr_threads, shared_memory_bytes>>>(
            input->ptr(), 
            size, 
            output->ptr()
        );
        CUDU_ASSERT(cudaDeviceSynchronize());

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
T min(
    const cudu::DataBlock<T>& data,
    const unsigned max_nr_threads)
{
    if (data.size() == 1)
    {
        return data.download_single(0);
    }

    size_t size = data.size();
    unsigned nr_blocks = std::max<unsigned>(std::ceil(float(size) / max_nr_threads), 1);
    const unsigned shared_memory_bytes = max_nr_threads * sizeof(T);

    cudu::DataBlock<T> buffer_1(nr_blocks);
    cudu::DataBlock<T> buffer_2(nr_blocks);

    const cudu::DataBlock<T>* input = &data;
    cudu::DataBlock<T>* output = &buffer_1;

    while (size > 1)
    {
        min_kernel<<<nr_blocks, max_nr_threads, shared_memory_bytes>>>(
            input->ptr(),
            size,
            output->ptr()
        );
        CUDU_ASSERT(cudaDeviceSynchronize());

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
    const cudu::ConstArrayRef2D<float> surface_altitude,
    const float surface_altitude_min,
    const float surface_altitude_max,
    const cudu::ConstArrayRef3D<float> surface_material,
    const cudu::ConstArrayRef2D<float> temperature,
    const cudu::ConstArrayRef2D<bool> ocean_mask,
    const cudu::ConstArrayRef2D<bool> ice_mask,
    const cudu::ConstArrayRef2D<float> precipitation,
    const cudu::ConstArrayRef2D<float> river_depth,
    const cudu::ConstArrayRef1D<worldgen::ClimateZone> climate_zones,
    const worldgen::RenderingOptions rendering_options,
    cudu::ArrayRef3D<unsigned char> image_bgr)
{
    if (CUDU_THREAD_ID() >= surface_altitude.size())
    {
        return;
    }

    size_t i;
    size_t j;
    unravel_index(CUDU_THREAD_ID(), image_bgr.shape(1), i, j);

    worldgen::Rgb surface_color{0, 0, 0};

    for (size_t k = 0; k < surface_material.shape(2); ++k)
    {
        const float weight = surface_material(i, j, k);
        const worldgen::MaterialVisual material_visual = rendering_options.materials[k];
        const worldgen::Rgb material_color = lerp_rgb(
            material_visual.low_altitude,
            material_visual.high_altitude,
            surface_altitude(i, j) / surface_altitude_max
        );
        surface_color.r += weight * material_color.r;
        surface_color.g += weight * material_color.g;
        surface_color.b += weight * material_color.b;
    }

    worldgen::ClimateZone climate_zone = climate_zones[0];
    float climate_zone_score = fabsf(
        precipitation(i, j) - climate_zone.precipitation_range.center()
    );
    for (const worldgen::ClimateZone& zone : climate_zones)
    {
        const float score = fabsf(
            precipitation(i, j) - zone.precipitation_range.center()
        );
        if (score < climate_zone_score)
        {
            climate_zone_score = score;
            climate_zone = zone;
        }
    }

    const float temperature_relative_distance =
        fabsf(temperature(i, j) - climate_zone.temperature_range.center()) / 
        climate_zone.temperature_range.size();
    const float temperature_match = 1 - fminf(temperature_relative_distance, 1);

    const float precipitation_range = climate_zone.precipitation_range.size();
    const float precipitation_relative_distance =
        fabsf(precipitation(i, j) - climate_zone.precipitation_range.center()) / 
        climate_zone.precipitation_range.size();
    const float precipitation_match = 1 - fminf(precipitation_relative_distance, 1);

    const float density = temperature_match * precipitation_match;
    const worldgen::Rgb zone_color = lerp_rgb(climate_zone.visual_sparse, climate_zone.visual_dense, density);
    const worldgen::Rgb base_color = ease_out_rgb(surface_color, zone_color, density, 2);

    image_bgr(i, j, 0) = base_color.b;
    image_bgr(i, j, 1) = base_color.g;
    image_bgr(i, j, 2) = base_color.r;

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
            surface_color,
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
            fabsf(surface_altitude(i, j) / rendering_options.water_depth_limit), 
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
            fabsf(river_depth(i, j) / rendering_options.water_depth_limit),
            1
        );
        const worldgen::Rgb river_color = lerp_rgb(
            rendering_options.water_shallow,
            rendering_options.water_deep,
            depth_fraction
        );

        image_bgr(i, j, 0) = river_color.b;
        image_bgr(i, j, 1) = river_color.g;
        image_bgr(i, j, 2) = river_color.r;

        return;
    }
}

cudu::Array3D<unsigned char> worldgen::image(
    const cudu::Array2D<float>& surface_altitude,
    const cudu::Array3D<float>& surface_material,
    const cudu::Array2D<float>& temperature,
    const cudu::Array2D<bool>& ocean_mask,
    const cudu::Array2D<bool>& ice_mask,
    const cudu::Array2D<float>& precipitation,
    const cudu::Array2D<float>& river_depth,
    const std::vector<worldgen::ClimateZone>& climate_zones,
    const RenderingOptions& rendering_options,
    const unsigned max_nr_threads)
{
    const float surface_altitude_max = max(surface_altitude.block(), max_nr_threads);
    const float surface_altitude_min = min(surface_altitude.block(), max_nr_threads);

    const auto zones = cudu::Array1D<worldgen::ClimateZone>::from_container(climate_zones);

    const size_t n_x = surface_altitude.shape(0);
    const size_t n_y = surface_altitude.shape(1);
    cudu::Array3D<unsigned char> image_bgr({n_x, n_y, 3});

    const cudu::Workload workload = cudu::Workload::for_jobs(n_x * n_y, max_nr_threads);
    image_kernel<<<workload.nr_blocks, workload.nr_threads>>>(
        surface_altitude,
        surface_altitude_min,
        surface_altitude_max,
        surface_material,
        temperature,
        ocean_mask,
        ice_mask,
        precipitation,
        river_depth,
        zones,
        rendering_options,
        image_bgr
    );
    CUDU_ASSERT(cudaDeviceSynchronize());

    return image_bgr;
}
