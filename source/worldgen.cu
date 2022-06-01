/**
 * Copyright (c) 2022 Raoul Harel
 * All rights reserved
 */

#define _USE_MATH_DEFINES
#include <math.h>  // M_PI

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "../external/cuda_noise.h"

#include "worldgen.cuh"

constexpr size_t BATCH_SIZE = 512 * 512;

__device__ float clamp(float value, float min, float max) {
  return fminf(max, fmaxf(min, value));
}

__host__ __device__ float
distance_squared_2d(float x0, float y0, float x1, float y1) {
  return (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
}

__host__ __device__ float distance_2d(float x0, float y0, float x1, float y1) {
  return sqrtf(distance_squared_2d(x0, y0, x1, y1));
}

__device__ void
unravel_index(const size_t index, const size_t extent, size_t& i, size_t& j) {
  i = index / extent;
  j = index % extent;
}

__global__ void binary_image_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<bool> data,
    const worldgen::Rgb truth_color,
    cudu::device::ArrayRef3D<unsigned char> image) {

  if (job_offset + CUDU_THREAD_ID() >= data.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), image.shape()[1], i, j);

  const auto color = data(i, j) ? truth_color : worldgen::Rgb{0, 0, 0};

  image(i, j, 0) = color.b;
  image(i, j, 1) = color.g;
  image(i, j, 2) = color.r;
}

cudu::device::Array3D<unsigned char> worldgen::binary_image(
    const cudu::device::Array2D<bool>& data,
    const Rgb& truth_color) {

  cudu::device::Array3D<unsigned char> image(
      {data.shape()[0], data.shape()[1], 3});
  CUDU_LAUNCH_BATCHES(
      binary_image_kernel,
      data.size(),
      BATCH_SIZE,
      cudu::device::ref_const(data),
      truth_color,
      cudu::device::ref(image));
  return image;
}

__global__ void colormap_image_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> data,
    const float value_min,
    const float value_max,
    const float value_mid,
    const cudu::device::ConstArrayRef2D<unsigned char> colormap,
    cudu::device::ArrayRef3D<unsigned char> image) {

  if (job_offset + CUDU_THREAD_ID() >= data.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), image.shape()[1], i, j);

  const auto fraction =
      data(i, j) < value_mid
          ? 0.0f + 0.5f * (data(i, j) - value_min) / (value_mid - value_min)
          : 0.5f + 0.5f * (data(i, j) - value_mid) / (value_max - value_mid);
  const auto color_index =
      size_t(clamp(fraction, 0, 1) * (colormap.shape()[0] - 1));

  image(i, j, 0) = colormap(color_index, 0);
  image(i, j, 1) = colormap(color_index, 1);
  image(i, j, 2) = colormap(color_index, 2);
}

cudu::device::Array3D<unsigned char> worldgen::colormap_image(
    const cudu::device::Array2D<float>& data,
    const cudu::device::Array2D<unsigned char>& colormap,
    const float min,
    const float max,
    const float mid) {

  cudu::device::Array3D<unsigned char> image(
      {data.shape()[0], data.shape()[1], 3});
  CUDU_LAUNCH_BATCHES(
      colormap_image_kernel,
      data.size(),
      BATCH_SIZE,
      cudu::device::ref_const(data),
      min,
      max,
      mid,
      cudu::device::ref_const(colormap),
      cudu::device::ref(image));
  return image;
}

cudu::device::Array3D<unsigned char> worldgen::colormap_image(
    const cudu::device::Array2D<float>& data,
    const cudu::device::Array2D<unsigned char>& colormap,
    const float min,
    const float max) {

  return colormap_image(data, colormap, min, max, (min + max) / 2);
}

__global__ void distance_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<bool> mask,
    const size_t iteration_index,
    cudu::device::ArrayRef2D<float> distance) {

  if (job_offset + CUDU_THREAD_ID() >= distance.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), distance.shape()[1], i, j);

#define DISTANCE_UNASSIGNED(VALUE) (VALUE > iteration_index)

  if (!DISTANCE_UNASSIGNED(distance(i, j))) {
    return;
  } else if (mask(i, j)) {
    distance(i, j) = 0;
    return;
  }

  const float PLACEHOLDER_DISTANCE = mask.size();
  float distance_min = PLACEHOLDER_DISTANCE;
  for (int di = -1; di <= 1; ++di) {
    for (int dj = -1; dj <= 1; ++dj) {
      const int neighbor_i = int(i) + di;
      const int neighbor_j = int(j) + dj;
      if (0 <= neighbor_i && neighbor_i < distance.shape()[0] &&
          0 <= neighbor_j && neighbor_j < distance.shape()[1] &&
          distance(neighbor_i, neighbor_j) < distance_min) {
        distance_min =
            (distance(neighbor_i, neighbor_j) +
             distance_2d(i, j, neighbor_i, neighbor_j));
      }
    }
  }
  if (distance_min != PLACEHOLDER_DISTANCE) {
    distance(i, j) = distance_min;
  }

#undef DISTANCE_UNASSIGNED
}

cudu::device::Array2D<float> worldgen::distance(
    const cudu::device::Array2D<bool>& mask,
    const uint32_t limit) {

  cudu::device::Array2D<float> result(
      mask.shape(),
      distance_2d(0, 0, float(mask.shape()[0]), float(mask.shape()[1])));

  for (size_t i = 0; i < limit; ++i) {
    CUDU_LAUNCH_BATCHES(
        distance_kernel,
        result.size(),
        BATCH_SIZE,
        cudu::device::ref_const(mask),
        i,
        cudu::device::ref(result));
  }
  return result;
}

__device__ float random_uniform_in_range(
    const float min,
    const float max,
    curandStatePhilox4_32_10& state) {

  return min + (max - min) * curand_uniform(&state);
}

__global__ void new_rng_state_kernel(
    const size_t job_offset,
    const uint64_t seed,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> state) {

  const size_t i = job_offset + CUDU_THREAD_ID();

  if (i < state.size()) {
    curand_init(seed, i, 0, &state[i]);
  }
}

cudu::device::Array2D<curandStatePhilox4_32_10> new_rng_state(
    const cudu::Shape2D& shape,
    const uint64_t seed) {

  cudu::device::Array2D<curandStatePhilox4_32_10> result(shape);
  CUDU_LAUNCH_BATCHES(
      new_rng_state_kernel,
      result.size(),
      BATCH_SIZE,
      seed,
      cudu::device::ref(result));
  return result;
}

__global__ void elevation_kernel(
    const size_t job_offset,
    const size_t nr_layers,
    const float scale,
    const float lacunarity,
    const float decay,
    const float bias,
    const float stretch,
    const int seed,
    cudu::device::ArrayRef2D<float> result) {

  if (job_offset + CUDU_THREAD_ID() >= result.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), result.shape()[1], i, j);

  const float max_dimension = fmaxf(result.shape()[0], result.shape()[1]);

  result(i, j) =
      bias +
      stretch * cudaNoise::repeaterSimplex(
                    {float(i) / max_dimension, float(j) / max_dimension, 0},
                    scale,
                    seed,
                    nr_layers,
                    lacunarity,
                    decay);
}

cudu::device::Array2D<float> worldgen::elevation(
    const size_t width,
    const size_t height,
    const size_t nr_layers,
    const float scale,
    const float lacunarity,
    const float decay,
    const float bias,
    const float stretch,
    const int seed) {

  cudu::device::Array2D<float> result(cudu::Shape2D(height, width));
  CUDU_LAUNCH_BATCHES(
      elevation_kernel,
      result.size(),
      BATCH_SIZE,
      nr_layers,
      scale,
      lacunarity,
      decay,
      bias,
      stretch,
      seed,
      cudu::device::ref(result));
  return result;
}

__global__ void ocean_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> elevation,
    cudu::device::ArrayRef2D<bool> ocean) {

  const size_t i = job_offset + CUDU_THREAD_ID();
  if (i < ocean.size()) {
    ocean[i] = elevation[i] <= 0;
  }
}

cudu::device::Array2D<bool> worldgen::ocean(
    const cudu::device::Array2D<float>& elevation) {

  cudu::device::Array2D<bool> result(elevation.shape());
  CUDU_LAUNCH_BATCHES(
      ocean_kernel,
      result.size(),
      BATCH_SIZE,
      cudu::device::ref_const(elevation),
      cudu::device::ref(result));
  return result;
}

__device__ float latitude_from_index(const float i, const float extent) {
  return (M_PI / 2) * fabsf(i - 0.5f * extent) / (0.5f * extent);
}

__global__ void temperature_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> elevation,
    const cudu::device::ConstArrayRef2D<float> ocean_distance,
    const float latitude_difference,
    const float elevation_factor,
    const float ocean_difference,
    const float ocean_difference_falloff,
    const float noise,
    const float bias,
    cudu::device::ArrayRef2D<curandStatePhilox4_32_10> rng_state,
    cudu::device::ArrayRef2D<float> temperature) {

  if (job_offset + CUDU_THREAD_ID() >= temperature.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), temperature.shape()[1], i, j);

  const float latitude = latitude_from_index(i, temperature.shape()[0]);
  const float relative_equator_offset = 1 - fabsf(cosf(latitude));
  const float latitude_effect = latitude_difference * relative_equator_offset;
  const float elevation_effect = fmaxf(elevation(i, j), 0) * elevation_factor;
  const float ocean_distance_effect_falloff =
      ocean_difference_falloff * ocean_distance(i, j);
  const float ocean_distance_effect =
      ocean_difference < 0
          ? clamp(
                ocean_difference + ocean_distance_effect_falloff,
                ocean_difference,
                0)
          : clamp(
                ocean_difference + ocean_distance_effect_falloff,
                0,
                ocean_difference);
  const float noise_factor =
      random_uniform_in_range(-noise, +noise, rng_state(i, j));

  temperature(i, j) = bias + latitude_effect + elevation_effect +
                      ocean_distance_effect + noise_factor;
}

cudu::device::Array2D<float> worldgen::temperature(
    const cudu::device::Array2D<float>& elevation,
    const cudu::device::Array2D<float>& ocean_distance,
    const float latitude_difference,
    const float elevation_factor,
    const float ocean_difference,
    const float ocean_difference_falloff,
    const float noise,
    const float bias,
    const uint64_t rng_seed) {

  cudu::device::Array2D<curandStatePhilox4_32_10> device_rng =
      new_rng_state(elevation.shape(), rng_seed);
  cudu::device::Array2D<float> result(elevation.shape());
  CUDU_LAUNCH_BATCHES(
      temperature_kernel,
      result.size(),
      BATCH_SIZE,
      cudu::device::ref_const(elevation),
      cudu::device::ref_const(ocean_distance),
      latitude_difference,
      elevation_factor,
      ocean_difference,
      ocean_difference_falloff,
      noise,
      bias,
      cudu::device::ref(device_rng),
      cudu::device::ref(result));
  return result;
}

__global__ void ice_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<bool> ocean,
    const cudu::device::ConstArrayRef2D<float> temperature,
    const float temperature_threshold_land,
    const float temperature_threshold_ocean,
    cudu::device::ArrayRef2D<bool> ice) {

  const size_t i = job_offset + CUDU_THREAD_ID();
  if (i < ice.size()) {
    const auto temperature_threshold =
        ocean[i] ? temperature_threshold_ocean : temperature_threshold_land;
    ice[i] = temperature[i] <= temperature_threshold;
  }
}

cudu::device::Array2D<bool> worldgen::ice(
    const cudu::device::Array2D<bool>& ocean,
    const cudu::device::Array2D<float>& temperature,
    const float temperature_threshold_land,
    const float temperature_threshold_ocean) {

  cudu::device::Array2D<bool> result(ocean.shape());
  CUDU_LAUNCH_BATCHES(
      ice_kernel,
      result.size(),
      BATCH_SIZE,
      cudu::device::ref_const(ocean),
      cudu::device::ref_const(temperature),
      temperature_threshold_land,
      temperature_threshold_ocean,
      cudu::device::ref(result));
  return result;
}

__global__ void precipitation_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> ocean_distance,
    const cudu::device::ConstArrayRef2D<float> temperature,
    const float ocean_distance_factor,
    const float multiplier,
    const cudu::device::ConstArrayRef1D<float> temperature_effect_coefficients,
    cudu::device::ArrayRef2D<float> precipitation) {

  const size_t i = job_offset + CUDU_THREAD_ID();
  if (i >= precipitation.size()) {
    return;
  }

  const float t = temperature[i];
  const float precipitation_max =
      multiplier * (temperature_effect_coefficients[0] +
                    temperature_effect_coefficients[1] * t +
                    temperature_effect_coefficients[2] * t * t +
                    temperature_effect_coefficients[3] * t * t * t);

  const float ocean_distance_effect = ocean_distance_factor * ocean_distance[i];

  precipitation[i] =
      fmaxf(precipitation_max + precipitation_max * ocean_distance_effect, 0);
}

cudu::device::Array2D<float> worldgen::precipitation(
    const cudu::device::Array2D<float>& ocean_distance,
    const cudu::device::Array2D<float>& temperature,
    const float ocean_distance_factor,
    const float multiplier,
    const std::array<float, 4>& temperature_effect_coefficients) {

  const auto temperature_effect_coefficients_on_device =
      cudu::device::Array1D<float>::from_ptr(
          temperature_effect_coefficients.data(),
          temperature_effect_coefficients.size());

  cudu::device::Array2D<float> result(ocean_distance.shape());
  CUDU_LAUNCH_BATCHES(
      precipitation_kernel,
      result.size(),
      BATCH_SIZE,
      cudu::device::ref_const(ocean_distance),
      cudu::device::ref_const(temperature),
      ocean_distance_factor,
      multiplier,
      cudu::device::ref_const(temperature_effect_coefficients_on_device),
      cudu::device::ref(result));
  return result;
}

__global__ void lake_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> elevation,
    const cudu::device::ConstArrayRef2D<bool> ocean,
    const cudu::device::ConstArrayRef2D<float> depth_previously,
    const cudu::device::ConstArrayRef2D<float> drain_distance_previously,
    cudu::device::ArrayRef2D<float> depth_now,
    cudu::device::ArrayRef2D<float> drain_distance_now,
    cudu::device::ArrayRef2D<float> level) {

  if (job_offset + CUDU_THREAD_ID() >= depth_now.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), depth_now.shape()[1], i, j);

  if (ocean(i, j)) {
    return;
  }

  bool adjacent_to_water = false;

  // Downstream properties.
  float downstream_level = 0;
  size_t dsi = i;
  size_t dsj = j;

  for (int di = -1; di <= 1; ++di) {
    for (int dj = -1; dj <= 1; ++dj) {

      const int ni = i + di;
      const int nj = j + dj;

      if (ni < 0 || ni >= depth_now.shape()[0] || nj < 0 ||
          nj >= depth_now.shape()[1] || (ni == i && nj == j)) {
        continue;
      }

      const bool water_at_neighbor =
          ocean(ni, nj) || depth_previously(ni, nj) > 0;

      if (!water_at_neighbor) {
        continue;
      }

      adjacent_to_water = true;

      const float neighbor_level =
          ocean(ni, nj) ? 0 : (elevation(ni, nj) + depth_previously(ni, nj));

      if ((dsi == i && dsj == j) || neighbor_level < downstream_level) {
        downstream_level = neighbor_level;
        dsi = ni;
        dsj = nj;
      }
    }
  }

  if (!adjacent_to_water) {
    return;
  }

  const float local_level = elevation(i, j) + depth_previously(i, j);

  if (depth_previously(i, j) > 0) {
    if (local_level > downstream_level) {
      depth_now(i, j) = fmaxf(downstream_level - elevation(i, j), 1);
      drain_distance_now(i, j) = drain_distance_previously(dsi, dsj) + 1;
    } else {
      depth_now(i, j) = depth_previously(i, j);
      drain_distance_now(i, j) = drain_distance_previously(i, j);
    }
  } else {
    if (local_level > downstream_level) {
      depth_now(i, j) = 1;
    } else {
      depth_now(i, j) = downstream_level - elevation(i, j);
    }
    drain_distance_now(i, j) = drain_distance_previously(dsi, dsj) + 1;
  }

  level(i, j) = elevation(i, j) + depth_now(i, j);
}

__global__ void lake_threshold_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> elevation,
    const cudu::device::ConstArrayRef2D<bool> ice,
    const float threshold,
    cudu::device::ArrayRef2D<float> depth,
    cudu::device::ArrayRef2D<float> adjusted_elevation) {

  const size_t i = job_offset + CUDU_THREAD_ID();
  if (i < depth.size()) {
    if (depth[i] < threshold || ice[i]) {
      depth[i] = 0;
      adjusted_elevation[i] = elevation[i];
    } else {
      adjusted_elevation[i] = elevation[i] - threshold;
    }
  }
}

worldgen::LakeDatasets worldgen::lakes(
    const cudu::device::Array2D<float>& elevation,
    const cudu::device::Array2D<bool>& ocean,
    const cudu::device::Array2D<bool>& ice,
    const size_t iteration_limit,
    const float depth_threshold) {

  cudu::device::Array2D<float> depth_previously(elevation.shape(), 0);
  cudu::device::Array2D<float> depth_now(elevation.shape(), 0);
  cudu::device::Array2D<float> drain_distance_previously(elevation.shape(), 0);
  cudu::device::Array2D<float> drain_distance_now(elevation.shape(), 0);
  cudu::device::Array2D<float> level(elevation.shape(), 0);

  for (size_t i = 0; i < iteration_limit; ++i) {
    std::swap(depth_now, depth_previously);
    std::swap(drain_distance_now, drain_distance_previously);
    CUDU_LAUNCH_BATCHES(
        lake_kernel,
        depth_now.size(),
        BATCH_SIZE,
        cudu::device::ref_const(elevation),
        cudu::device::ref_const(ocean),
        cudu::device::ref_const(depth_previously),
        cudu::device::ref_const(drain_distance_previously),
        cudu::device::ref(depth_now),
        cudu::device::ref(drain_distance_now),
        cudu::device::ref(level));
  }

  cudu::device::Array2D<float> adjusted_elevation(elevation.shape(), 0);

  CUDU_LAUNCH_BATCHES(
      lake_threshold_kernel,
      depth_now.size(),
      BATCH_SIZE,
      cudu::device::ref_const(elevation),
      cudu::device::ref_const(ice),
      depth_threshold,
      cudu::device::ref(depth_now),
      cudu::device::ref(adjusted_elevation));

  return {
      std::move(depth_now),
      std::move(drain_distance_now),
      std::move(level),
      std::move(adjusted_elevation)};
}

struct DrainCoords {
  static constexpr size_t INVALID = size_t(-1);

  size_t i = INVALID;
  size_t j = INVALID;

  __device__ bool valid() const { return i != INVALID && j != INVALID; }
};

__global__ void river_drain_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> elevation,
    const cudu::device::ConstArrayRef2D<bool> ocean,
    const cudu::device::ConstArrayRef2D<float> water_level,
    const cudu::device::ConstArrayRef2D<float> drain_distance,
    cudu::device::ArrayRef2D<DrainCoords> drain) {

  if (job_offset + CUDU_THREAD_ID() >= drain.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), drain.shape()[1], i, j);

  if (drain(i, j).valid()) {
    return;
  }

  if (ocean(i, j)) {
    drain(i, j) = {i, j};
    return;
  }

  const float local_level = water_level(i, j);

  DrainCoords local_drain;
  float local_drain_distance = drain.size();

  for (int di = -1; di <= 1; ++di) {
    for (int dj = -1; dj <= 1; ++dj) {

      const int ni = i + di;
      const int nj = j + dj;

      if (ni < 0 || ni >= drain.shape()[0] || nj < 0 ||
          nj >= drain.shape()[1] || (ni == i && nj == j)) {
        continue;
      }

      if (ocean(ni, nj)) {
        drain(i, j) = {size_t(ni), size_t(nj)};
        return;
      }

      const float neighbor_level = water_level(ni, nj);

      if (!drain(ni, nj).valid() || local_level < neighbor_level) {
        continue;
      }

      if (!local_drain.valid() ||
          drain_distance(ni, nj) < local_drain_distance) {
        local_drain = {size_t(ni), size_t(nj)};
        local_drain_distance = drain_distance(ni, nj);
      }
    }
  }

  drain(i, j) = local_drain;
}

__global__ void river_intensity_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<bool> ocean,
    const cudu::device::ConstArrayRef2D<float> precipitation,
    const cudu::device::ConstArrayRef2D<DrainCoords> drain,
    const size_t iteration_limit,
    const float precipitation_factor,
    cudu::device::ArrayRef2D<float> intensity,
    cudu::device::ArrayRef2D<float> adjusted_elevation) {

  if (job_offset + CUDU_THREAD_ID() >= intensity.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), intensity.shape()[1], i, j);

  if (ocean(i, j)) {
    return;
  }

  // Stream head coordinates.
  size_t hi = i;
  size_t hj = j;

  for (size_t iteration_index = 0;
       iteration_index < iteration_limit && !ocean(hi, hj);
       ++iteration_index) {

    intensity(hi, hj) += precipitation(i, j) * precipitation_factor;

    const auto local_drain = drain(hi, hj);
    if (local_drain.valid()) {
      if (adjusted_elevation(local_drain.i, local_drain.j) >
          adjusted_elevation(hi, hj)) {
        adjusted_elevation(local_drain.i, local_drain.j) =
            adjusted_elevation(hi, hj);
      }
      hi = local_drain.i;
      hj = local_drain.j;
    } else {
      return;
    }
  }
}

__global__ void river_intensity_threshold_kernel(
    const size_t job_offset,
    const float threshold,
    cudu::device::ArrayRef2D<float> intensity) {

  const size_t i = job_offset + CUDU_THREAD_ID();
  if (i < intensity.size()) {
    if (intensity[i] < threshold) {
      intensity[i] = 0;
    }
  }
}

__global__ void river_erosion_kernel(
    const size_t job_offset,
    const float erosion_factor,
    cudu::device::ArrayRef2D<float> intensity,
    cudu::device::ArrayRef2D<float> adjusted_elevation) {

  if (job_offset + CUDU_THREAD_ID() >= intensity.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), intensity.shape()[1], i, j);

  if (intensity(i, j) > 0) {
    return;
  }

  float intensity_max = 0;
  size_t intense_i = 0;
  size_t intense_j = 0;

  for (int di = -1; di <= 1; ++di) {
    for (int dj = -1; dj <= 1; ++dj) {

      const int ni = i + di;
      const int nj = j + dj;

      if (ni < 0 || ni >= intensity.shape()[0] || nj < 0 ||
          nj >= intensity.shape()[1] || (ni == i && nj == j)) {
        continue;
      }

      if (intensity(ni, nj) > intensity_max) {
        intensity_max = intensity(ni, nj);
        intense_i = ni;
        intense_j = nj;
      }
    }
  }

  if (intensity_max > 0) {
    adjusted_elevation(i, j) -= intensity_max * erosion_factor;
    if (adjusted_elevation(i, j) < adjusted_elevation(intense_i, intense_j)) {
      intensity(i, j) = intensity(intense_i, intense_j);
    }
  }
}

__global__ void river_depth_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> elevation,
    const cudu::device::ConstArrayRef2D<bool> ice,
    const cudu::device::ConstArrayRef2D<float> lake_depth,
    const cudu::device::ConstArrayRef2D<float> intensity,
    const float depth_factor,
    cudu::device::ArrayRef2D<float> depth,
    cudu::device::ArrayRef2D<float> adjusted_elevation) {

  const size_t i = job_offset + CUDU_THREAD_ID();
  if (i < depth.size()) {
    if (lake_depth[i] > 0) {
      adjusted_elevation[i] = elevation[i];
    } else if (!ice[i]) {
      depth[i] = intensity[i] * depth_factor;
      adjusted_elevation[i] -= depth[i];
    }
  }
}

worldgen::RiverDatasets worldgen::rivers(
    const cudu::device::Array2D<float>& elevation,
    const cudu::device::Array2D<bool>& ocean,
    const cudu::device::Array2D<bool>& ice,
    const cudu::device::Array2D<float>& precipitation,
    const cudu::device::Array2D<float>& lake_depth,
    const cudu::device::Array2D<float>& lake_level,
    const cudu::device::Array2D<float>& lake_drain_distance,
    const size_t iteration_limit,
    const float precipitation_factor,
    const size_t erosion_iterations,
    const float erosion_factor,
    const float intensity_threshold,
    const float depth_factor) {

  cudu::device::Array2D<DrainCoords> drain(elevation.shape(), DrainCoords());

  for (size_t i = 0; i < iteration_limit; ++i) {
    CUDU_LAUNCH_BATCHES(
        river_drain_kernel,
        drain.size(),
        BATCH_SIZE,
        cudu::device::ref_const(elevation),
        cudu::device::ref_const(ocean),
        cudu::device::ref_const(lake_level),
        cudu::device::ref_const(lake_drain_distance),
        cudu::device::ref(drain));
  }

  cudu::device::Array2D<float> intensity(elevation.shape(), 0);
  cudu::device::Array2D<float> adjusted_elevation = elevation.copy();

  CUDU_LAUNCH_BATCHES(
      river_intensity_kernel,
      intensity.size(),
      BATCH_SIZE,
      cudu::device::ref_const(ocean),
      cudu::device::ref_const(precipitation),
      cudu::device::ref_const(drain),
      iteration_limit,
      precipitation_factor,
      cudu::device::ref(intensity),
      cudu::device::ref(adjusted_elevation));

  CUDU_LAUNCH_BATCHES(
      river_intensity_threshold_kernel,
      intensity.size(),
      BATCH_SIZE,
      intensity_threshold,
      cudu::device::ref(intensity));

  for (size_t i = 0; i < erosion_iterations; ++i) {
    CUDU_LAUNCH_BATCHES(
        river_erosion_kernel,
        intensity.size(),
        BATCH_SIZE,
        erosion_factor,
        cudu::device::ref(intensity),
        cudu::device::ref(adjusted_elevation));
  }

  cudu::device::Array2D<float> depth(elevation.shape(), 0);

  CUDU_LAUNCH_BATCHES(
      river_depth_kernel,
      depth.size(),
      BATCH_SIZE,
      cudu::device::ref_const(elevation),
      cudu::device::ref_const(ice),
      cudu::device::ref_const(lake_depth),
      cudu::device::ref_const(intensity),
      depth_factor,
      cudu::device::ref(depth),
      cudu::device::ref(adjusted_elevation));

  return {
      std::move(intensity), std::move(depth), std::move(adjusted_elevation)};
}

__device__ float biome_qualifier_score(
    const worldgen::Range& range,
    const float value) {
  const auto distance_to_range = value < range.min   ? range.min - value
                                 : value > range.max ? value - range.max
                                                     : 0;
  const auto relative_distance_to_range =
      distance_to_range / (range.max - range.min);
  return 1 - fminf(relative_distance_to_range, 0.995);
}

__global__ void biomes_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<bool> ocean,
    const cudu::device::ConstArrayRef2D<float> temperature,
    const cudu::device::ConstArrayRef2D<bool> ice,
    const cudu::device::ConstArrayRef2D<float> precipitation,
    const cudu::device::ConstArrayRef2D<float> lake_depth,
    const cudu::device::ConstArrayRef2D<float> river_depth,
    const cudu::device::ConstArrayRef2D<float> river_distance,
    const cudu::device::ConstArrayRef1D<worldgen::BiomeSpec> biome_specs,
    cudu::device::ArrayRef2D<worldgen::BiomeMixture> biomes) {

  const size_t i = job_offset + CUDU_THREAD_ID();
  if (i >= biomes.size() || ocean[i] || lake_depth[i] > 0 ||
      river_depth[i] > 0) {
    return;
  }

  worldgen::BiomeMixture mixture;  // Invariant: score[0] >= score[1]

  for (size_t biome_index = 0; biome_index < biome_specs.size();
       ++biome_index) {

    const auto spec = biome_specs[biome_index];
    const auto score =
        biome_qualifier_score(spec.temperature_range, temperature[i]) *
        biome_qualifier_score(spec.precipitation_range, precipitation[i]);

    if (score > mixture.score[0]) {
      mixture.index[1] = mixture.index[0];
      mixture.score[1] = mixture.score[0];
      mixture.index[0] = biome_index;
      mixture.score[0] = score;
    } else if (score > mixture.score[1]) {
      mixture.index[1] = biome_index;
      mixture.score[1] = score;
    }
  }

  biomes[i] = mixture;
}

cudu::device::Array2D<worldgen::BiomeMixture> worldgen::biomes(
    const cudu::device::Array2D<bool>& ocean,
    const cudu::device::Array2D<float>& temperature,
    const cudu::device::Array2D<bool>& ice,
    const cudu::device::Array2D<float>& precipitation,
    const cudu::device::Array2D<float>& lake_depth,
    const cudu::device::Array2D<float>& river_depth,
    const cudu::device::Array2D<float>& river_distance,
    const cudu::device::Array1D<BiomeSpec>& biome_specs) {

  cudu::device::Array2D<worldgen::BiomeMixture> result(
      ocean.shape(), BiomeMixture());
  CUDU_LAUNCH_BATCHES(
      biomes_kernel,
      result.size(),
      BATCH_SIZE,
      cudu::device::ref_const(ocean),
      cudu::device::ref_const(temperature),
      cudu::device::ref_const(ice),
      cudu::device::ref_const(precipitation),
      cudu::device::ref_const(lake_depth),
      cudu::device::ref_const(river_depth),
      cudu::device::ref_const(river_distance),
      cudu::device::ref_const(biome_specs),
      cudu::device::ref(result));
  return result;
}

__global__ void biome_image_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<bool> ocean,
    const cudu::device::ConstArrayRef2D<float> lake_depth,
    const cudu::device::ConstArrayRef2D<float> river_depth,
    const cudu::device::ConstArrayRef2D<worldgen::BiomeMixture> biomes,
    const cudu::device::ConstArrayRef1D<worldgen::BiomeSpec> biome_specs,
    const worldgen::Rgb aquatic_biome_color,
    cudu::device::ArrayRef3D<unsigned char> image) {

  if (job_offset + CUDU_THREAD_ID() >= biomes.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), biomes.shape()[1], i, j);

  if (ocean(i, j) || lake_depth(i, j) > 0 || river_depth(i, j) > 0) {
    image(i, j, 0) = aquatic_biome_color.b;
    image(i, j, 1) = aquatic_biome_color.g;
    image(i, j, 2) = aquatic_biome_color.r;
    return;
  }

  if (!biomes(i, j).valid()) {
    return;
  }

  const auto biome_index = biomes(i, j).index[0];
  const auto color = biome_specs[biome_index].dense_color;

  image(i, j, 0) = color.b;
  image(i, j, 1) = color.g;
  image(i, j, 2) = color.r;
}

cudu::device::Array3D<unsigned char> worldgen::biome_image(
    const cudu::device::Array2D<bool>& ocean,
    const cudu::device::Array2D<float>& lake_depth,
    const cudu::device::Array2D<float>& river_depth,
    const cudu::device::Array2D<BiomeMixture>& biomes,
    const cudu::device::Array1D<BiomeSpec>& biome_specs,
    const Rgb& aquatic_biome_color) {

  cudu::device::Array3D<unsigned char> image(
      {biomes.shape()[0], biomes.shape()[1], 3});
  CUDU_LAUNCH_BATCHES(
      biome_image_kernel,
      biomes.size(),
      BATCH_SIZE,
      cudu::device::ref_const(ocean),
      cudu::device::ref_const(lake_depth),
      cudu::device::ref_const(river_depth),
      cudu::device::ref_const(biomes),
      cudu::device::ref_const(biome_specs),
      aquatic_biome_color,
      cudu::device::ref(image));
  return image;
}

__device__ float lerp(const float a, const float b, const float ratio) {
  return a * (1.0f - ratio) + b * ratio;
}

__device__ worldgen::Rgb lerp_rgb(
    const worldgen::Rgb& from,
    const worldgen::Rgb& to,
    const float ratio) {
  return worldgen::Rgb{
      unsigned char(lroundf(lerp(from.r, to.r, ratio))),
      unsigned char(lroundf(lerp(from.g, to.g, ratio))),
      unsigned char(lroundf(lerp(from.b, to.b, ratio)))};
}

__device__ worldgen::Rgb ease_out_rgb(
    const worldgen::Rgb& from,
    const worldgen::Rgb& to,
    const float ratio) {
  return lerp_rgb(from, to, 1 - powf(1 - ratio, 2));
}

__global__ void combined_image_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> elevation,
    const cudu::device::ConstArrayRef2D<bool> ocean,
    const cudu::device::ConstArrayRef2D<float> lake_depth,
    const cudu::device::ConstArrayRef2D<float> river_depth,
    const cudu::device::ConstArrayRef1D<worldgen::BiomeSpec> biome_specs,
    const cudu::device::ConstArrayRef2D<worldgen::BiomeMixture> biomes,
    const float deep_water_threshold,
    const worldgen::Rgb deep_water_color,
    const worldgen::Rgb shallow_water_color,
    const worldgen::InterpolationKind color_interpolation_kind,
    cudu::device::ArrayRef3D<unsigned char> image) {

  if (job_offset + CUDU_THREAD_ID() >= elevation.size()) {
    return;
  }

  size_t i;
  size_t j;
  unravel_index(job_offset + CUDU_THREAD_ID(), elevation.shape()[1], i, j);

  worldgen::Rgb color;

  if (ocean(i, j)) {
    const float depth_ratio =
        fminf(fabsf(elevation(i, j) / deep_water_threshold), 1);
    color = lerp_rgb(shallow_water_color, deep_water_color, depth_ratio);

  } else if (lake_depth(i, j) > 0) {
    const float depth_ratio = fminf(lake_depth(i, j) / deep_water_threshold, 1);
    color = lerp_rgb(shallow_water_color, deep_water_color, depth_ratio);

  } else if (river_depth(i, j) > 0) {
    const float depth_ratio =
        fminf(river_depth(i, j) / deep_water_threshold, 1);
    color = lerp_rgb(shallow_water_color, deep_water_color, depth_ratio);

  } else if (biomes(i, j).valid()) {
    const auto interpolate_color =
        color_interpolation_kind == worldgen::InterpolationKind::EaseOut
            ? ease_out_rgb
            : lerp_rgb;
    const worldgen::BiomeSpec dominant_biome_specs[2] = {
        biome_specs[biomes(i, j).index[0]],
        biome_specs[biomes(i, j).index[1]],
    };
    const worldgen::Rgb dominant_biome_colors[2] = {
        lerp_rgb(
            dominant_biome_specs[0].sparse_color,
            dominant_biome_specs[0].dense_color,
            biomes(i, j).score[0]),
        lerp_rgb(
            dominant_biome_specs[1].sparse_color,
            dominant_biome_specs[1].dense_color,
            biomes(i, j).score[1])};
    color = interpolate_color(
        dominant_biome_colors[0],
        dominant_biome_colors[1],
        biomes(i, j).score[1] /
            (biomes(i, j).score[0] + biomes(i, j).score[1]));
  }

  image(i, j, 0) = color.b;
  image(i, j, 1) = color.g;
  image(i, j, 2) = color.r;
}

cudu::device::Array3D<unsigned char> worldgen::combined_image(
    const cudu::device::Array2D<float>& elevation,
    const cudu::device::Array2D<bool>& ocean,
    const cudu::device::Array2D<float>& lake_depth,
    const cudu::device::Array2D<float>& river_depth,
    const cudu::device::Array1D<BiomeSpec>& biome_specs,
    const cudu::device::Array2D<BiomeMixture>& biomes,
    const float deep_water_threshold,
    const Rgb& deep_water_color,
    const Rgb& shallow_water_color,
    const InterpolationKind color_interpolation_kind) {

  cudu::device::Array3D<unsigned char> image(
      {elevation.shape()[0], elevation.shape()[1], 3});

  CUDU_LAUNCH_BATCHES(
      combined_image_kernel,
      elevation.size(),
      BATCH_SIZE,
      cudu::device::ref_const(elevation),
      cudu::device::ref_const(ocean),
      cudu::device::ref_const(lake_depth),
      cudu::device::ref_const(river_depth),
      cudu::device::ref_const(biome_specs),
      cudu::device::ref_const(biomes),
      deep_water_threshold,
      deep_water_color,
      shallow_water_color,
      color_interpolation_kind,
      cudu::device::ref(image));

  return image;
}

__global__ void heightmap_kernel(
    const size_t job_offset,
    const cudu::device::ConstArrayRef2D<float> elevation,
    const cudu::device::ConstArrayRef2D<bool> ocean,
    const cudu::device::ConstArrayRef2D<float> lake_depth,
    cudu::device::ArrayRef2D<float> heightmap) {

  const size_t i = job_offset + CUDU_THREAD_ID();

  if (i < heightmap.size()) {
    heightmap[i] = ocean[i] ? 0 : elevation[i] + lake_depth[i];
  }
}

cudu::device::Array2D<float> worldgen::heightmap(
    const cudu::device::Array2D<float>& elevation,
    const cudu::device::Array2D<bool>& ocean,
    const cudu::device::Array2D<float>& lake_depth) {

  cudu::device::Array2D<float> result(elevation.shape());

  CUDU_LAUNCH_BATCHES(
      heightmap_kernel,
      result.size(),
      BATCH_SIZE,
      cudu::device::ref_const(elevation),
      cudu::device::ref_const(ocean),
      cudu::device::ref_const(lake_depth),
      cudu::device::ref(result));

  return result;
}