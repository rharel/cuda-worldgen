/**
 * Copyright (c) 2022 Raoul Harel
 * All rights reserved
 */

#pragma once

#include <stdint.h>

#include "cudu.h"
#include "model.h"

namespace worldgen {

  cudu::device::Array3D<unsigned char> binary_image(
      const cudu::device::Array2D<bool>& data,
      const Rgb& truth_color);

  cudu::device::Array3D<unsigned char> colormap_image(
      const cudu::device::Array2D<float>& data,
      const cudu::device::Array2D<unsigned char>& colormap,
      float min,
      float max,
      float mid);

  cudu::device::Array3D<unsigned char> colormap_image(
      const cudu::device::Array2D<float>& data,
      const cudu::device::Array2D<unsigned char>& colormap,
      float min,
      float max);

  cudu::device::Array2D<float> distance(
      const cudu::device::Array2D<bool>& mask,
      uint32_t limit);

  cudu::device::Array2D<float> elevation(
      size_t width,
      size_t height,
      size_t nr_layers,
      float scale,
      float lacunarity,
      float decay,
      float bias,
      float stretch,
      int seed);

  cudu::device::Array2D<bool> ocean(
      const cudu::device::Array2D<float>& elevation);

  cudu::device::Array2D<float> temperature(
      const cudu::device::Array2D<float>& elevation,
      const cudu::device::Array2D<float>& ocean_distance,
      float latitude_difference,
      float elevation_factor,
      float ocean_difference,
      float ocean_difference_falloff,
      float noise,
      float bias,
      uint64_t rng_seed);

  cudu::device::Array2D<bool> ice(
      const cudu::device::Array2D<bool>& ocean,
      const cudu::device::Array2D<float>& temperature,
      float temperature_threshold_land,
      float temperature_threshold_ocean);

  cudu::device::Array2D<float> precipitation(
      const cudu::device::Array2D<float>& ocean_distance,
      const cudu::device::Array2D<float>& temperature,
      float ocean_distance_factor,
      float multiplier,
      const std::array<float, 4>& temperature_effect_coefficients);

  struct LakeDatasets {
    cudu::device::Array2D<float> depth;
    cudu::device::Array2D<float> drain_distance;
    cudu::device::Array2D<float> level;
    cudu::device::Array2D<float> adjusted_elevation;
  };
  LakeDatasets lakes(
      const cudu::device::Array2D<float>& elevation,
      const cudu::device::Array2D<bool>& ocean,
      const cudu::device::Array2D<bool>& ice,
      size_t iteration_limit,
      float depth_threshold);

  struct RiverDatasets {
    cudu::device::Array2D<float> intensity;
    cudu::device::Array2D<float> depth;
    cudu::device::Array2D<float> adjusted_elevation;
  };
  RiverDatasets rivers(
      const cudu::device::Array2D<float>& elevation,
      const cudu::device::Array2D<bool>& ocean,
      const cudu::device::Array2D<bool>& ice,
      const cudu::device::Array2D<float>& precipitation,
      const cudu::device::Array2D<float>& lake_depth,
      const cudu::device::Array2D<float>& lake_level,
      const cudu::device::Array2D<float>& lake_drain_distance,
      size_t iteration_limit,
      float precipitation_factor,
      size_t erosion_iterations,
      float erosion_factor,
      float intensity_threshold,
      float depth_factor);

  cudu::device::Array2D<BiomeMixture> biomes(
      const cudu::device::Array2D<bool>& ocean,
      const cudu::device::Array2D<float>& temperature,
      const cudu::device::Array2D<bool>& ice,
      const cudu::device::Array2D<float>& precipitation,
      const cudu::device::Array2D<float>& lake_depth,
      const cudu::device::Array2D<float>& river_depth,
      const cudu::device::Array2D<float>& river_distance,
      const cudu::device::Array1D<BiomeSpec>& biome_specs);

  cudu::device::Array3D<unsigned char> biome_image(
      const cudu::device::Array2D<bool>& ocean,
      const cudu::device::Array2D<float>& lake_depth,
      const cudu::device::Array2D<float>& river_depth,
      const cudu::device::Array2D<BiomeMixture>& biomes,
      const cudu::device::Array1D<BiomeSpec>& biome_specs,
      const Rgb& aquatic_biome_color);

  cudu::device::Array3D<unsigned char> combined_image(
      const cudu::device::Array2D<float>& elevation,
      const cudu::device::Array2D<bool>& ocean,
      const cudu::device::Array2D<float>& lake_depth,
      const cudu::device::Array2D<float>& river_depth,
      const cudu::device::Array1D<BiomeSpec>& biome_specs,
      const cudu::device::Array2D<BiomeMixture>& biomes,
      float deep_water_threshold,
      const Rgb& deep_water_color,
      const Rgb& shallow_water_color,
      InterpolationKind color_interpolation_kind);

  cudu::device::Array2D<float> heightmap(
      const cudu::device::Array2D<float>& elevation,
      const cudu::device::Array2D<bool>& ocean,
      const cudu::device::Array2D<float>& lake_depth);

}  // namespace worldgen
