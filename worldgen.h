/**
 * Copyright (c) 2020 Raoul Harel
 * All rights reserved
 */

#pragma once

#include <array>
#include <stdint.h>
#include <vector>

#include "cudu.h"
#include "model.h"

namespace worldgen 
{
    cudu::device::Array2D<float> altitude(
        uint32_t level_of_detail,
        float noise_initial,
        float noise_scale_factor,
        float bias,
        const cudu::device::Array2D<float>& bias_map,
        float bias_map_shift,
        float bias_map_stretch,
        uint64_t rng_seed);

    cudu::device::Array3D<float> material(
        const cudu::Shape2D& shape,
        const std::vector<SurfaceMaterial>& materials,
        unsigned cell_count,
        unsigned cell_blur,
        uint64_t rng_seed);

    cudu::device::Array2D<float> temperature(
        const cudu::device::Array2D<float>& altitude,
        float latitude_factor,
        float altitude_factor,
        float noise,
        float bias,
        uint64_t rng_seed);

    cudu::device::Array2D<bool> ocean_mask(
        const cudu::device::Array2D<float>& altitude,
        const cudu::device::Array2D<float>& temperature,
        float boil_temperature);
    
    cudu::device::Array2D<bool> ice_mask(
        const cudu::device::Array2D<float>& temperature,
        const cudu::device::Array2D<bool>& ocean_mask,
        float ocean_freeze_temperature);

    cudu::device::Array2D<float> precipitation(
        const cudu::device::Array2D<float>& temperature,
        const cudu::device::Array2D<bool>& ocean_mask,
        const std::vector<ClimateZone>& climate_zones,
        const worldgen::VicinityLookupParams& ocean_lookup_params,
        float ocean_distance_smoothing,
        float ocean_distance_factor);

    cudu::device::Array2D<float> rivers(
        const cudu::device::Array2D<float>& altitude,
        const cudu::device::Array2D<bool>& ocean_mask,
        const cudu::device::Array2D<bool>& ice_mask,
        float block_size,
        const worldgen::VicinityLookupParams& ocean_lookup_params);

    cudu::device::Array3D<unsigned char> image(
        const cudu::device::Array2D<float>& altitude,
        const cudu::device::Array3D<float>& material,
        const cudu::device::Array2D<float>& temperature,
        const cudu::device::Array2D<bool>& ocean_mask,
        const cudu::device::Array2D<bool>& ice_mask,
        const cudu::device::Array2D<float>& precipitation,
        const cudu::device::Array2D<float>& river_depth,
        const std::vector<worldgen::ClimateZone>& climate_zones,
        const RenderingOptions& rendering_options);
}
