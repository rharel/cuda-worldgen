#pragma once

#include <array>
#include <stdint.h>
#include <vector>

#include "cudu.h"
#include "model.h"

namespace worldgen 
{
    cudu::Array2D<float> surface_altitude(
        uint32_t level_of_detail,
        float noise_initial,
        float noise_scale_factor,
        float bias,
        uint64_t rng_seed,
        unsigned max_nr_threads);

    cudu::Array3D<float> surface_material(
        const std::array<size_t, 2>& shape,
        const std::vector<SurfaceMaterial>& materials,
        unsigned cell_count,
        float cell_blur,
        uint64_t rng_seed,
        unsigned max_nr_threads);

    cudu::Array2D<float> temperature(
        const cudu::Array2D<float>& surface_altitude,
        float latitude_factor,
        float altitude_factor,
        float noise,
        float bias,
        uint64_t rng_seed,
        unsigned max_nr_threads);

    cudu::Array2D<bool> ocean(
        const cudu::Array2D<float>& surface_altitude,
        const cudu::Array2D<float>& temperature,
        float boil_temperature,
        unsigned max_nr_threads);
    
    cudu::Array2D<bool> ice(
        const cudu::Array2D<float>& temperature,
        const cudu::Array2D<bool>& ocean_mask,
        float ocean_freeze_temperature,
        unsigned max_nr_threads);

    cudu::Array2D<float> precipitation(
        const cudu::Array2D<float>& temperature,
        const cudu::Array2D<bool>& ocean_mask,
        const std::vector<ClimateZone>& climate_zones,
        float climate_zone_weight_min,
        const worldgen::VicinityLookupParams& ocean_lookup_params,
        float ocean_distance_factor,
        unsigned max_nr_threads);

    cudu::Array2D<float> rivers(
        const cudu::Array2D<float>& surface_altitude,
        const cudu::Array2D<bool>& ocean_mask,
        const cudu::Array2D<bool>& ice_mask,
        size_t block_size,
        const worldgen::VicinityLookupParams& ocean_lookup_params,
        unsigned max_nr_threads);

    cudu::Array3D<unsigned char> image(
        const cudu::Array2D<float>& surface_altitude,
        const cudu::Array3D<float>& surface_material,
        const cudu::Array2D<float>& temperature,
        const cudu::Array2D<bool>& ocean_mask,
        const cudu::Array2D<bool>& ice_mask,
        const cudu::Array2D<float>& precipitation,
        const cudu::Array2D<float>& river_depth,
        const std::vector<worldgen::ClimateZone>& climate_zones,
        const RenderingOptions& rendering_options,
        unsigned max_nr_threads);
}
