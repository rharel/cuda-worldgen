#pragma once

#include "cudu.h"

namespace worldgen
{
    struct Rgb
    {
        unsigned char r;
        unsigned char g;
        unsigned char b;
    };

    struct MaterialVisual
    {
        Rgb low_altitude;
        Rgb high_altitude;
    };

    struct ClimateZoneVisual
    {
        Rgb sparse;
        Rgb dense;
    };

    struct RenderingOptions
    {
        cudu::ArrayRef1D<MaterialVisual> materials;
        
        Rgb water_shallow;
        Rgb water_deep;
        float water_depth_limit;
        
        Rgb ice_thin;
        Rgb ice_thick;
        float ice_temperature_limit;

        cudu::ArrayRef1D<ClimateZoneVisual> climate_zones;
    };

    struct SurfaceMaterial
    {
        float likelihood;
        Rgb visual_altitude_low;
        Rgb visual_altitude_high;
    };

    struct MaterialCell
    {
        size_t material_index;
        size_t row;
        size_t col;
    };

    struct Range
    {
        float min;
        float max;

        __host__ __device__ float size() const
        {
            return max - min;
        }

        __host__ __device__ float center() const
        {
            return (min + max) / 2;
        }
    };

    struct ClimateZone
    {
        Range latitude_range;
        Range temperature_range;
        Range precipitation_range;
        Rgb visual_sparse;
        Rgb visual_dense;
    };

    struct VicinityLookupParams
    {
        float radius;
        float radius_step;
        unsigned nr_samples;
        uint64_t rng_seed;
    };
}
