/**
 * Copyright (c) 2020 Raoul Harel
 * All rights reserved
 */

#pragma once

#include "cudu.h"

namespace worldgen
{
    struct Rgb
    {
        unsigned char r = 0;
        unsigned char g = 0;
        unsigned char b = 0;

        bool operator==(const Rgb& other) const
        {
            return (
                r == other.r && 
                g == other.g &&
                b == other.b
            );
        }

        bool operator!=(const Rgb& other) const
        {
            return !((*this) == other);
        }
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
        cudu::device::ArrayRef1D<MaterialVisual> materials;
        cudu::device::ArrayRef1D<ClimateZoneVisual> climate_zones;
        
        Rgb water_shallow;
        Rgb water_deep;
        float water_depth_limit;

        Rgb river_shallow;
        Rgb river_deep;
        float river_depth_limit;
        
        Rgb ice_thin;
        Rgb ice_thick;
        float ice_temperature_limit;
    };

    struct SurfaceMaterial
    {
        float likelihood;
        Rgb visual_altitude_low;
        Rgb visual_altitude_high;

        bool operator==(const SurfaceMaterial& other) const
        {
            return (
                likelihood == other.likelihood &&
                visual_altitude_low == other.visual_altitude_low &&
                visual_altitude_high == other.visual_altitude_high
            );
        }

        bool operator!=(const SurfaceMaterial& other) const
        {
            return !((*this) == other);
        }
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

        __host__ __device__ bool contains(float value) const
        {
            return min <= value && value <= max;
        }

        __host__ __device__ float distance_to(float value) const
        {
            return (
                value < min ? min - value :
                value > max ? value - max :
                0
            );
        }

        __host__ __device__ float distance_relative_to(float value) const
        {
            return distance_to(value) / size();
        }

        bool operator==(const Range& other) const
        {
            return (
                min == other.min &&
                max == other.max
            );
        }

        bool operator!=(const Range& other) const
        {
            return !((*this) == other);
        }
    };

    struct ClimateZone
    {
        Range latitude_range;
        Range temperature_range;
        Range precipitation_range;
        Rgb visual_sparse;
        Rgb visual_dense;

        bool operator==(const ClimateZone& other) const
        {
            return (
                latitude_range == other.latitude_range &&
                temperature_range == other.temperature_range &&
                precipitation_range == other.precipitation_range &&
                visual_sparse == other.visual_sparse &&
                visual_dense == other.visual_dense
            );
        }

        bool operator!=(const ClimateZone& other) const
        {
            return !((*this) == other);
        }
    };

    struct VicinityLookupParams
    {
        float radius;
        unsigned nr_samples;
        uint64_t rng_seed;

        bool operator==(const VicinityLookupParams& other) const
        {
            return (
                radius == other.radius &&
                nr_samples == other.nr_samples &&
                rng_seed == other.rng_seed
            );
        }

        bool operator!=(const VicinityLookupParams& other) const
        {
            return !((*this) == other);
        }
    };

    enum class Component
    {
        Altitude,
        Material,
        Temperature,
        Ocean,
        Ice,
        Precipitation,
        Rivers,
        Image
    };
}
