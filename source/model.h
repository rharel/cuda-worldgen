/**
 * Copyright (c) 2022 Raoul Harel
 * All rights reserved
 */

#pragma once

#include <string>
#include <unordered_set>

#include "cudu.h"

namespace worldgen {

  struct Rgb {
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;

    bool operator==(const Rgb& other) const {
      return r == other.r && g == other.g && b == other.b;
    }

    bool operator!=(const Rgb& other) const { return !(*this == other); }
  };

  enum class InterpolationKind { Linear, EaseOut };

  struct Range {
    float min = 0;
    float max = 0;

    float size() const { return max - min; }

    bool empty() const { return max <= min; }

    __host__ __device__ bool contains(const float value) const {
      return min <= value && value <= max;
    }

    bool operator==(const Range& other) const {
      return min == other.min && max == other.max;
    }

    bool operator!=(const Range& other) const { return !(*this == other); }
  };

  struct BiomeSpec {
    Range temperature_range;
    Range precipitation_range;
    Rgb sparse_color;
    Rgb dense_color;

    bool operator==(const BiomeSpec& other) const {
      return temperature_range == other.temperature_range &&
             precipitation_range == other.precipitation_range &&
             sparse_color == other.sparse_color &&
             dense_color == other.dense_color;
    }

    bool operator!=(const BiomeSpec& other) const { return !(*this == other); }
  };

  struct BiomeMixture {
    size_t index[2] = {0, 0};
    float score[2] = {-1, -1};

    __device__ bool valid() const { return score[0] >= 0 && score[1] >= 0; }
  };

  cudu::device::Array2D<unsigned char> colormap(const std::string& path);

  enum class DatasetKind {
    Elevation,
    Ocean,
    OceanDistance,
    Temperature,
    Ice,
    Precipitation,
    Lakes,
    Rivers,
    RiverDistance,
    Biomes,
    Heightmap,
    Last
  };

  enum class ImageKind {
    Elevation,
    Ocean,
    OceanDistance,
    Temperature,
    Ice,
    Precipitation,
    LakeDepth,
    LakeLevel,
    LakeDrainDistance,
    RiverIntensity,
    RiverDepth,
    RiverDistance,
    Biomes,
    Combined,
    Heightmap
  };

  std::unordered_set<DatasetKind> dependencies_of_image(ImageKind image_kind);

  std::unordered_set<DatasetKind> dependencies_of_dataset(
      DatasetKind dataset_kind);

  std::unordered_set<DatasetKind> dependents_of_dataset(
      DatasetKind dataset_kind);

}  // namespace worldgen
