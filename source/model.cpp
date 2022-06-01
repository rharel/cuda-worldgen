/**
 * Copyright (c) 2022 Raoul Harel
 * All rights reserved
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "model.h"

cudu::device::Array2D<unsigned char> worldgen::colormap(
    const std::string& path) {

  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("failed to read colormap: " + path);
  }
  return cudu::device::Array2D<unsigned char>::from_ptr(
      image.data, image.cols, 3);
}

std::unordered_set<worldgen::DatasetKind> worldgen::dependencies_of_image(
    const worldgen::ImageKind image_kind) {

  switch (image_kind) {
    case ImageKind::Elevation:
      return {DatasetKind::Elevation};
    case ImageKind::Ocean:
      return {DatasetKind::Ocean};
    case ImageKind::OceanDistance:
      return {DatasetKind::OceanDistance};
    case ImageKind::Temperature:
      return {DatasetKind::Temperature};
    case ImageKind::Ice:
      return {DatasetKind::Ice};
    case ImageKind::Precipitation:
      return {DatasetKind::Precipitation};
    case ImageKind::LakeDepth:
    case ImageKind::LakeLevel:
    case ImageKind::LakeDrainDistance:
      return {DatasetKind::Lakes};
    case ImageKind::RiverIntensity:
    case ImageKind::RiverDepth:
      return {DatasetKind::Rivers};
    case ImageKind::RiverDistance:
      return {DatasetKind::RiverDistance};
    case ImageKind::Biomes:
      return {DatasetKind::Biomes};
    case ImageKind::Combined:
      return {
          DatasetKind::Elevation,
          DatasetKind::Ocean,
          DatasetKind::Rivers,
          DatasetKind::Biomes};
    case ImageKind::Heightmap:
      return {DatasetKind::Heightmap};
    default:
      return {};
  }
}

std::unordered_set<worldgen::DatasetKind> worldgen::dependencies_of_dataset(
    const worldgen::DatasetKind dataset_kind) {

  switch (dataset_kind) {
    case DatasetKind::Ocean:
      return {DatasetKind::Elevation};
    case DatasetKind::OceanDistance:
      return {DatasetKind::Ocean};
    case DatasetKind::Temperature:
      return {DatasetKind::Elevation, DatasetKind::OceanDistance};
    case DatasetKind::Ice:
      return {DatasetKind::Ocean, DatasetKind::Temperature};
    case DatasetKind::Precipitation:
      return {DatasetKind::OceanDistance, DatasetKind::Temperature};
    case DatasetKind::Lakes:
      return {DatasetKind::Elevation, DatasetKind::Ocean, DatasetKind::Ice};
    case DatasetKind::Rivers:
      return {
          DatasetKind::Elevation,
          DatasetKind::Ocean,
          DatasetKind::Ice,
          DatasetKind::Precipitation,
          DatasetKind::Lakes};
    case DatasetKind::RiverDistance:
      return {DatasetKind::Rivers};
    case DatasetKind::Biomes:
      return {
          DatasetKind::Ocean,
          DatasetKind::Temperature,
          DatasetKind::Precipitation,
          DatasetKind::Rivers,
          DatasetKind::RiverDistance};
    case DatasetKind::Heightmap:
      return {DatasetKind::Elevation, DatasetKind::Ocean, DatasetKind::Rivers};
    default:
      return {};
  }
}

std::unordered_set<worldgen::DatasetKind> worldgen::dependents_of_dataset(
    DatasetKind dataset_kind) {

  using DatasetKind_t = std::underlying_type_t<DatasetKind>;
  std::unordered_set<DatasetKind> dependents;
  for (DatasetKind_t i = 0; i < static_cast<DatasetKind_t>(DatasetKind::Last);
       ++i) {

    const auto dataset_kind_i = static_cast<DatasetKind>(i);
    const auto dependencies = dependencies_of_dataset(dataset_kind_i);
    if (dependencies.find(dataset_kind) != dependencies.end()) {
      dependents.insert(dataset_kind_i);
    }
  }
  return dependents;
}