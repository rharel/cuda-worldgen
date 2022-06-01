/**
 * Copyright (c) 2022 Raoul Harel
 * All rights reserved
 */

#pragma once

#include <stdint.h>
#include <array>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include "../external/json.hpp"

#include "model.h"

namespace worldgen {

  void from_json(const nlohmann::json& json, Rgb& color) {
    if (!json.is_string()) {
      throw std::runtime_error(
          "failed to parse color: expected string, got: " +
          std::string(json.type_name()));
    }

    const std::regex color_regex("^#[0-9a-fA-F]{6}$");
    const auto color_str = json.get<std::string>();
    if (!std::regex_match(color_str, color_regex)) {
      throw std::runtime_error("failed to parse hex color: " + color_str);
    }

    const auto hex_str = color_str.substr(1);
    const unsigned long hex = std::stoul(hex_str, nullptr, 16);
    color.r = unsigned char((hex & 0xff0000) >> 16);
    color.g = unsigned char((hex & 0x00ff00) >> 8);
    color.b = unsigned char((hex & 0x0000ff) >> 0);
  }

  void from_json(
      const nlohmann::json& json,
      InterpolationKind& interpolation_kind) {

    if (!json.is_string()) {
      throw std::runtime_error(
          "failed to parse interpolation kind: expected string, got: " +
          std::string(json.type_name()));
    }

    const auto interpolation_kind_str = json.get<std::string>();

    if (interpolation_kind_str == "linear") {
      interpolation_kind = InterpolationKind::Linear;
    } else if (interpolation_kind_str == "ease-out") {
      interpolation_kind = InterpolationKind::EaseOut;
    } else {
      throw std::runtime_error(
          "failed to parse interpolation kind: expected one of ['linear', "
          "'ease-out'], got: " +
          interpolation_kind_str);
    }
  }

  void from_json(const nlohmann::json& json, Range& range) {
    json["min"].get_to(range.min);
    json["max"].get_to(range.max);
  }

  void from_json(const nlohmann::json& json, BiomeSpec& biome) {
    json["temperature"].get_to(biome.temperature_range);
    json["precipitation"].get_to(biome.precipitation_range);
    json["sparse_color"].get_to(biome.sparse_color);
    json["dense_color"].get_to(biome.dense_color);
  }

  template <typename T>
  void try_parse(const nlohmann::json& json, const std::string& key, T& value) {
    if (json.find(key) == json.end()) {
      throw std::runtime_error("missing property: " + key);
    }
    try {
      json[key].get_to(value);
    } catch (const std::exception& exception) {
      throw std::runtime_error(
          "failed to parse property " + key + ": " + exception.what());
    }
  }

  template <typename T>
  void try_parse_optional(
      const nlohmann::json& json,
      const std::string& key,
      T& value) {

    if (json.find(key) != json.end()) {
      try_parse(json, key, value);
    }
  }

  struct Config {
    int elevation_rng_seed;
    size_t elevation_width;
    size_t elevation_height;
    size_t elevation_nr_layers;
    float elevation_scale;
    float elevation_lacunarity;
    float elevation_decay;
    float elevation_bias;
    float elevation_stretch;
    std::string elevation_texture_path;
    std::string elevation_colormap_path;
    float elevation_colormap_min;
    float elevation_colormap_max;

    Rgb ocean_mask_color;
    uint32_t ocean_distance_limit;
    std::string ocean_distance_colormap_path;
    float ocean_distance_colormap_max;

    uint64_t temperature_rng_seed;
    float temperature_latitude_difference;
    float temperature_elevation_factor;
    float temperature_ocean_difference;
    float temperature_ocean_difference_falloff;
    float temperature_noise;
    float temperature_bias;
    std::string temperature_colormap_path;
    float temperature_colormap_min;
    float temperature_colormap_max;

    float ice_temperature_threshold_land;
    float ice_temperature_threshold_ocean;
    Rgb ice_mask_color;

    float precipitation_ocean_distance_factor;
    std::array<float, 4> precipitation_temperature_coefficients;
    float precipitation_multiplier;
    std::string precipitation_colormap_path;
    float precipitation_colormap_max;

    size_t lake_iteration_limit;
    float lake_depth_threshold;
    std::string lake_colormap_path;
    float lake_colormap_depth_max;
    float lake_colormap_drain_max;
    float lake_colormap_level_max;

    size_t river_iteration_limit;
    float river_precipitation_factor;
    size_t river_erosion_iterations;
    float river_erosion_factor;
    float river_intensity_threshold;
    float river_depth_factor;
    std::string river_colormap_path;
    float river_colormap_depth_max;
    float river_colormap_intensity_max;

    uint32_t river_distance_limit;
    std::string river_distance_colormap_path;
    float river_distance_colormap_max;

    std::vector<BiomeSpec> biomes;

    float image_deep_water_threshold;
    Rgb image_deep_water_color;
    Rgb image_shallow_water_color;
    InterpolationKind image_color_interpolation_kind;

    std::string heightmap_colormap_path;
    float heightmap_colormap_max;

    std::string output_directory_path = ".";

    Range batch_mode_seed_range;
    std::vector<int> batch_mode_seeds;

    static Config from_json_file(const std::string& path) {
      nlohmann::json json_config;

      std::ifstream file_stream(path);
      if (!file_stream.is_open()) {
        throw std::runtime_error("failed to read configuration: " + path);
      }
      file_stream >> json_config;

      Config config;

#define PARSE(NAME) try_parse(json_config, #NAME, config.NAME);
#define PARSE_OPTIONAL(NAME) \
  try_parse_optional(json_config, #NAME, config.NAME);

      PARSE(elevation_rng_seed);
      PARSE(elevation_width);
      PARSE(elevation_height);
      PARSE(elevation_nr_layers);
      PARSE(elevation_scale);
      PARSE(elevation_lacunarity);
      PARSE(elevation_decay);
      PARSE(elevation_bias);
      PARSE(elevation_stretch);
      PARSE(elevation_texture_path);
      PARSE(elevation_colormap_path);
      PARSE(elevation_colormap_min);
      PARSE(elevation_colormap_max);

      PARSE(ocean_mask_color);
      PARSE(ocean_distance_limit);
      PARSE(ocean_distance_colormap_path);
      PARSE(ocean_distance_colormap_max);

      PARSE(temperature_rng_seed);
      PARSE(temperature_latitude_difference);
      PARSE(temperature_elevation_factor);
      PARSE(temperature_ocean_difference);
      PARSE(temperature_ocean_difference_falloff);
      PARSE(temperature_noise);
      PARSE(temperature_bias);
      PARSE(temperature_colormap_path);
      PARSE(temperature_colormap_min);
      PARSE(temperature_colormap_max);

      PARSE(ice_temperature_threshold_land);
      PARSE(ice_temperature_threshold_ocean);
      PARSE(ice_mask_color);

      PARSE(precipitation_ocean_distance_factor);
      PARSE(precipitation_temperature_coefficients);
      PARSE(precipitation_multiplier);
      PARSE(precipitation_colormap_path);
      PARSE(precipitation_colormap_max);

      PARSE(lake_iteration_limit);
      PARSE(lake_depth_threshold);
      PARSE(lake_colormap_path);
      PARSE(lake_colormap_depth_max);
      PARSE(lake_colormap_drain_max);
      PARSE(lake_colormap_level_max);

      PARSE(river_iteration_limit);
      PARSE(river_precipitation_factor);
      PARSE(river_erosion_iterations);
      PARSE(river_erosion_factor);
      PARSE(river_intensity_threshold);
      PARSE(river_depth_factor);
      PARSE(river_colormap_path);
      PARSE(river_colormap_depth_max);
      PARSE(river_colormap_intensity_max);

      PARSE(river_distance_limit);
      PARSE(river_distance_colormap_path);
      PARSE(river_distance_colormap_max);

      PARSE(biomes);

      PARSE(image_deep_water_threshold);
      PARSE(image_deep_water_color);
      PARSE(image_shallow_water_color);
      PARSE(image_color_interpolation_kind);

      PARSE(heightmap_colormap_path);
      PARSE(heightmap_colormap_max);

      PARSE_OPTIONAL(output_directory_path);

      PARSE_OPTIONAL(batch_mode_seed_range);
      PARSE_OPTIONAL(batch_mode_seeds);

#undef PARSE
#undef PARSE_OPTIONAL

      return config;
    }
  };  // namespace worldgen

}  // namespace worldgen
