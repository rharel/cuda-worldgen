/**
 * Copyright (c) 2020 Raoul Harel
 * All rights reserved
 */

#define __STDC_WANT_LIB_EXT1__ 1  // localtime_s

#include <atomic>
#include <chrono>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "config.h"
#include "cudu.h"
#include "filewatch.h"
#include "worldgen.cuh"

constexpr char WINDOW_TITLE[] = "Worldgen";
constexpr int WINDOW_REFRESH_RATE_MS = 100;

cv::Mat displayed_image;

void update_status(const std::string& text) {
  std::cout << text << std::endl;

  if (displayed_image.empty()) {
    return;
  }

  const auto text_padding = 5;
  const auto text_height = 20;
  const auto thickness = 1;
  cv::Mat annotated_image = displayed_image.clone();
  cv::rectangle(
      annotated_image,
      cv::Rect(0, 0, annotated_image.cols, text_height + 2 * text_padding),
      CV_RGB(40, 40, 40),
      cv::FILLED);
  cv::putText(
      annotated_image,
      text,
      cv::Point(0, text_height + text_padding),
      cv::FONT_HERSHEY_DUPLEX,
      cv::getFontScaleFromHeight(
          cv::FONT_HERSHEY_DUPLEX, text_height, thickness),
      CV_RGB(255, 255, 255),
      thickness);
  cv::imshow(WINDOW_TITLE, annotated_image);
  cv::waitKey(WINDOW_REFRESH_RATE_MS);
}

std::unordered_set<worldgen::DatasetKind>
dataset_kinds_affected_by_config_change(
    const worldgen::Config& a,
    const worldgen::Config& b) {

  using worldgen::DatasetKind;

  std::unordered_set<DatasetKind> directly_affected;

#define AFFECTS(NAME, DATASET_KIND)         \
  if (a.NAME != b.NAME) {                   \
    directly_affected.insert(DATASET_KIND); \
  }

  AFFECTS(elevation_rng_seed, DatasetKind::Elevation);
  AFFECTS(elevation_width, DatasetKind::Elevation);
  AFFECTS(elevation_height, DatasetKind::Elevation);
  AFFECTS(elevation_nr_layers, DatasetKind::Elevation);
  AFFECTS(elevation_scale, DatasetKind::Elevation);
  AFFECTS(elevation_lacunarity, DatasetKind::Elevation);
  AFFECTS(elevation_decay, DatasetKind::Elevation);
  AFFECTS(elevation_bias, DatasetKind::Elevation);
  AFFECTS(elevation_stretch, DatasetKind::Elevation);
  AFFECTS(elevation_texture_path, DatasetKind::Elevation);

  AFFECTS(ocean_distance_limit, DatasetKind::OceanDistance);

  AFFECTS(temperature_rng_seed, DatasetKind::Temperature);
  AFFECTS(temperature_latitude_difference, DatasetKind::Temperature);
  AFFECTS(temperature_elevation_factor, DatasetKind::Temperature);
  AFFECTS(temperature_ocean_difference, DatasetKind::Temperature);
  AFFECTS(temperature_ocean_difference_falloff, DatasetKind::Temperature);
  AFFECTS(temperature_noise, DatasetKind::Temperature);
  AFFECTS(temperature_bias, DatasetKind::Temperature);

  AFFECTS(ice_temperature_threshold_land, DatasetKind::Ice);
  AFFECTS(ice_temperature_threshold_ocean, DatasetKind::Ice);

  AFFECTS(precipitation_ocean_distance_factor, DatasetKind::Precipitation);
  AFFECTS(precipitation_temperature_coefficients, DatasetKind::Precipitation);
  AFFECTS(precipitation_multiplier, DatasetKind::Precipitation);

  AFFECTS(lake_iteration_limit, DatasetKind::Lakes);
  AFFECTS(lake_depth_threshold, DatasetKind::Lakes);

  AFFECTS(river_iteration_limit, DatasetKind::Rivers);
  AFFECTS(river_precipitation_factor, DatasetKind::Rivers);
  AFFECTS(river_erosion_iterations, DatasetKind::Rivers);
  AFFECTS(river_erosion_factor, DatasetKind::Rivers);
  AFFECTS(river_intensity_threshold, DatasetKind::Rivers);
  AFFECTS(river_depth_factor, DatasetKind::Rivers);

  AFFECTS(biomes, DatasetKind::Biomes);

#undef AFFECTS

  std::unordered_set<worldgen::DatasetKind> all_affected(directly_affected);
  bool collecting_dependents = true;
  while (collecting_dependents) {
    const size_t nr_affected_before = all_affected.size();

    auto newly_affected(all_affected);
    for (const auto dataset_kind : all_affected) {
      newly_affected.merge(worldgen::dependents_of_dataset(dataset_kind));
    }
    all_affected = newly_affected;

    const size_t nr_affected_after = all_affected.size();
    collecting_dependents = nr_affected_after > nr_affected_before;
  }
  return all_affected;
}

struct Cache {
  std::unordered_set<worldgen::DatasetKind> cached_dataset_kinds;

  cudu::device::Array2D<float> elevation;
  cudu::host::Array3D<unsigned char> elevation_image;

  cudu::device::Array2D<bool> ocean;
  cudu::host::Array3D<unsigned char> ocean_image;

  cudu::device::Array2D<float> ocean_distance;
  cudu::host::Array3D<unsigned char> ocean_distance_image;

  cudu::device::Array2D<float> temperature;
  cudu::host::Array3D<unsigned char> temperature_image;

  cudu::device::Array2D<bool> ice;
  cudu::host::Array3D<unsigned char> ice_image;

  cudu::device::Array2D<float> precipitation;
  cudu::host::Array3D<unsigned char> precipitation_image;

  worldgen::LakeDatasets lakes;
  cudu::host::Array3D<unsigned char> lake_depth_image;
  cudu::host::Array3D<unsigned char> lake_level_image;
  cudu::host::Array3D<unsigned char> lake_drain_distance_image;

  worldgen::RiverDatasets rivers;
  cudu::host::Array3D<unsigned char> river_intensity_image;
  cudu::host::Array3D<unsigned char> river_depth_image;

  cudu::device::Array2D<float> river_distance;
  cudu::host::Array3D<unsigned char> river_distance_image;

  cudu::device::Array2D<worldgen::BiomeMixture> biomes;
  cudu::host::Array3D<unsigned char> biomes_image;

  cudu::host::Array3D<unsigned char> combined_image;

  cudu::device::Array2D<float> heightmap;
  cudu::host::Array3D<unsigned char> heightmap_image;
};

cudu::host::Array3D<unsigned char>& cached_image(
    const worldgen::ImageKind image_kind,
    Cache& cache) {

  switch (image_kind) {
    case worldgen::ImageKind::Elevation:
      return cache.elevation_image;
    case worldgen::ImageKind::Ocean:
      return cache.ocean_image;
    case worldgen::ImageKind::OceanDistance:
      return cache.ocean_distance_image;
    case worldgen::ImageKind::Temperature:
      return cache.temperature_image;
    case worldgen::ImageKind::Ice:
      return cache.ice_image;
    case worldgen::ImageKind::Precipitation:
      return cache.precipitation_image;
    case worldgen::ImageKind::LakeDepth:
      return cache.lake_depth_image;
    case worldgen::ImageKind::LakeDrainDistance:
      return cache.lake_drain_distance_image;
    case worldgen::ImageKind::LakeLevel:
      return cache.lake_level_image;
    case worldgen::ImageKind::RiverIntensity:
      return cache.river_intensity_image;
    case worldgen::ImageKind::RiverDepth:
      return cache.river_depth_image;
    case worldgen::ImageKind::RiverDistance:
      return cache.river_distance_image;
    case worldgen::ImageKind::Biomes:
      return cache.biomes_image;
    case worldgen::ImageKind::Combined:
      return cache.combined_image;
    case worldgen::ImageKind::Heightmap:
      return cache.heightmap_image;
    default:
      throw std::runtime_error("unreachable");
  }
}

void save_image(
    const cudu::host::Array3D<unsigned char>& image,
    const std::string& image_kind_name,
    const int elevation_rng_seed,
    const std::string& directory_path) {

  const time_t time = std::time(nullptr);
  tm local_time;
  localtime_s(&local_time, &time);

  std::ostringstream oss;
  oss << std::put_time(&local_time, "%Y_%m_%d_%H_%M");
  const std::string time_str = oss.str();

  const std::string file_name = time_str + "_seed_" +
                                std::to_string(elevation_rng_seed) + "_" +
                                image_kind_name + ".png";

  const cv::Mat image_cv(
      int(image.shape()[0]),
      int(image.shape()[1]),
      CV_8UC3,
      const_cast<unsigned char*>(image.begin()));
  cv::imwrite(directory_path + "/" + file_name, image_cv);

  std::cout << "Saved " + file_name << std::endl;
}

void display_image(const cudu::host::Array3D<unsigned char>& image) {
  const cv::Mat image_cv(
      int(image.shape()[0]),
      int(image.shape()[1]),
      CV_8UC3,
      const_cast<unsigned char*>(image.begin()));
  displayed_image = image_cv;
  cv::imshow(WINDOW_TITLE, image_cv);
  cv::waitKey(WINDOW_REFRESH_RATE_MS);
}

bool cache_dataset(
    const worldgen::DatasetKind dataset_kind,
    const worldgen::Config& config,
    Cache& cache) {

  const auto dataset_cached = cache.cached_dataset_kinds.find(dataset_kind) !=
                              cache.cached_dataset_kinds.end();

  bool dependencies_stale = false;
  for (const auto dependency_dataset_kind :
       dependencies_of_dataset(dataset_kind)) {
    dependencies_stale |= cache_dataset(dependency_dataset_kind, config, cache);
  }

  if (dataset_cached && !dependencies_stale) {
    return false;
  }

  switch (dataset_kind) {
    case worldgen::DatasetKind::Elevation: {
      update_status("Computing elevation...");

      if (!config.elevation_texture_path.empty()) {
        const cv::Mat image = cv::imread(
            config.elevation_texture_path, cv::ImreadModes::IMREAD_GRAYSCALE);

        if (image.empty()) {
          update_status("Elevation texture load error");
        }

        cudu::host::Array2D<float> elevation_data(
            cudu::Shape2D(size_t(image.cols), size_t(image.rows)));

        for (size_t i = 0; i < elevation_data.size(); ++i) {
          elevation_data[i] =
              config.elevation_bias +
              (image.data[i] / 255.f) * config.elevation_stretch;
        }
        cache.elevation.upload(elevation_data);

      } else {
        cache.elevation = worldgen::elevation(
            config.elevation_width,
            config.elevation_height,
            config.elevation_nr_layers,
            config.elevation_scale,
            config.elevation_lacunarity,
            config.elevation_decay,
            config.elevation_bias,
            config.elevation_stretch,
            config.elevation_rng_seed);
      }
      break;
    }
    case worldgen::DatasetKind::Ocean:
      update_status("Computing ocean...");
      cache.ocean = worldgen::ocean(cache.elevation);
      break;
    case worldgen::DatasetKind::OceanDistance:
      update_status("Computing ocean distance...");
      cache.ocean_distance =
          worldgen::distance(cache.ocean, config.ocean_distance_limit);
      break;
    case worldgen::DatasetKind::Temperature:
      update_status("Computing temperature...");
      cache.temperature = worldgen::temperature(
          cache.elevation,
          cache.ocean_distance,
          config.temperature_latitude_difference,
          config.temperature_elevation_factor,
          config.temperature_ocean_difference,
          config.temperature_ocean_difference_falloff,
          config.temperature_noise,
          config.temperature_bias,
          config.temperature_rng_seed);
      break;
    case worldgen::DatasetKind::Ice:
      update_status("Computing ice...");
      cache.ice = worldgen::ice(
          cache.ocean,
          cache.temperature,
          config.ice_temperature_threshold_land,
          config.ice_temperature_threshold_ocean);
      break;
    case worldgen::DatasetKind::Precipitation:
      update_status("Computing precipitation...");
      cache.precipitation = worldgen::precipitation(
          cache.ocean_distance,
          cache.temperature,
          config.precipitation_ocean_distance_factor,
          config.precipitation_multiplier,
          config.precipitation_temperature_coefficients);
      break;
    case worldgen::DatasetKind::Lakes:
      update_status("Computing lakes...");
      cache.lakes = worldgen::lakes(
          cache.elevation,
          cache.ocean,
          cache.ice,
          config.lake_iteration_limit,
          config.lake_depth_threshold);
      break;
    case worldgen::DatasetKind::Rivers:
      update_status("Computing rivers...");
      cache.rivers = worldgen::rivers(
          cache.lakes.adjusted_elevation,
          cache.ocean,
          cache.ice,
          cache.precipitation,
          cache.lakes.depth,
          cache.lakes.level,
          cache.lakes.drain_distance,
          config.river_iteration_limit,
          config.river_precipitation_factor,
          config.river_erosion_iterations,
          config.river_erosion_factor,
          config.river_intensity_threshold,
          config.river_depth_factor);
      break;
    case worldgen::DatasetKind::RiverDistance:
      update_status("Computing river distance...");
      cache.river_distance = worldgen::distance(
          cache.rivers.depth.as<bool>(), config.river_distance_limit);
      break;
    case worldgen::DatasetKind::Biomes:
      update_status("Computing biomes...");
      cache.biomes = worldgen::biomes(
          cache.ocean,
          cache.temperature,
          cache.ice,
          cache.precipitation,
          cache.lakes.depth,
          cache.rivers.depth,
          cache.river_distance,
          cudu::device::Array1D<worldgen::BiomeSpec>::from_ptr(
              config.biomes.data(), config.biomes.size()));
      break;
    case worldgen::DatasetKind::Heightmap:
      update_status("Computing heightmap...");
      cache.heightmap = worldgen::heightmap(
          cache.rivers.adjusted_elevation, cache.ocean, cache.lakes.depth);
      break;
    default:
      throw std::runtime_error("unreachable");
  }

  cache.cached_dataset_kinds.insert(dataset_kind);

  return true;
}

void cache_image(
    const worldgen::ImageKind image_kind,
    const worldgen::Config& config,
    Cache& cache) {

  for (const auto dependency_dataset_kind :
       worldgen::dependencies_of_image(image_kind)) {
    cache_dataset(dependency_dataset_kind, config, cache);
  }

  switch (image_kind) {
    case worldgen::ImageKind::Elevation:
      update_status("Rendering elevation...");
      cache.elevation_image =
          worldgen::colormap_image(
              cache.elevation,
              worldgen::colormap(config.elevation_colormap_path),
              config.elevation_colormap_min,
              config.elevation_colormap_max,
              0)
              .download();
      break;
    case worldgen::ImageKind::Ocean:
      update_status("Rendering ocean...");
      cache.ocean_image =
          worldgen::binary_image(cache.ocean, config.ocean_mask_color)
              .download();
      break;
    case worldgen::ImageKind::OceanDistance:
      update_status("Rendering ocean distance...");
      cache.ocean_distance_image =
          worldgen::colormap_image(
              cache.ocean_distance,
              worldgen::colormap(config.ocean_distance_colormap_path),
              0,
              config.ocean_distance_colormap_max)
              .download();
      break;
    case worldgen::ImageKind::Temperature:
      update_status("Rendering temperature...");
      cache.temperature_image =
          worldgen::colormap_image(
              cache.temperature,
              worldgen::colormap(config.temperature_colormap_path),
              config.temperature_colormap_min,
              config.temperature_colormap_max,
              0)
              .download();
      break;
    case worldgen::ImageKind::Ice:
      update_status("Rendering ice...");
      cache.ice_image =
          worldgen::binary_image(cache.ice, config.ice_mask_color).download();
      break;
    case worldgen::ImageKind::Precipitation:
      update_status("Rendering precipitation...");
      cache.precipitation_image =
          worldgen::colormap_image(
              cache.precipitation,
              worldgen::colormap(config.precipitation_colormap_path),
              0,
              config.precipitation_colormap_max)
              .download();
      break;
    case worldgen::ImageKind::LakeDepth:
      update_status("Rendering lake depth...");
      cache.lake_depth_image =
          worldgen::colormap_image(
              cache.lakes.depth,
              worldgen::colormap(config.lake_colormap_path),
              0,
              config.lake_colormap_depth_max)
              .download();
      break;
    case worldgen::ImageKind::LakeDrainDistance:
      update_status("Rendering lake drain distance...");
      cache.lake_drain_distance_image =
          worldgen::colormap_image(
              cache.lakes.drain_distance,
              worldgen::colormap(config.lake_colormap_path),
              0,
              config.lake_colormap_drain_max)
              .download();
      break;
    case worldgen::ImageKind::LakeLevel:
      update_status("Rendering lake level...");
      cache.lake_level_image =
          worldgen::colormap_image(
              cache.lakes.level,
              worldgen::colormap(config.lake_colormap_path),
              0,
              config.lake_colormap_level_max)
              .download();
      break;
    case worldgen::ImageKind::RiverIntensity:
      update_status("Rendering river intensity...");
      cache.river_intensity_image =
          worldgen::colormap_image(
              cache.rivers.intensity,
              worldgen::colormap(config.river_colormap_path),
              0,
              config.river_colormap_intensity_max)
              .download();
      break;
    case worldgen::ImageKind::RiverDepth:
      update_status("Rendering river depth...");
      cache.river_depth_image =
          worldgen::colormap_image(
              cache.rivers.depth,
              worldgen::colormap(config.river_colormap_path),
              0,
              config.river_colormap_depth_max)
              .download();
      break;
    case worldgen::ImageKind::RiverDistance:
      update_status("Rendering river distance...");
      cache.river_distance_image =
          worldgen::colormap_image(
              cache.river_distance,
              worldgen::colormap(config.river_distance_colormap_path),
              0,
              config.river_distance_colormap_max)
              .download();
      break;
    case worldgen::ImageKind::Biomes: {
      update_status("Rendering biomes...");
      cache.biomes_image =
          worldgen::biome_image(
              cache.ocean,
              cache.lakes.depth,
              cache.rivers.depth,
              cache.biomes,
              cudu::device::Array1D<worldgen::BiomeSpec>::from_ptr(
                  config.biomes.data(), config.biomes.size()),
              config.image_deep_water_color)
              .download();
      break;
    }
    case worldgen::ImageKind::Combined:
      update_status("Rendering combined image...");
      cache.combined_image =
          worldgen::combined_image(
              cache.rivers.adjusted_elevation,
              cache.ocean,
              cache.lakes.depth,
              cache.rivers.depth,
              cudu::device::Array1D<worldgen::BiomeSpec>::from_ptr(
                  config.biomes.data(), config.biomes.size()),
              cache.biomes,
              config.image_deep_water_threshold,
              config.image_deep_water_color,
              config.image_shallow_water_color,
              config.image_color_interpolation_kind)
              .download();
      break;
    case worldgen::ImageKind::Heightmap:
      update_status("Rendering heightmap...");
      cache.heightmap_image =
          worldgen::colormap_image(
              cache.heightmap,
              worldgen::colormap(config.heightmap_colormap_path),
              0,
              config.heightmap_colormap_max)
              .download();
      break;
    default:
      throw std::runtime_error("unreachable");
  }
}

void update_display(
    const worldgen::ImageKind image_kind,
    const worldgen::Config& config,
    Cache& cache) {

  cache_image(image_kind, config, cache);
  display_image(cached_image(image_kind, cache));
}

void update_cache(
    const worldgen::Config& old_config,
    const worldgen::Config& new_config,
    Cache& cache) {

  for (const auto dataset_kind :
       dataset_kinds_affected_by_config_change(old_config, new_config)) {
    cache.cached_dataset_kinds.erase(dataset_kind);
  }
}

std::string image_kind_name(const worldgen::ImageKind image_kind) {
  switch (image_kind) {
    case worldgen::ImageKind::Elevation:
      return "elevation";
    case worldgen::ImageKind::Ocean:
      return "ocean";
    case worldgen::ImageKind::OceanDistance:
      return "ocean_distance";
    case worldgen::ImageKind::Temperature:
      return "temperature";
    case worldgen::ImageKind::Ice:
      return "ice";
    case worldgen::ImageKind::Precipitation:
      return "precipitation";
    case worldgen::ImageKind::LakeDepth:
      return "lake_depth";
    case worldgen::ImageKind::LakeLevel:
      return "lake_level";
    case worldgen::ImageKind::LakeDrainDistance:
      return "lake_drain_distance";
    case worldgen::ImageKind::RiverIntensity:
      return "river_intensity";
    case worldgen::ImageKind::RiverDepth:
      return "river_depth";
    case worldgen::ImageKind::RiverDistance:
      return "river_distance";
    case worldgen::ImageKind::Biomes:
      return "biomes";
    case worldgen::ImageKind::Combined:
      return "combined_image";
    case worldgen::ImageKind::Heightmap:
      return "heightmap";
    default:
      throw std::runtime_error("unreachable");
  }
}

void render_and_save(const worldgen::Config& config, Cache& cache) {
  cache.cached_dataset_kinds.clear();
  cache_image(worldgen::ImageKind::Combined, config, cache);
  cache_image(worldgen::ImageKind::Heightmap, config, cache);
  save_image(
      cache.combined_image,
      image_kind_name(worldgen::ImageKind::Combined),
      config.elevation_rng_seed,
      config.output_directory_path);
  save_image(
      cache.heightmap_image,
      image_kind_name(worldgen::ImageKind::Heightmap),
      config.elevation_rng_seed,
      config.output_directory_path);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    const auto executable_name =
        std::filesystem::path(argv[0]).filename().string();
    std::cerr << "Usage: " << executable_name << " CONFIG_FILE" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string config_path(argv[1]);
  worldgen::Config config;
  try {
    config = worldgen::Config::from_json_file(config_path);
  } catch (const std::exception& exception) {
    std::cerr << "Failed to load config: " << exception.what() << std::endl;
    return EXIT_FAILURE;
  }

  Cache cache;

  if (!config.batch_mode_seed_range.empty()) {
    for (auto seed = int(config.batch_mode_seed_range.min);
         seed < config.batch_mode_seed_range.max;
         ++seed) {
      config.elevation_rng_seed = seed;
      std::cout << "Computing image with seed: " << seed << std::endl;
      render_and_save(config, cache);
    }
    return EXIT_SUCCESS;
  } else if (config.batch_mode_seeds.size() > 0) {
    for (const int seed : config.batch_mode_seeds) {
      std::cout << "Computing image with seed: " << seed << std::endl;
      config.elevation_rng_seed = seed;
      render_and_save(config, cache);
    }
    return EXIT_SUCCESS;
  }

  struct ImageKeyBinding {
    char keycode;
    const char* image_name;
    worldgen::ImageKind image_kind;
  };
  constexpr ImageKeyBinding IMAGE_KEY_BINDINGS[] = {
      {'1', "elevation", worldgen::ImageKind::Elevation},
      {'2', "ocean", worldgen::ImageKind::Ocean},
      {'3', "ocean distance", worldgen::ImageKind::OceanDistance},
      {'4', "temperature", worldgen::ImageKind::Temperature},
      {'5', "ice", worldgen::ImageKind::Ice},
      {'6', "precipitation", worldgen::ImageKind::Precipitation},
      {'7', "lake depth", worldgen::ImageKind::LakeDepth},
      {'8', "lake level", worldgen::ImageKind::LakeLevel},
      {'9', "lake drain distance", worldgen::ImageKind::LakeDrainDistance},
      {'0', "river intensity", worldgen::ImageKind::RiverIntensity},
      {'q', "river depth", worldgen::ImageKind::RiverDepth},
      {'w', "river distance", worldgen::ImageKind::RiverDistance},
      {'e', "biomes", worldgen::ImageKind::Biomes},
      {'r', "combined image", worldgen::ImageKind::Combined},
      {'t', "height map texture", worldgen::ImageKind::Heightmap},
  };

  constexpr char KEYCODE_ELEVATION_SEED_INCREMENT = '=';
  constexpr char KEYCODE_ELEVATION_SEED_DECREMENT = '-';
  constexpr char KEYCODE_EXPORT_IMAGE = 'p';
  constexpr char KEYCODE_ESCAPE = 27;

  for (const auto& binding : IMAGE_KEY_BINDINGS) {
    std::cout << "Press '" << binding.keycode << "' to view "
              << binding.image_name << std::endl;
  }
  std::cout << "Press '" << KEYCODE_ELEVATION_SEED_INCREMENT
            << "' to increment elevation seed" << std::endl;
  std::cout << "Press '" << KEYCODE_ELEVATION_SEED_DECREMENT
            << "' to decrement elevation seed" << std::endl;
  std::cout << "Press '" << KEYCODE_EXPORT_IMAGE << "' to export image"
            << std::endl;

  auto display_image_kind = worldgen::ImageKind::Combined;

  std::atomic_bool stop_display_update_thread = false;
  std::atomic_bool display_update_needed = true;
  std::thread display_update_thread([&config,
                                     &cache,
                                     &display_image_kind,
                                     &stop_display_update_thread,
                                     &display_update_needed]() {
    while (!stop_display_update_thread) {
      if (display_update_needed) {
        update_display(display_image_kind, config, cache);
        display_update_needed = false;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(WINDOW_REFRESH_RATE_MS));
    }
  });

  filewatch::FileWatcher config_watcher(
      config_path, std::chrono::milliseconds(1000));
  const auto on_config_file_change =
      [&config, &cache, &display_update_needed](
          const std::string& path, const filewatch::EventKind event_kind) {
        if (event_kind != filewatch::EventKind::Created &&
            event_kind != filewatch::EventKind::Modified) {
          return;
        }
        update_status("Reloading config...");
        worldgen::Config new_config;
        try {
          new_config = worldgen::Config::from_json_file(path);
        } catch (const std::exception& exception) {
          update_status("Config error");
          std::cerr << "Failed to load config: " << exception.what()
                    << std::endl;
          return;
        }
        update_cache(config, new_config, cache);
        config = new_config;
        display_update_needed = true;
      };

  std::thread config_watcher_thread(
      [&config_watcher, &on_config_file_change]() {
        config_watcher.start(on_config_file_change);
      });

  cv::namedWindow(WINDOW_TITLE, cv::WINDOW_AUTOSIZE);
  cv::setWindowProperty(WINDOW_TITLE, cv::WND_PROP_TOPMOST, true);

  char keycode = 0;

  while (cv::getWindowProperty(WINDOW_TITLE, 0) >= 0 &&
         keycode != KEYCODE_ESCAPE) {

    keycode = cv::waitKey(WINDOW_REFRESH_RATE_MS);
    switch (keycode) {
      case KEYCODE_EXPORT_IMAGE:
        save_image(
            cached_image(display_image_kind, cache),
            image_kind_name(display_image_kind),
            config.elevation_rng_seed,
            config.output_directory_path);
        break;
      case KEYCODE_ELEVATION_SEED_INCREMENT: {
        worldgen::Config new_config = config;
        new_config.elevation_rng_seed = config.elevation_rng_seed + 1;
        update_cache(config, new_config, cache);
        config = new_config;
        display_update_needed = true;
        std::cout << "Elevation seed: " << config.elevation_rng_seed
                  << std::endl;
        break;
      }
      case KEYCODE_ELEVATION_SEED_DECREMENT: {
        if (config.elevation_rng_seed > 0) {
          worldgen::Config new_config = config;
          new_config.elevation_rng_seed = config.elevation_rng_seed - 1;
          update_cache(config, new_config, cache);
          config = new_config;
          display_update_needed = true;
        }
        std::cout << "Elevation seed: " << config.elevation_rng_seed
                  << std::endl;
        break;
      }
      default:
        for (const auto& binding : IMAGE_KEY_BINDINGS) {
          if (keycode == binding.keycode) {
            display_image_kind = binding.image_kind;
            display_update_needed = true;
          }
        }
    }
  }

  stop_display_update_thread = true;
  display_update_thread.join();

  config_watcher.stop();
  config_watcher_thread.join();

  return EXIT_SUCCESS;
}
