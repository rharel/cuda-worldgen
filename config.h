#pragma once

#include <fstream>
#include <stdint.h>
#include <string>
#include <vector>

#include "external/include/json.hpp"

#include "model.h"

namespace worldgen 
{
    void from_json(const nlohmann::json& json, Rgb& color)
    {
        if (json.is_object())
        {
            color.r = json["r"].get<unsigned char>();
            color.g = json["g"].get<unsigned char>();
            color.b = json["b"].get<unsigned char>();
        }
        else
        {
            const auto str = json.get<std::string>();
            const unsigned long hex = std::stoul(str, nullptr, 16);
            color.r = (hex & 0xff0000) >> 16;
            color.g = (hex & 0x00ff00) >> 8;
            color.b = (hex & 0x0000ff) >> 0;
        }
    }

    void from_json(const nlohmann::json& json, SurfaceMaterial& material)
    {
        material.likelihood = json["likelihood"].get<float>();
        material.visual_altitude_low = json["visual_altitude_low"].get<Rgb>();
        material.visual_altitude_high = json["visual_altitude_high"].get<Rgb>();
    }

    void from_json(const nlohmann::json& json, Range& range)
    {
        range.min = json["min"].get<float>();
        range.max = json["max"].get<float>();
    }

    void from_json(const nlohmann::json& json, ClimateZone& zone)
    {
        zone.latitude_range = json["latitude"].get<Range>();
        zone.temperature_range = json["temperature"].get<Range>();
        zone.precipitation_range = json["precipitation"].get<Range>();
        zone.visual_dense = json["visual_dense"].get<Rgb>();
        zone.visual_sparse= json["visual_sparse"].get<Rgb>();
    }

    void from_json(const nlohmann::json& json, VicinityLookupParams& lookup)
    {
        lookup.radius = json["radius"].get<float>();
        lookup.radius_step = json["radius_step"].get<float>();
        lookup.nr_samples = json["nr_samples"].get<unsigned>();
        lookup.rng_seed = json["rng_seed"].get<uint64_t>();
    }

    struct Config 
    {
        uint32_t surface_altitude_level_of_detail;
        float surface_altitude_noise_initial;
        float surface_altitude_noise_scale_factor;
        float surface_altitude_bias;
        uint64_t surface_altitude_rng_seed;
        
        std::vector<SurfaceMaterial> surface_materials;
        unsigned surface_material_cell_count;
        float surface_material_cell_blur;
        uint64_t surface_material_cell_rng_seed;

        float temperature_latitude_factor;
        float temperature_altitude_factor;
        float temperature_noise;
        float temperature_bias;
        uint64_t temperature_rng_seed;

        float ocean_freeze_temperature;
        float ocean_boil_temperature;

        Rgb visual_water_shallow;
        Rgb visual_water_deep;
        float visual_water_depth_limit;

        Rgb visual_ice_thin;
        Rgb visual_ice_thick;
        float visual_ice_temperature_limit;

        std::vector<ClimateZone> climate_zones;
        float climate_zone_weight_min;

        VicinityLookupParams precipitation_ocean_lookup;
        float precipitation_ocean_distance_factor;

        unsigned river_block_size;
        VicinityLookupParams river_ocean_lookup;

        unsigned compute_max_nr_threads;

        static Config from_json_file(const std::string& path) 
        {
            nlohmann::json json_config;

            std::ifstream file_stream(path);
            if (!file_stream.is_open()) {
                throw std::runtime_error("failed to open configuration file");
            }
            file_stream >> json_config;

            Config config;
            
            config.surface_altitude_level_of_detail = json_config["surface_altitude_level_of_detail"].get<uint32_t>();
            config.surface_altitude_noise_initial = json_config["surface_altitude_noise_initial"].get<float>();
            config.surface_altitude_noise_scale_factor = json_config["surface_altitude_noise_scale_factor"].get<float>();
            config.surface_altitude_bias = json_config["surface_altitude_bias"].get<float>();
            config.surface_altitude_rng_seed = json_config["surface_altitude_rng_seed"].get<uint64_t>();
            
            config.surface_materials = json_config["surface_materials"].get<std::vector<SurfaceMaterial>>();
            config.surface_material_cell_count = json_config["surface_material_cell_count"].get<unsigned>();
            config.surface_material_cell_blur = json_config["surface_material_cell_blur"].get<float>();
            config.surface_material_cell_rng_seed = json_config["surface_material_cell_rng_seed"].get<uint64_t>();
            
            config.temperature_latitude_factor = json_config["temperature_latitude_factor"].get<float>();
            config.temperature_altitude_factor = json_config["temperature_altitude_factor"].get<float>();
            config.temperature_noise = json_config["temperature_noise"].get<float>();
            config.temperature_bias = json_config["temperature_bias"].get<float>();
            config.temperature_rng_seed = json_config["temperature_rng_seed"].get<uint64_t>();

            config.ocean_freeze_temperature = json_config["ocean_freeze_temperature"].get<float>();
            config.ocean_boil_temperature = json_config["ocean_boil_temperature"].get<float>();

            config.visual_water_shallow = json_config["visual_water_shallow"].get<Rgb>();
            config.visual_water_deep = json_config["visual_water_deep"].get<Rgb>();
            config.visual_water_depth_limit = json_config["visual_water_depth_limit"].get<float>();

            config.visual_ice_thin = json_config["visual_ice_thin"].get<Rgb>();
            config.visual_ice_thick = json_config["visual_ice_thick"].get<Rgb>();
            config.visual_ice_temperature_limit = json_config["visual_ice_temperature_limit"].get<float>();

            config.climate_zones = json_config["climate_zones"].get<std::vector<ClimateZone>>();
            config.climate_zone_weight_min = json_config["climate_zone_weight_min"].get<float>();

            config.precipitation_ocean_distance_factor = json_config["precipitation_ocean_distance_factor"].get<float>();
            config.precipitation_ocean_lookup = json_config["precipitation_ocean_lookup"].get<VicinityLookupParams>();

            config.river_block_size = json_config["river_block_size"].get<unsigned>();
            config.river_ocean_lookup = json_config["river_ocean_lookup"].get<VicinityLookupParams>();

            config.compute_max_nr_threads = json_config["compute_max_nr_threads"].get<unsigned>();
            
            return config;
        }
    };
}
