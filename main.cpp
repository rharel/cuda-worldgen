/**
 * Copyright (c) 2020 Raoul Harel
 * All rights reserved
 */

#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#define NOMINMAX
#include <windows.h>
#include <WinUser.h>
#undef NOMINMAX

#include "config.h"
#include "cudu.h"
#include "filewatch.h"
#include "worldgen.h"

const char MAIN_WINDOW_TITLE[] = "World";

worldgen::Config load_config(const std::string& path)
{
    std::cout << "Loading configuration: " << path << std::endl;
    auto config = worldgen::Config::from_json_file(path);
    return config;
}

std::vector<worldgen::Component> config_diff_recomputation_effect(
    const worldgen::Config& a,
    const worldgen::Config& b)
{
    std::vector<worldgen::Component> affected;

    const auto is_affected = [&affected](const worldgen::Component component)
    {
        return std::find(affected.begin(), affected.end(), component) != affected.end();
    };

    if (a.surface_altitude_source_influence != b.surface_altitude_source_influence ||
        a.surface_altitude_level_of_detail != b.surface_altitude_level_of_detail ||
        a.surface_altitude_noise_initial != b.surface_altitude_noise_initial ||
        a.surface_altitude_noise_scale_factor != b.surface_altitude_noise_scale_factor ||
        a.surface_altitude_bias != b.surface_altitude_bias ||
        a.surface_altitude_rng_seed != b.surface_altitude_rng_seed)
    {
        affected.push_back(worldgen::Component::Altitude);
    }

    bool material_technical_changed = false;
    bool material_visual_changed = false;
    
    if (a.surface_materials.size() != b.surface_materials.size())
    {
        material_technical_changed = true;
    }
    else
    {
        for (size_t i = 0; i < a.surface_materials.size(); ++i)
        {
            if (a.surface_materials[i].likelihood != b.surface_materials[i].likelihood)
            {
                material_technical_changed = true;
                break;
            }
            else if (a.surface_materials[i].visual_altitude_low != b.surface_materials[i].visual_altitude_low ||
                     a.surface_materials[i].visual_altitude_high != b.surface_materials[i].visual_altitude_high)
            {
                material_visual_changed = true;
            }
        }
    }

    if (material_technical_changed ||
        a.surface_material_cell_count != b.surface_material_cell_count ||
        a.surface_material_cell_blur != b.surface_material_cell_blur ||
        a.surface_material_cell_rng_seed != b.surface_material_cell_rng_seed)
    {
        affected.push_back(worldgen::Component::Material);
    }

    if (is_affected(worldgen::Component::Altitude) ||
        a.temperature_latitude_factor != b.temperature_latitude_factor ||
        a.temperature_altitude_factor != b.temperature_altitude_factor ||
        a.temperature_noise != b.temperature_noise ||
        a.temperature_bias != b.temperature_bias ||
        a.temperature_rng_seed != b.temperature_rng_seed)
    {
        affected.push_back(worldgen::Component::Temperature);
    }

    if (is_affected(worldgen::Component::Altitude) ||
        is_affected(worldgen::Component::Temperature) ||
        a.ocean_boil_temperature != b.ocean_boil_temperature)
    {
        affected.push_back(worldgen::Component::Ocean);
    }

    if (is_affected(worldgen::Component::Temperature) ||
        is_affected(worldgen::Component::Ocean) ||
        a.ocean_freeze_temperature != b.ocean_freeze_temperature)
    {
        affected.push_back(worldgen::Component::Ice);
    }

    bool climate_zone_technical_changed = false;
    bool climate_zone_visual_changed = false;

    if (a.climate_zones.size() != b.climate_zones.size())
    {
        climate_zone_technical_changed = true;
    }
    else
    {
        for (size_t i = 0; i < a.climate_zones.size(); ++i)
        {
            if (a.climate_zones[i].latitude_range != b.climate_zones[i].latitude_range ||
                a.climate_zones[i].precipitation_range != b.climate_zones[i].precipitation_range ||
                a.climate_zones[i].temperature_range != b.climate_zones[i].temperature_range)
            {
                climate_zone_technical_changed = true;
                break;
            }
            else if (a.climate_zones[i].visual_sparse != b.climate_zones[i].visual_sparse ||
                a.climate_zones[i].visual_dense != b.climate_zones[i].visual_dense)
            {
                climate_zone_visual_changed = true;
            }
        }
    }

    if (is_affected(worldgen::Component::Temperature) ||
        is_affected(worldgen::Component::Ocean) ||
        climate_zone_technical_changed ||
        a.precipitation_ocean_distance_smoothing != b.precipitation_ocean_distance_smoothing ||
        a.precipitation_ocean_distance_factor != b.precipitation_ocean_distance_factor ||
        a.precipitation_ocean_lookup != b.precipitation_ocean_lookup)
    {
        affected.push_back(worldgen::Component::Precipitation);
    }

    if (is_affected(worldgen::Component::Altitude) ||
        is_affected(worldgen::Component::Ocean) || 
        is_affected(worldgen::Component::Ice) ||
        a.river_block_size != b.river_block_size ||
        a.river_ocean_lookup != b.river_ocean_lookup)
    {
        affected.push_back(worldgen::Component::Rivers);
    }

    if (is_affected(worldgen::Component::Altitude) ||
        is_affected(worldgen::Component::Material) ||
        is_affected(worldgen::Component::Temperature) ||
        is_affected(worldgen::Component::Ocean) ||
        is_affected(worldgen::Component::Ice) ||
        is_affected(worldgen::Component::Precipitation) ||
        is_affected(worldgen::Component::Rivers) ||
        material_visual_changed ||
        climate_zone_visual_changed ||
        a.visual_water_shallow != b.visual_water_shallow ||
        a.visual_water_deep != b.visual_water_deep ||
        a.visual_water_depth_limit != b.visual_water_depth_limit ||
        a.visual_ice_thin != b.visual_ice_thin ||
        a.visual_ice_thick != b.visual_ice_thick ||
        a.visual_ice_temperature_limit != b.visual_ice_temperature_limit)
    {
        affected.push_back(worldgen::Component::Image);
    }
    
    return affected;
}

struct WorldData
{
    cudu::device::Array2D<float> altitude;
    cudu::device::Array3D<float> surface_material;
    cudu::device::Array2D<float> temperature;
    cudu::device::Array2D<bool> ocean_mask;
    cudu::device::Array2D<bool> ice_mask;
    cudu::device::Array2D<float> precipitation;
    cudu::device::Array2D<float> river_depth;
    cudu::host::Array3D<unsigned char> image;
};

void generate_world(
    const worldgen::Config& config, 
    const std::vector<worldgen::Component>& components,
    const cudu::device::Array1D<float>& boundary_left,
    const cudu::device::Array1D<float>& boundary_bottom,
    WorldData& world,
    bool gui = true)
{
    for (worldgen::Component component : components)
    {
        if (component == worldgen::Component::Altitude)
        {
            /*cudu::device::Array2D<float> bias_map;
            if (!config.surface_altitude_source_path.empty())
            {
                std::cout << "Loading surface altitude image..." << std::endl;
                const cv::Mat grayscale = cv::imread(
                    config.surface_altitude_source_path,
                    cv::ImreadModes::IMREAD_GRAYSCALE
                );
                cudu::host::Array2D<float> bias({
                    size_t(grayscale.cols),
                    size_t(grayscale.rows)
                });
                for (size_t i = 0; i < bias.size(); ++i)
                {
                    bias[i] = config.surface_altitude_source_influence * (2 * (grayscale.data[i] / 255.f) - 1);
                }
                bias_map.upload(bias);
            }
            else
            {
                const size_t edge_size = std::pow(2, config.surface_altitude_level_of_detail) + 1;
                bias_map.upload(cudu::host::Array2D<float>({ edge_size, edge_size }, 0));
            }*/
            std::cout << "Computing surface altitude..." << std::endl;
            world.altitude = worldgen::altitude(
                config.surface_altitude_level_of_detail,
                config.surface_altitude_noise_initial,
                config.surface_altitude_noise_scale_factor,
                config.surface_altitude_bias,
                config.surface_altitude_rng_seed
            );
        }
        else if (component == worldgen::Component::Material)
        {
            std::cout << "Computing surface material..." << std::endl;
            world.surface_material = worldgen::material(
                world.altitude.shape(),
                config.surface_materials,
                config.surface_material_cell_count,
                config.surface_material_cell_blur,
                config.surface_material_cell_rng_seed
            );
        }
        else if (component == worldgen::Component::Temperature)
        {
            std::cout << "Computing temperature..." << std::endl;
            world.temperature = worldgen::temperature(
                world.altitude,
                config.temperature_latitude_factor,
                config.temperature_altitude_factor,
                config.temperature_noise,
                config.temperature_bias,
                config.temperature_rng_seed
            );
        }
        else if (component == worldgen::Component::Ocean)
        {
            std::cout << "Computing ocean..." << std::endl;
            world.ocean_mask = worldgen::ocean_mask(
                world.altitude,
                world.temperature,
                config.ocean_boil_temperature
            );
        }
        else if (component == worldgen::Component::Ice)
        {
            std::cout << "Computing ice..." << std::endl;
            world.ice_mask = worldgen::ice_mask(
                world.temperature,
                world.ocean_mask,
                config.ocean_freeze_temperature
            );
        }
        else if (component == worldgen::Component::Precipitation)
        {
            std::cout << "Computing precipitation..." << std::endl;
            world.precipitation = worldgen::precipitation(
                world.temperature,
                world.ocean_mask,
                config.climate_zones,
                config.precipitation_ocean_lookup,
                config.precipitation_ocean_distance_smoothing,
                config.precipitation_ocean_distance_factor
            );
        }
        else if (component == worldgen::Component::Rivers)
        {
            std::cout << "Computing rivers..." << std::endl;
            world.river_depth = worldgen::rivers(
                world.altitude,
                world.ocean_mask,
                world.ice_mask,
                config.river_block_size,
                config.river_ocean_lookup
            );
        }
    }

    std::cout << "Generating image..." << std::endl;
    
    std::vector<worldgen::MaterialVisual> material_visuals_vec;
    
    for (const worldgen::SurfaceMaterial& material: config.surface_materials)
    {
        material_visuals_vec.push_back(worldgen::MaterialVisual{ 
            material.visual_altitude_low, 
            material.visual_altitude_high 
        });
    }
    auto material_visuals = cudu::device::Array1D<worldgen::MaterialVisual>::from_ptr(
        material_visuals_vec.data(), 
        material_visuals_vec.size()
    );

    std::vector<worldgen::ClimateZoneVisual> climate_zone_visuals_vec;
    
    for (const worldgen::ClimateZone& zone: config.climate_zones)
    {
        climate_zone_visuals_vec.push_back(worldgen::ClimateZoneVisual{ 
            zone.visual_sparse, 
            zone.visual_dense 
        });
    }
    auto climate_zone_visuals = cudu::device::Array1D<worldgen::ClimateZoneVisual>::from_ptr(
        climate_zone_visuals_vec.data(), 
        climate_zone_visuals_vec.size()
    );

    worldgen::RenderingOptions rendering_options{
        material_visuals,
        climate_zone_visuals,

        config.visual_water_shallow,
        config.visual_water_deep,
        config.visual_water_depth_limit,

        config.visual_river_shallow,
        config.visual_river_deep,
        config.visual_river_depth_limit,

        config.visual_ice_thin,
        config.visual_ice_thick,
        config.visual_ice_temperature_limit
    };

    cudu::device::Array3D<unsigned char> bgr_array = worldgen::image(
        world.altitude,
        world.surface_material,
        world.temperature,
        world.ocean_mask,
        world.ice_mask,
        world.precipitation,
        world.river_depth,
        config.climate_zones,
        rendering_options
    );
    world.image = bgr_array.download();

    if (gui)
    {
        const cv::Mat image(
            int(world.image.shape()[0]),
            int(world.image.shape()[1]),
            CV_8UC3,
            world.image.begin()
        );
        cv::namedWindow(MAIN_WINDOW_TITLE, cv::WINDOW_AUTOSIZE);
        cv::imshow(MAIN_WINDOW_TITLE, image);
        cv::waitKey(30);

        /*HWND main_window = FindWindowA(nullptr, MAIN_WINDOW_TITLE);

        if (main_window)
        {
            SetWindowPos(main_window, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
        }*/
    }
    
    std::cout << "Ready" << std::endl;
}

void save(const cudu::host::Array3D<unsigned char>& image)
{
    const time_t time = std::time(nullptr);
    const tm local_time = *std::localtime(&time);

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y_%m_%d_%H_%M_%S");
    const std::string time_str = oss.str();

    const std::string file_name = "world_" + time_str + ".png";

    const cv::Mat image_cv(
        int(image.shape()[0]),
        int(image.shape()[1]),
        CV_8UC3,
        const_cast<unsigned char*>(image.begin())
    );
    cv::imwrite(file_name, image_cv);

    std::cout << "Saved " + file_name << std::endl;
}

worldgen::Config run(
    const std::string& config_path,
    WorldData& world)
{
    const worldgen::Config config = load_config(config_path);
    generate_world(
        config, 
        {
            worldgen::Component::Altitude,
            worldgen::Component::Material,
            worldgen::Component::Temperature,
            worldgen::Component::Ocean,
            worldgen::Component::Ice,
            worldgen::Component::Precipitation,
            worldgen::Component::Rivers
        },
        cudu::device::Array1D<float>(),
        cudu::device::Array1D<float>(),
        world
    );
    return config;
}

worldgen::Config rerun(
    const std::string& config_path,
    const worldgen::Config& config_old,
    WorldData& world)
{
    const worldgen::Config config_new = load_config(config_path);
    generate_world(
        config_new,
        config_diff_recomputation_effect(config_old, config_new),
        cudu::device::Array1D<float>(),
        cudu::device::Array1D<float>(),
        world
    );
    return config_new;
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " CONFIG_FILE" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string config_path(argv[1]);
    
    WorldData world;
    worldgen::Config config = run(config_path, world);

    filewatch::FileWatcher config_watcher(
        config_path,
        std::chrono::milliseconds(1000)
    );
    const auto on_config_file_change = [&config, &world](
        const std::string& path,
        const filewatch::EventKind event_kind)
    {
        switch (event_kind) {
            case filewatch::EventKind::Created: config = rerun(path, config, world); break;
            case filewatch::EventKind::Modified: config = rerun(path, config, world); break;
            default: break;
        }
    };
    std::thread config_watcher_thread([&config_watcher, &on_config_file_change]() {
        config_watcher.start(on_config_file_change);
    });
    
    constexpr char KEYCODE_ESCAPE = 27;
    constexpr char KEYCODE_SAVE = 's';
    char keycode = 0;
    
    while (cv::getWindowProperty(MAIN_WINDOW_TITLE, 0) >= 0 &&
           keycode != KEYCODE_ESCAPE)
    {
        switch (cv::waitKey(30))
        {
            case KEYCODE_SAVE: save(world.image);  break;
            default: break;
        }
    }

    config_watcher.stop();
    config_watcher_thread.join();

    return EXIT_SUCCESS;
}
