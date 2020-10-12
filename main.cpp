#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "config.h"
#include "cudu.h"
#include "filewatch.h"
#include "worldgen.h"

worldgen::Config load_config(const std::string& path)
{
    std::cout << "Loading configuration: " << path << std::endl;
    auto config = worldgen::Config::from_json_file(path);
    return config;
}

void generate_world(const worldgen::Config& config)
{
    std::cout << "Computing surface altitude..." << std::endl;
    cudu::Array2D<float> surface_altitude = worldgen::surface_altitude(
        config.surface_altitude_level_of_detail,
        config.surface_altitude_noise_initial,
        config.surface_altitude_noise_scale_factor,
        config.surface_altitude_bias,
        config.surface_altitude_rng_seed,
        config.compute_max_nr_threads
    );
    std::cout << "Computing surface material..." << std::endl;
    cudu::Array3D<float> surface_material = worldgen::surface_material(
        {surface_altitude.shape(0), surface_altitude.shape(1)},
        config.surface_materials,
        config.surface_material_cell_count,
        config.surface_material_cell_blur,
        config.surface_material_cell_rng_seed,
        config.compute_max_nr_threads
    );
    std::cout << "Computing temperature..." << std::endl;
    cudu::Array2D<float> temperature = worldgen::temperature(
        surface_altitude,
        config.temperature_latitude_factor,
        config.temperature_altitude_factor,
        config.temperature_noise,
        config.temperature_bias,
        config.temperature_rng_seed,
    config.compute_max_nr_threads
    );
    std::cout << "Computing ocean..." << std::endl;
    cudu::Array2D<bool> ocean_mask = worldgen::ocean(
        surface_altitude,
        temperature,
        config.ocean_boil_temperature,
        config.compute_max_nr_threads
    );
    std::cout << "Computing ice..." << std::endl;
    cudu::Array2D<bool> ice_mask = worldgen::ice(
        temperature,
        ocean_mask,
        config.ocean_freeze_temperature,
        config.compute_max_nr_threads
    );
    std::cout << "Computing precipitation..." << std::endl;
    cudu::Array2D<float> precipitation = worldgen::precipitation(
        temperature,
        ocean_mask,
        config.climate_zones,
        config.climate_zone_weight_min,
        config.precipitation_ocean_lookup,
        config.precipitation_ocean_distance_factor,
        config.compute_max_nr_threads
    );
    std::cout << "Computing rivers..." << std::endl;
    cudu::Array2D<float> river_depth = worldgen::rivers(
        surface_altitude,
        ocean_mask,
        ice_mask,
        config.river_block_size,
        config.river_ocean_lookup,
        config.compute_max_nr_threads
    );

    std::cout << "Generating image..." << std::endl;
    
    std::vector<worldgen::MaterialVisual> material_visuals_vec;
    for (const worldgen::SurfaceMaterial& material : config.surface_materials)
    {
        material_visuals_vec.push_back(worldgen::MaterialVisual{ material.visual_altitude_low, material.visual_altitude_high });
    }
    auto material_visuals = cudu::Array1D<worldgen::MaterialVisual>::from_container(material_visuals_vec);

    std::vector<worldgen::ClimateZoneVisual> climate_zone_visuals_vec;
    for (const worldgen::ClimateZone& zone : config.climate_zones)
    {
        climate_zone_visuals_vec.push_back(worldgen::ClimateZoneVisual{ zone.visual_sparse, zone.visual_dense });
    }
    auto climate_zone_visuals = cudu::Array1D<worldgen::ClimateZoneVisual>::from_container(climate_zone_visuals_vec);

    worldgen::RenderingOptions rendering_options{
        
        material_visuals,

        config.visual_water_shallow,
        config.visual_water_deep,
        config.visual_water_depth_limit,

        config.visual_ice_thin,
        config.visual_ice_thick,
        config.visual_ice_temperature_limit,

        climate_zone_visuals
    };

    cudu::Array3D<unsigned char> bgr_array = worldgen::image(
        surface_altitude,
        surface_material,
        temperature,
        ocean_mask,
        ice_mask,
        precipitation,
        river_depth,
        config.climate_zones,
        rendering_options,
        config.compute_max_nr_threads
    );

    std::vector<unsigned char> bgr_vector(bgr_array.size());

    bgr_array.block().download_all(bgr_vector.data());

    const cv::Mat image(
        int(bgr_array.shape(0)),
        int(bgr_array.shape(1)),
        CV_8UC3,
        bgr_vector.data()
    );
    cv::imshow("World", image);
    cv::waitKey(30);
}

void run(const std::string& config_path)
{
    try 
    {
        const worldgen::Config config = load_config(config_path);
        generate_world(config);
    }
    catch (const std::exception& exception) 
    {
        std::cerr << "failed to load configuration: " << exception.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void on_config_file_change(
    const std::string& path,
    const filewatch::EventKind event_kind)
{
    switch (event_kind) {
        case filewatch::EventKind::Created: run(path); break;
        case filewatch::EventKind::Modified: run(path); break;
        default: break;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " CONFIG_FILE" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string config_path(argv[1]);

    run(config_path);
    
    filewatch::FileWatcher config_watcher(
        config_path,
        std::chrono::milliseconds(1000)
    );
    std::thread config_watcher_thread([&config_watcher]() {
        config_watcher.start(on_config_file_change);
    });
    
    constexpr char KEYCODE_ESCAPE = 27;
    char keycode = 0;
    
    while (keycode != KEYCODE_ESCAPE)
    {
        keycode = cv::waitKey(30);
    }

    config_watcher.stop();
    config_watcher_thread.join();

    return EXIT_SUCCESS;
}
