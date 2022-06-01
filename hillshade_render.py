"""
This script and accompanying blend file implement the shaded relief
rendering technique as described by Daniel Huffman [1].

1. https://somethingaboutmaps.wordpress.com/2017/11/16/creating-shaded-relief-in-blender/
"""

import sys

from pathlib import Path

import bpy


log_file = open("hillshade_render.log", "w")


def log(*args):
    print(*args, file=log_file, flush=True)


def render_image(heightmap_image_path: Path, color_image_path: Path, render_image_path: Path):
    log("Rendering image:", render_image_path.name)

    heightmap_image = bpy.data.images.load(str(heightmap_image_path))
    heightmap_image.colorspace_settings.name = "Linear"

    color_image = bpy.data.images.load(str(color_image_path))

    if heightmap_image.size[0] != color_image.size[0] or heightmap_image.size[1] != color_image.size[1]:
        log("Error: image dimension mismatch")

    log("Image size:", heightmap_image.size[0], "x", heightmap_image.size[1])

    image_aspect_ratio = heightmap_image.size[0] / heightmap_image.size[1]
    plane = bpy.data.objects["Plane"]
    plane.scale[0] = image_aspect_ratio * plane.scale[1]

    log("Plane scale:", plane.scale[0], "x", plane.scale[1])

    plane_material_nodes = bpy.data.materials["Plane Material"].node_tree.nodes
    plane_material_nodes["Heightmap Texture"].image = heightmap_image
    plane_material_nodes["Color Texture"].image = color_image

    camera = bpy.data.cameras["Camera"]
    camera.ortho_scale = 2 * max(plane.scale)
    bpy.context.scene.render.resolution_x = heightmap_image.size[0]
    bpy.context.scene.render.resolution_y = heightmap_image.size[1]
    bpy.context.scene.render.filepath = str(render_image_path)

    log("Orthographic scale:", camera.ortho_scale)

    bpy.ops.render.render(write_still=True)


def render_output_images(input_file_paths: list[Path]):
    for heightmap_image_path in input_file_paths:
        log("Input:", heightmap_image_path)
        image_name = "_".join(heightmap_image_path.name.split("_")[:-1])
        color_image_path = heightmap_image_path.with_name(image_name + "_combined_image.png")
        render_image_path = heightmap_image_path.with_name(image_name + "_render.png")
        if not heightmap_image_path.exists():
            log("Missing file:", heightmap_image_path)
        elif not color_image_path.exists():
            log("Missing file:", color_image_path)
        elif render_image_path.exists():
            log("Render image already exists, skipping:", render_image_path)
        else:
            render_image(heightmap_image_path, color_image_path, render_image_path)

input_file_paths = [Path(path_str).resolve() for path_str in sys.argv[sys.argv.index("--") + 1:]]

render_output_images(input_file_paths)
