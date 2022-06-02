Pseudo random world generator using procedural elevation, temperature, lakes, rivers, biomes, etc.
To compile, you may need to alter the CUDA compute flag to match the capability of your GPU, and also specify the path to your local installation of OpenCV.
Run the compiled executable with a given [configuration](config/earth_from_orbit.json), and a GUI should open and further instructions should appear in the console.
Also included is a [Python script](hillshade_render.py) for rendering exported color and heightmap images as a relief map in Blender.

![Sample world render](https://www.rharel.com/world-9.fe5a8368.jpeg)
