import numpy as np
import grid_map_py as gm

# 1. Create a GridMap object
my_map = gm.GridMap()
resolution = 0.01
length = np.array([1.2, 0.8]) # [length_x, length_y]

# 2. Set its geometry
my_map.set_geometry(length=length, resolution=resolution)

# 3. Create a dummy uint16 depth image
rows, cols = my_map.get_size()
depth_image = np.random.randint(500, 3000, size=(rows, cols), dtype=np.uint16)

# 4. Add the depth data as a new layer
my_map.add_layer_from_numpy("elevation", depth_image)

# 5. Get values back
print("Layers in map:", my_map.get_layers())
print("Map resolution:", my_map.get_resolution())
