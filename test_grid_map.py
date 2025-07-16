import numpy as np
import grid_map_py as gm

# 1. Create a map and add an initial layer
my_map = gm.GridMap()
my_map.set_geometry(length=np.array([1.2, 0.8]), resolution=0.1)
initial_image = np.full(my_map.get_size(), 1000, dtype=np.uint16) # 1 meter everywhere
my_map.set_layer_from_numpy("elevation", initial_image)

# 2. GET a layer as a NumPy array
elevation_data = my_map.get_layer_as_numpy("elevation")
print("Original data type:", elevation_data.dtype)
print("Value at (0, 0):", elevation_data[0, 0])

# 3. MODIFY the NumPy array in Python
# Let's set the center of the map to be higher (2000 mm = 2m)
center_row, center_col = elevation_data.shape[0] // 2, elevation_data.shape[1] // 2
elevation_data[center_row-2:center_row+2, center_col-2:center_col+2] = 2000

# 4. UPDATE the layer in the grid map with the modified data
# Note: our function expects uint16, but our get_layer returns float32. We must cast it back.
my_map.set_layer_from_numpy("elevation", (elevation_data * 1000).astype(np.uint16))

# 5. VERIFY the change by getting the data again
updated_elevation_data = my_map.get_layer_as_numpy("elevation")
print("Updated value at center:", updated_elevation_data[center_row, center_col])
print("Original value at (0,0) is unchanged:", updated_elevation_data[0, 0])
