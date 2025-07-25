import numpy as np
import grid_map_py as gm
from scipy.spatial.transform import Rotation

# 1. Setup the map
my_map = gm.GridMap()
my_map.set_geometry(length=np.array([10.0, 10.0]), resolution=0.1, position=np.array([0.0, 0.0]))
my_map.set_frame_id("world")

# 2. Define camera parameters
# A simple dummy depth image (e.g., 64x48)
depth_image = np.full((48, 64), 2000, dtype=np.uint16) # A flat wall 2 meters away
depth_image[20:30, 30:40] = 1000 # A closer box 1 meter away

# Camera intrinsics (fx, fy, cx, cy)
intrinsics = np.array([
    [50.0, 0.0,  32.0],
    [0.0,  50.0, 24.0],
    [0.0,  0.0,  1.0]
], dtype=np.float64)

# Camera pose: looking at the center of the map from 5m away on the y-axis
# The camera looks along its -Z axis. To look at the origin from +Y,
# we position it at (0, 5, 0) and rotate it -90 degrees around the X-axis.
cam_pos = [0, 5, 0]
cam_rot = Rotation.from_euler('x', 90, degrees=True).as_matrix()

camera_pose = np.eye(4, dtype=np.float64)
camera_pose[:3, :3] = cam_rot
camera_pose[:3, 3] = cam_pos

# 3. Call the new function to update the map
my_map.update_from_depth_image(
    layer_name="elevation",
    depth_image=depth_image,
    camera_intrinsics=intrinsics,
    camera_pose=camera_pose
)

# 4. (Optional) You can retrieve the layer to verify it
# The map should now have elevation values around z=0, where the "wall" was projected.
elevation_layer = my_map.get_layer_as_numpy("elevation")

# Find where the map was updated (non-zero values)
updated_points = elevation_layer[np.abs(elevation_layer) > 1e-6]
print(f"Map updated with {len(updated_points)} points.")
print(f"Sample elevation values: {updated_points[:5]}")
