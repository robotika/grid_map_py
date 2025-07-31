import numpy as np
from scipy.spatial.transform import Rotation # For camera pose handling

class PurePythonGridMap:
    """
    A pure Python implementation of a GridMap, mimicking the core functionality
    of the ANYbotics grid_map library for debugging purposes.
    This version is designed to be fully compatible with the API of our
    'grid_map_py' wrapper.
    """

    def __init__(self):
        """
        Initializes an empty grid map. Geometry must be set separately.
        """
        self._layers = {}  # Dictionary to store layers: {'layer_name': numpy_array}
        self._length = np.array([0.0, 0.0]) # [length_x, length_y]
        self._resolution = 0.0
        self._position = np.array([0.0, 0.0]) # [pos_x, pos_y]
        self._frame_id = "map"
        self._timestamp = 0

    def set_geometry(self, length: np.ndarray, resolution: float, position: np.ndarray = np.array([0.0, 0.0])) -> None:
        """
        Sets the map geometry (dimensions, resolution, and position).
        Reinitializes all layers if geometry changes.

        Args:
            length (np.ndarray): [length_x, length_y] in meters.
            resolution (float): Resolution of the grid cells in meters/cell.
            position (np.ndarray): [pos_x, pos_y] of the map center in meters.
        """
        if not isinstance(length, np.ndarray) or length.shape != (2,):
            raise ValueError("Length must be a 2-element NumPy array.")
        if not isinstance(position, np.ndarray) or position.shape != (2,):
            raise ValueError("Position must be a 2-element NumPy array.")
        if resolution <= 0:
            raise ValueError("Resolution must be positive.")

        self._length = length
        self._resolution = resolution
        self._position = position

        # Calculate grid size (rows, cols) based on length and resolution
        # GridMap uses (rows, cols) where rows correspond to length_y and cols to length_x
        rows = int(np.round(self._length[1] / self._resolution))
        cols = int(np.round(self._length[0] / self._resolution))
        self._size = np.array([rows, cols], dtype=int)

        # Reinitialize layers with new dimensions
        for layer_name, layer_data in self._layers.items():
            self._layers[layer_name] = np.zeros(self._size, dtype=layer_data.dtype)
        
        print(f"Geometry set: Length={self._length}, Resolution={self._resolution}, Position={self._position}, Size={self._size}")


    def get_size(self) -> np.ndarray:
        """
        Returns the size of the grid map in cells (rows, cols).
        """
        return self._size

    def get_layers(self) -> list[str]:
        """
        Returns a list of layer names in the map.
        """
        return list(self._layers.keys())

    def exists(self, layer_name: str) -> bool:
        """
        Checks if a layer exists in the map.
        """
        return layer_name in self._layers

    def set_layer_from_numpy(self, layer_name: str, image: np.ndarray) -> None:
        """
        Adds or updates a layer from a NumPy array.
        The input image is assumed to be uint16 (mm) and converted to float (m).

        Args:
            layer_name (str): The name of the layer.
            image (np.ndarray): A 2D NumPy array (uint16) representing the layer data.
        """
        if image.ndim != 2:
            raise ValueError("Input image must be 2-dimensional.")
        if image.shape[0] != self._size[0] or image.shape[1] != self._size[1]:
            raise ValueError(f"NumPy array dimensions {image.shape} do not match grid map dimensions {self._size}!")
        if image.dtype != np.uint16:
            print(f"Warning: Input image dtype is {image.dtype}, expected uint16. Data will be cast.")
            image = image.astype(np.uint16)

        # Convert uint16 (mm) to float (m) for internal storage
        self._layers[layer_name] = image.astype(np.float32) / 1000.0
        # print(f"Layer '{layer_name}' {'updated' if self.exists(layer_name) else 'added'} from numpy.")

    def get_layer_as_numpy(self, layer_name: str) -> np.ndarray:
        """
        Gets a layer's data as a NumPy array (float32, in meters).
        """
        if not self.exists(layer_name):
            raise KeyError(f"Layer '{layer_name}' does not exist.")
        return self._layers[layer_name].copy() # Return a copy to prevent external modification of internal state

    def set_frame_id(self, frame_id: str) -> None:
        """
        Sets the frame ID of the map.
        """
        self._frame_id = frame_id

    def get_frame_id(self) -> str:
        """
        Returns the frame ID of the map.
        """
        return self._frame_id

    def set_timestamp(self, timestamp: int) -> None:
        """
        Sets the timestamp of the map.
        """
        self._timestamp = timestamp

    def get_timestamp(self) -> int:
        """
        Returns the timestamp of the map.
        """
        return self._timestamp

    def set_position(self, position: np.ndarray) -> None:
        """
        Sets the 2D position of the map center.
        """
        if not isinstance(position, np.ndarray) or position.shape != (2,):
            raise ValueError("Position must be a 2-element NumPy array.")
        self._position = position

    def get_position(self) -> np.ndarray:
        """
        Returns the 2D position of the map center.
        """
        return self._position

    def get_length(self) -> np.ndarray:
        """
        Returns the length of the map in meters [length_x, length_y].
        """
        return self._length

    def get_resolution(self) -> float:
        """
        Returns the resolution of the map in meters/cell.
        """
        return self._resolution

    def _is_inside(self, world_x: float, world_y: float) -> bool:
        """
        Checks if a world coordinate (x, y) is inside the map boundaries.
        """
        map_min_x = self._position[0] - self._length[0] / 2.0
        map_max_x = self._position[0] + self._length[0] / 2.0
        map_min_y = self._position[1] - self._length[1] / 2.0
        map_max_y = self._position[1] + self._length[1] / 2.0
        # Use strict inequality for max boundary to match C++ behavior
        return map_min_x <= world_x < map_max_x and map_min_y <= world_y < map_max_y

    def _world_to_grid_index(self, world_x: float, world_y: float) -> tuple[int, int]:
        """
        Converts world coordinates (x, y) to grid cell indices (row, col).
        This method now mimics the exact logic from the C++ grid_map library.
        """
        # Position relative to the map's center.
        position_in_map_frame_x = world_x - self._position[0]
        position_in_map_frame_y = world_y - self._position[1]

        # Half lengths of the map.
        half_length_x = self._length[0] / 2.0
        half_length_y = self._length[1] / 2.0

        # Calculate indices based on the C++ source code formula.
        # This correctly maps world coordinates to array indices where (0,0) is top-left.
        row_idx = int(round((half_length_y - position_in_map_frame_y) / self._resolution))
        col_idx = int(round((half_length_x + position_in_map_frame_x) / self._resolution)) # Note the '+' for x-axis

        return row_idx, col_idx

    def _is_valid_grid_index(self, row: int, col: int) -> bool:
        """
        Checks if grid indices (row, col) are within the map's bounds.
        """
        return 0 <= row < self._size[0] and 0 <= col < self._size[1]


    def update_from_depth_image(self,
                                layer_name: str,
                                depth_image: np.ndarray,
                                camera_intrinsics: np.ndarray,
                                camera_pose: np.ndarray) -> None:
        """
        Updates a grid map layer from a depth image.

        Args:
            map (PurePythonGridMap): The grid map to update.
            layer_name (str): The name of the layer to update (e.g., "elevation").
            depth_image (np.ndarray): The depth image as a NumPy array (uint16, in millimeters).
            camera_intrinsics (np.ndarray): The 3x3 camera intrinsic matrix (fx, 0, cx; 0, fy, cy; 0, 0, 1).
            camera_pose (np.ndarray): The 4x4 transformation matrix representing the camera's pose in the world frame.
        """
        if depth_image.ndim != 2:
            raise ValueError("Depth image must be 2-dimensional.")
        if camera_intrinsics.shape != (3, 3):
            raise ValueError("Camera intrinsics must be a 3x3 matrix.")
        if camera_pose.shape != (4, 4):
            raise ValueError("Camera pose must be a 4x4 matrix.")

        if not self.exists(layer_name):
            # Create a new layer if it doesn't exist, initialized to NaN (unknown)
            self._layers[layer_name] = np.full(self._size, np.nan, dtype=np.float32)

        layer = self._layers[layer_name]

        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        rows_img, cols_img = depth_image.shape

        rotation_matrix = camera_pose[:3, :3]
        translation_vector = camera_pose[:3, 3]

        # Iterate through each pixel of the depth image
        for v in range(rows_img):
            for u in range(cols_img):
                depth_mm = depth_image[v, u]
                if depth_mm == 0: # Skip invalid depth pixels
                    continue

                depth_m = float(depth_mm) / 1000.0

                # Unproject pixel to a 3D point in the camera's frame
                point_in_camera = np.array([
                    (float(u) - cx) * depth_m / fx,
                    (float(v) - cy) * depth_m / fy,
                    depth_m
                ])

                # Transform the point from the camera's frame to the world frame
                point_in_world = np.dot(rotation_matrix, point_in_camera) + translation_vector

                world_x = point_in_world[0]
                world_y = point_in_world[1]
                world_z = point_in_world[2]

                # Check if the point is within the map's world boundaries
                if self._is_inside(world_x, world_y):
                    # Convert world coordinates to grid cell indices
                    row_idx, col_idx = self._world_to_grid_index(world_x, world_y)

                    # Check if grid indices are valid
                    if self._is_valid_grid_index(row_idx, col_idx):
                        # This simple Python version just overwrites the value.
                        # A more advanced version would handle fusion (averaging, etc.).
                        layer[row_idx, col_idx] = world_z


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing PurePythonGridMap ---")

    # 1. Setup the map
    my_map = PurePythonGridMap()
    resolution = 0.1
    length = np.array([15.0, 10.0]) # [length_x, length_y]
    my_map.set_geometry(length=length, resolution=resolution, position=np.array([0.0, 0.0]))
    my_map.set_frame_id("world")

    # 2. Define camera parameters
    depth_image = np.full((48, 64), 2000, dtype=np.uint16) # A flat wall 2 meters away
    depth_image[20:30, 30:40] = 1000 # A closer box 1 meter away

    intrinsics = np.array([
        [50.0, 0.0,  32.0],
        [0.0,  50.0, 24.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64)

    # Camera pose: looking at the center of the map from 5m away on the y-axis
    cam_pos = np.array([0, 5, 0], dtype=np.float64)
    # A +90 degree rotation around X-axis points the camera towards the origin.
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

    # 4. Verify the change
    elevation_layer = my_map.get_layer_as_numpy("elevation")
    
    # Count non-NaN points
    updated_points_count = np.sum(~np.isnan(elevation_layer))
    print(f"Map updated with {updated_points_count} points.")

    if updated_points_count > 0:
        # Find some non-NaN values for sampling
        non_nan_values = elevation_layer[~np.isnan(elevation_layer)]
        print(f"Sample elevation values: {non_nan_values[:5]}")

    # Test get/set layer
    print("\n--- Testing get/set layer ---")
    my_map.set_layer_from_numpy("test_layer", np.full(my_map.get_size(), 500, dtype=np.uint16))
    test_layer_data = my_map.get_layer_as_numpy("test_layer")
    print(f"Value from 'test_layer' at (0,0): {test_layer_data[0,0]} (should be 0.5m)")
    print(f"All layers: {my_map.get_layers()}")

    # Test geometry getters
    print("\n--- Testing geometry getters ---")
    print(f"Map resolution: {my_map.get_resolution()}")
    print(f"Map length: {my_map.get_length()}")
    print(f"Map size (rows, cols): {my_map.get_size()}")
    print(f"Map position: {my_map.get_position()}")
    print(f"Map frame ID: {my_map.get_frame_id()}")
