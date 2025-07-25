import numpy as np
import grid_map_py as gm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def create_map():
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
    return my_map


def draw1(my_map):
    # Assuming 'my_map' is your grid_map object from the previous step
    elevation_layer = my_map.get_layer_as_numpy("elevation")

    # Create the plot
    plt.figure(figsize=(8, 6))
    # 'imshow' displays an array as an image.
    # We use a colormap ('terrain') to map values to colors.
    img = plt.imshow(elevation_layer, cmap='terrain')

    # Add a colorbar to show the mapping of colors to height values
    plt.colorbar(img, label='Elevation (m)')
    plt.title('2D Elevation Map')
    plt.xlabel('Map Columns')
    plt.ylabel('Map Rows')
    plt.show()


def draw2(my_map):
    # Assuming 'my_map' is your grid_map object
    elevation_layer = my_map.get_layer_as_numpy("elevation")

    # Create X and Y coordinates for each point in the grid
    rows, cols = elevation_layer.shape
    x = np.arange(0, cols, 1)
    y = np.arange(0, rows, 1)
    X, Y = np.meshgrid(x, y)

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, elevation_layer, cmap='viridis')

    ax.set_title('3D Surface Plot')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Elevation (m)')
    plt.show()


def draw3(my_map):
    import plotly.graph_objects as go
    
    # Assuming 'my_map' is your grid_map object
    elevation_layer = my_map.get_layer_as_numpy("elevation")

    # Create the interactive 3D surface plot
    fig = go.Figure(data=[go.Surface(z=elevation_layer, colorscale='Viridis')])

    fig.update_layout(
        title='Interactive 3D Elevation Map',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Elevation (m)',
            aspectratio=dict(x=1, y=1, z=0.4)  # Adjust z-axis scale for better viewing
        )
    )

    # This will open a new browser tab with the interactive plot
    fig.show()


draw3(create_map())
