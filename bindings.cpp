#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "grid_map_core/GridMap.hpp"
#include "grid_map_core/iterators/GridMapIterator.hpp"

namespace py = pybind11;

/**
 * @brief Updates a grid map layer from a depth image.
 * * @param map The grid map to update.
 * @param layer_name The name of the layer to update (e.g., "elevation").
 * @param depth_image The depth image as a NumPy array (uint16, in millimeters).
 * @param camera_intrinsics The 3x3 camera intrinsic matrix (fx, 0, cx; 0, fy, cy; 0, 0, 1).
 * @param camera_pose The 4x4 transformation matrix representing the camera's pose in the world frame.
 */
void update_from_depth_image(grid_map::GridMap& map,
                             const std::string& layer_name,
                             py::array_t<uint16_t, py::array::c_style | py::array::forcecast> depth_image,
                             const Eigen::Matrix3d& camera_intrinsics,
                             const Eigen::Matrix4d& camera_pose_matrix) {

    py::buffer_info buf = depth_image.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Depth image must be 2-dimensional.");
    }

    Eigen::Isometry3d camera_pose(camera_pose_matrix);

    if (!map.exists(layer_name)) {
        map.add(layer_name, 0.0); // Add layer with default value if it doesn't exist
    }

    const double fx = camera_intrinsics(0, 0);
    const double fy = camera_intrinsics(1, 1);
    const double cx = camera_intrinsics(0, 2);
    const double cy = camera_intrinsics(1, 2);

    const int rows = buf.shape[0];
    const int cols = buf.shape[1];
    uint16_t* ptr = static_cast<uint16_t*>(buf.ptr);

    // Iterate through each pixel of the depth image
    for (int v = 0; v < rows; ++v) {
        for (int u = 0; u < cols; ++u) {
            uint16_t depth_mm = ptr[v * cols + u];
            if (depth_mm == 0) continue; // Skip invalid depth pixels

            double depth_m = static_cast<double>(depth_mm) / 1000.0;

            // Unproject pixel to a 3D point in the camera's frame
            Eigen::Vector3d point_in_camera;
            point_in_camera.x() = (static_cast<double>(u) - cx) * depth_m / fx;
            point_in_camera.y() = (static_cast<double>(v) - cy) * depth_m / fy;
            point_in_camera.z() = depth_m;

            // Transform the point from the camera's frame to the world frame
            Eigen::Vector3d point_in_world = camera_pose * point_in_camera;

            // Get the 2D position for the grid map
            grid_map::Position position(point_in_world.x(), point_in_world.y());
            
            // If the point is within the map, set the elevation
            if (map.isInside(position)) {
                map.atPosition(layer_name, position) = point_in_world.z();
            }
        }
    }
}


// --- The rest of your bindings.cpp file ---
// (The PYBIND11_MODULE section)
// ...
// We just need to add a line to it.

PYBIND11_MODULE(grid_map_py, m) {
    m.doc() = "Python bindings for the ANYbotics grid_map library";

    // ... (all the function pointer definitions from before) ...
    void (grid_map::GridMap::*setGeometryPtr)(const grid_map::Length&, double, const grid_map::Position&) = &grid_map::GridMap::setGeometry;
    void (grid_map::GridMap::*setPositionPtr)(const grid_map::Position&) = &grid_map::GridMap::setPosition;
    const grid_map::Position& (grid_map::GridMap::*getPositionPtr)() const = &grid_map::GridMap::getPosition;
    const grid_map::Length& (grid_map::GridMap::*getLengthPtr)() const = &grid_map::GridMap::getLength;
    double (grid_map::GridMap::*getResolutionPtr)() const = &grid_map::GridMap::getResolution;

    py::class_<grid_map::GridMap>(m, "GridMap")
        .def(py::init<>(), "Default constructor")
        // Setters
        .def("set_geometry", setGeometryPtr, "Sets the map geometry.", py::arg("length"), py::arg("resolution"), py::arg("position") = grid_map::Position::Zero())
        .def("set_position", setPositionPtr, "Sets the 2D position of the map center.", py::arg("position"))
        .def("set_frame_id", &grid_map::GridMap::setFrameId, py::arg("frame_id"))
        .def("set_timestamp", &grid_map::GridMap::setTimestamp, py::arg("timestamp"))

        // ADD THIS NEW FUNCTION BINDING
        .def("update_from_depth_image", &update_from_depth_image, "Updates the map from a depth image.",
             py::arg("layer_name"), py::arg("depth_image"), py::arg("camera_intrinsics"), py::arg("camera_pose"))

        // ... (the rest of the bindings for get_layer, etc.) ...
        .def("get_layer_as_numpy", [](grid_map::GridMap& map, const std::string& layer_name) -> grid_map::Matrix {
            return map.get(layer_name);
        }, "Gets a layer's data as a NumPy array.", py::arg("layer_name"))
        .def("exists", &grid_map::GridMap::exists, "Check if a layer exists.", py::arg("layer_name"))
        // Getters
        .def("get_position", getPositionPtr)
        .def("get_length", getLengthPtr)
        .def("get_resolution", getResolutionPtr)
        .def("get_layers", &grid_map::GridMap::getLayers)
        .def("get_frame_id", &grid_map::GridMap::getFrameId)
        .def("get_timestamp", &grid_map::GridMap::getTimestamp)
        .def("get_size", &grid_map::GridMap::getSize);
}
