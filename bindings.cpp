#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "grid_map_core/GridMap.hpp"
#include "grid_map_core/iterators/GridMapIterator.hpp"

namespace py = pybind11;

// This function now works as an "upsert": it adds a new layer
// or updates an existing one from a NumPy array.
void upsert_layer_from_numpy(grid_map::GridMap& map, const std::string& layer_name, py::array_t<uint16_t, py::array::c_style | py::array::forcecast> image) {
    py::buffer_info buf = image.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("NumPy array must be 2-dimensional.");
    }
    if (map.getSize()(0) != buf.shape[0] || map.getSize()(1) != buf.shape[1]) {
        throw std::runtime_error("NumPy array dimensions do not match grid map dimensions!");
    }

    // If layer does not exist, add it.
    if (!map.exists(layer_name)) {
        map.add(layer_name);
    }
    
    grid_map::Matrix& layer = map.get(layer_name);

    // Loop through the numpy array and fill the grid map layer.
    for (int r = 0; r < buf.shape[0]; ++r) {
        for (int c = 0; c < buf.shape[1]; ++c) {
            uint16_t val = *(reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(buf.ptr) + r * buf.strides[0] + c * buf.strides[1]));
            layer(r, c) = static_cast<float>(val) / 1000.0f; // Convert mm to meters
        }
    }
}

// This is the function that defines our Python module.
PYBIND11_MODULE(grid_map_py, m) {
    m.doc() = "Python bindings for the ANYbotics grid_map library";

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

        // Renamed function for clarity
        .def("set_layer_from_numpy", &upsert_layer_from_numpy, "Adds or updates a layer from a uint16 NumPy array.",
             py::arg("layer_name"), py::arg("image"))

        // NEW: Get a layer as a NumPy array. pybind11 handles the Eigen::Matrix -> numpy conversion.
        .def("get_layer_as_numpy", [](grid_map::GridMap& map, const std::string& layer_name) -> grid_map::Matrix {
            return map.get(layer_name);
        }, "Gets a layer's data as a NumPy array.", py::arg("layer_name"))

        // Getters
        .def("get_position", getPositionPtr)
        .def("get_length", getLengthPtr)
        .def("get_resolution", getResolutionPtr)
        .def("get_layers", &grid_map::GridMap::getLayers)
        .def("get_frame_id", &grid_map::GridMap::getFrameId)
        .def("get_timestamp", &grid_map::GridMap::getTimestamp)
        .def("get_size", &grid_map::GridMap::getSize)
        .def("exists", &grid_map::GridMap::exists, "Check if a layer exists.", py::arg("layer_name"));
}
