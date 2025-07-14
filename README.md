# Grid Map Python Bindings


This repository provides Python bindings for the powerful C++ grid map library from [ANYbotics](https://github.com/ANYbotics/grid_map).


This allows for the creation, manipulation, and processing of 2D grid maps with multiple data layers directly in Python, making it useful for robotics applications, path planning, and terrain mapping.



## Original Library

All credit for the core library goes to the team at ANYbotics.

* **Original Repository:** [github.com/ANYbotics/grid_map](https://github.com/ANYbotics/grid_map)
* dependency [gitlab.com/libeigen/eigen](https://gitlab.com/libeigen/eigen)



## Installation

```bash

# Clone this repository and its dependencies

git clone --recurse-submodules https://github.com/robotika/grid_map_py

cd grid_map_py



# Install build tools and build the wheel

pip install build

python -m build --wheel



# Install the built package

pip install dist/grid_map_py-*.whl
