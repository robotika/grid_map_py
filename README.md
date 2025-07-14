\# Grid Map Python Bindings



This repository provides Python bindings for the powerful C++ grid map library from \[ANYbotics](https://github.com/anybotics/grid\_map).



This allows for the creation, manipulation, and processing of 2D grid maps with multiple data layers directly in Python, making it useful for robotics applications, path planning, and terrain mapping.



\## Original Library



All credit for the core library goes to the team at ANYbotics.

\* \*\*Original Repository:\*\* \[github.com/anybotics/grid\_map](https://github.com/anybotics/grid\_map)



\## Installation



```bash

\# Clone this repository and its dependencies

git clone --recurse-submodules \[https://github.com/](https://github.com/)<your-username>/grid\_map\_py

cd grid\_map\_py



\# Install build tools and build the wheel

pip install build

python -m build --wheel



\# Install the built package

pip install dist/grid\_map\_py-\*.whl



