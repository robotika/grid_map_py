import sys
from setuptools import setup, Extension
import pybind11

# The definitive and complete list of C++ source files to compile.
sources = [
    'bindings.cpp',
    '../grid_map/grid_map_core/src/GridMap.cpp',
    '../grid_map/grid_map_core/src/GridMapMath.cpp',
    '../grid_map/grid_map_core/src/BufferRegion.cpp',
    '../grid_map/grid_map_core/src/SubmapGeometry.cpp',
    '../grid_map/grid_map_core/src/Polygon.cpp',
    '../grid_map/grid_map_core/src/CubicInterpolation.cpp',
    '../grid_map/grid_map_core/src/iterators/GridMapIterator.cpp',
    '../grid_map/grid_map_core/src/iterators/SubmapIterator.cpp',
    '../grid_map/grid_map_core/src/iterators/CircleIterator.cpp',
    '../grid_map/grid_map_core/src/iterators/SpiralIterator.cpp',
    '../grid_map/grid_map_core/src/iterators/LineIterator.cpp',
    '../grid_map/grid_map_core/src/iterators/PolygonIterator.cpp'
]

ext_modules = [
    Extension(
        'grid_map_py',
        sources,  # Use the complete list of sources
        include_dirs=[
            pybind11.get_include(),
            '../grid_map/grid_map_core/include',
            '../eigen'
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]

setup(
    name='grid_map_py',
    version='0.0.1',
    author='Your Name',
    author_email='your@email.com',
    description='Python bindings for ANYbotics grid_map',
    ext_modules=ext_modules,
)
