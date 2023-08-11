from setuptools import find_packages, setup

package_name = "hsa_visualization"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["dill", "jax", "opencv-python", "setuptools", "sympy>=1.11"],
    zip_safe=True,
    maintainer="Maximilian Stölzle",
    maintainer_email="maximilian@stoelzle.ch",
    description="ROS2 package for the visualization of an HSA robot using OpenCV.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["planar_viz_node = hsa_visualization.planar_viz_node:main"],
    },
)
