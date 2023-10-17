from glob import glob
import os
from setuptools import find_packages, setup

package_name = "hsa_joy_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (os.path.join("share", package_name), glob("launch/*.py")),
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Maximilian Stolzle",
    author_email="maximilian@stoelzle.ch",
    maintainer="Maximilian Stolzle",
    maintainer_email="maximilian@stoelzle.ch",
    description="Control of (planar) HSA robots based on Joy signals.",
    license="MIT",
    tests_require=["numpy", "pytest"],
    entry_points={
        "console_scripts": [
            "planar_hsa_bending_joy_control_node = hsa_joy_control.planar_hsa_bending_joy_control_node:main",
            "planar_hsa_cartesian_joy_control_node = hsa_joy_control.planar_hsa_cartesian_joy_control_node:main",
        ],
    },
)
