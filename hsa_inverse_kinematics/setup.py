from glob import glob
import os
from setuptools import setup

package_name = "hsa_inverse_kinematics"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["dill", "jax", "jsrm", "numpy", "setuptools", "sympy>=1.11"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="maximilian@stoelzle.ch",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "planar_cs_ik_node = hsa_inverse_kinematics.planar_cs_ik_node:main"
        ],
    },
)
