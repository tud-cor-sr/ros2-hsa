from setuptools import find_packages, setup

package_name = "hsa_sim"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "diffrax",
        "dill",
        "jax",
        "numpy",
        "pandas",
        "setuptools",
        "sympy>=1.11",
        "tornado",
        "tqdm",
    ],
    zip_safe=True,
    maintainer="Maximilian Stölzle",
    maintainer_email="maximilian@stoelzle.ch",
    description="A ROS2 wrapper for the HSA simulation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["planar_sim_node = hsa_sim.planar_sim_node:main"],
    },
)
