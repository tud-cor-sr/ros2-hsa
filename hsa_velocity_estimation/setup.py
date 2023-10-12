from setuptools import find_packages, setup

package_name = 'hsa_velocity_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['derivative', 'jax', 'numderivax', 'setuptools'],
    zip_safe=True,
    maintainer='Maximilian Stoelzle',
    maintainer_email='maximilian@stoelzle.ch',
    description='Package for estimating the velocity from position time sequences using numerical differentiation. Examples include configuration and end-effector position.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planar_hsa_velocity_estimator_node = hsa_velocity_estimation.planar_hsa_velocity_estimator_node:main'
        ],
    },
)
