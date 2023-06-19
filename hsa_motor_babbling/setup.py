from setuptools import find_packages, setup

package_name = 'hsa_motor_babbling'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', "pygbn"],
    zip_safe=True,
    maintainer='root',
    maintainer_email='maximilian@stoelzle.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planar_motor_babbling_node = hsa_motor_babbling.planar_motor_babbling_node:main'
        ],
    },
)
