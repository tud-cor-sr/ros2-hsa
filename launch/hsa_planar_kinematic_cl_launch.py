# CLosed-loop kinematic controller for planar HSA robot
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hsa_inverse_kinematics',
            executable='planar_cs_ik_node',
            name='inverse_kinematics'
        ),
        Node(
            package='dynamixel_sdk_examples',
            executable='read_write_node',
            name='dynamixel_control'
        ),
        Node(
            package='hsa_kinematic_control',
            executable='planar_kinematic_control_node',
            name='kinematic_control'
        )
    ])
