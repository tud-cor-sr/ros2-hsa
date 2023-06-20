# CLosed-loop kinematic controller for planar HSA robot
from launch import LaunchDescription
from launch_ros.actions import ExecuteProcess, Node

RECORD_BAG = False   # Record data to rosbag file
BAG_PATH = "/home/mstoelzle/phd/rosbags"

def generate_launch_description():
    launch_actions = [
        Node(
            package='hsa_inverse_kinematics',
            executable='planar_cs_ik_node',
            name='inverse_kinematics'
        ),
        Node(
            package='dynamixel_control',
            executable='read_write_node',
            name='dynamixel_control'
        ),
        Node(
            package='hsa_kinematic_control',
            executable='planar_kinematic_control_node',
            name='kinematic_control'
        )
    ]

    if RECORD_BAG:
        launch_actions.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-a', '-o', BAG_PATH],
            output='screen'
        ))

    return LaunchDescription(launch_actions)
