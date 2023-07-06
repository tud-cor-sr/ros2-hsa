# Planar HSA control launch file
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os

RECORD_BAG = False  # Record data to rosbag file
BAG_PATH = "/home/mstoelzle/phd/rosbags"
LOG_LEVEL = "warn"


def generate_launch_description():
    # Create the NatNet client node
    natnet_config = os.path.join(
        get_package_share_directory("mocap_optitrack_client"),
        "config",
        "natnetclient.yaml",
    )
    # Create the world to base client
    w2b_config = os.path.join(
        get_package_share_directory("hsa_inverse_kinematics"),
        "config",
        "world_to_base_y_up.yaml",
    )

    launch_actions = [
        Node(
            package="mocap_optitrack_client",
            executable="mocap_optitrack_client",
            name="natnet_client",
            parameters=[natnet_config],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="mocap_optitrack_w2b",
            executable="mocap_optitrack_w2b",
            name="world_to_base",
            parameters=[w2b_config],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="hsa_inverse_kinematics",
            executable="planar_cs_ik_node",
            name="inverse_kinematics",
        ),
        Node(
            package="dynamixel_control",
            executable="sync_read_single_write_node",
            name="dynamixel_control",
        ),
        Node(
            package="hsa_planar_control",
            executable="planar_mb_control_node",
            name="planar_mb_control",
        )
    ]

    if RECORD_BAG:
        launch_actions.append(
            ExecuteProcess(
                cmd=["ros2", "bag", "record", "-a", "-o", BAG_PATH], output="screen"
            )
        )

    return LaunchDescription(launch_actions)
