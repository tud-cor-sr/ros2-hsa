#!/usr/bin/env python3
from ament_index_python.packages import get_package_share_directory
import numpy as np
import os
import rclpy
from pathlib import Path
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32
import socket


def decode_stimulation(byte_data):
    # Decode the first byte to determine the stimulation type
    stimulation_type = int(byte_data[0])

    # Return the decoded stimulation type
    return stimulation_type


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("stimulation_receiver_node")

    assets_dir = Path(get_package_share_directory("joylike_operation")) / "assets"
    switching_sound_path = str(
            assets_dir
            / "relax-message-tone.mp3"
    )

    node.declare_parameter("joy_control_mode", "cartesian")
    joy_control_mode = node.get_parameter(
        "joy_control_mode"
    ).value  # bending, cartesian, or cartesian_switch

    node.declare_parameter("joy_signal_topic", "joy_signal")
    joy_signal_topic = node.get_parameter("joy_signal_topic").value
    pub = node.create_publisher(Joy, joy_signal_topic, rclpy.qos.qos_profile_system_default)

    if joy_control_mode == "cartesian_switch":
        # initialize activate direction as the x-axis
        active_axis = 0
        cartesian_switch_state_pub = node.create_publisher(
            Int32, "cartesian_switch_state", rclpy.qos.qos_profile_system_default
        )

    if joy_control_mode == "bending":
        node.num_axes = 1
    else:
        node.declare_parameter("num_axes", 2)
        node.num_axes = node.get_parameter("num_axes").value

    node.declare_parameter("host", "localhost")
    host = node.get_parameter("host").value
    node.declare_parameter("port", 5678)
    port = node.get_parameter("port").value

    # define frequency of the main loop
    node.declare_parameter("frequency", 100.0)
    freq = node.get_parameter("frequency").value
    rate = rclpy.rate.Rate(freq)

    # history of samples
    lhs = 35
    t_hs = np.zeros((lhs, ))  # time associated with samples
    st_hs = np.zeros((lhs, ))  # history of stimulations
    sw_hs = np.zeros((lhs, ))  # history of switches

    # at least 80% of samples over the specified duration need to tell us to switch
    node.declare_parameter("accuracy_for_switch", 0.8)
    node.accuracy_for_switch = node.get_parameter("accuracy_for_switch").value

    # node.get_logger().warn(switching_sound_path)

    # os.system(f"mpg123 {switching_sound_path}")
    # from pydub import AudioSegment
    # from pydub.playback import play

    # song = AudioSegment.from_mp3(switching_sound_path)
    # play(song)

    # node.get_logger().warn("Played song")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # set socket to be non-blocking
        s.setblocking(False)
        s.connect((host, port))

        node.get_logger().info(f"Listening on {host}:{port}")

        while rclpy.ok():
            # sleep until it is time to listen to socket again
            rate.sleep()

            tcp_data = s.recv(8)
            if not tcp_data:
                continue

            # node.get_logger().info(f"Received raw tcp data: {tcp_data}")

            # Decode the received data
            stimulation_type = decode_stimulation(tcp_data)
            node.get_logger().info(
                f"Decoded msg to stimulation type: {stimulation_type}"
            )

            # update history
            t_hs = np.roll(t_hs, -1)
            st_hs = np.roll(st_hs, -1)
            sw_hs = np.roll(sw_hs, -1)
            t_hs[-1] = node.get_clock().now().to_msg().sec
            st_hs[-1] = stimulation_type
            sw_hs[-1] = 0

            if joy_control_mode == "bending":
                joy_signal = [0.0]
                # map the stimulation type to the joy signal {-1, 0, 1}
                if stimulation_type == 16:
                    # no stimulation / effect
                    joy_signal = [0.0]
                elif stimulation_type == 1:
                    # bending to the left
                    joy_signal = [1.0]
                elif stimulation_type == 2:
                    # bending to the right
                    joy_signal = [-1.0]
                else:
                    node.get_logger().error(
                        f"Unknown stimulation type: {stimulation_type}"
                    )
                    continue
            elif joy_control_mode == "cartesian":
                joy_signal = [0.0, 0.0]
                # map the stimulation type to the joy signal
                if stimulation_type == 16:
                    # no stimulation / effect
                    joy_signal = [0.0, 0.0]
                elif stimulation_type == 1:
                    # move to the left
                    joy_signal = [-1.0, 0.0]
                elif stimulation_type == 2:
                    # move to the right
                    joy_signal = [1.0, 0.0]
                elif stimulation_type == 12:
                    # move up
                    joy_signal = [0.0, 1.0]
                elif stimulation_type == 6:
                    # move down
                    joy_signal = [0.0, -1.0]
                else:
                    node.get_logger().error(
                        f"Unknown stimulation type: {stimulation_type}"
                    )
                    continue
            elif joy_control_mode == "cartesian_switch":
                # publish message with current Cartesian switch state
                cartesian_switch_state_pub.publish(Int32(data=active_axis))

                joy_signal = [0.0 for _ in range(node.num_axes)]
                switch_stimulation = 12
                # if the buffer is not full yet, we don't want to switch
                if (t_hs != 0.0).sum() == lhs:
                    # check if we should switch the active axis
                    if (
                        # we have enough switch stimulations to actually switch
                        ((st_hs == switch_stimulation).sum() / lhs >= node.accuracy_for_switch) and
                        (sw_hs == 1).sum() == 0 # we didn't switch within the available history
                    ):
                        # switch the active axis
                        active_axis = (active_axis + 1) % node.num_axes
                        sw_hs[-1] = 1
                        # play a sound to indicate that we switched
                        # playsound(str(
                        #         assets_dir
                        #         / "relax-message-tone.mp3"
                        # ))
                        node.get_logger().warn(f"Switched active axis to {active_axis}") 

                # map the stimulation type to the joy signal
                if stimulation_type == 16:
                    # no stimulation / effect
                    pass
                elif stimulation_type == 1:
                    # move negative (i.e. left or down)
                    joy_signal[active_axis] = -1.0
                elif stimulation_type == 2:
                    # move positive (i.e. right or up)
                    joy_signal[active_axis] = 1.0
                elif stimulation_type == switch_stimulation:
                    # we don't want to switch the active axis yet
                    pass
                else:
                    node.get_logger().error(
                        f"Unknown stimulation type: {stimulation_type}"
                    )
                    continue
            else:
                raise ValueError(f"Unknown joy control mode: {joy_control_mode}")

            # publish Joy msg
            msg = Joy(axes=joy_signal)
            msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            # node.get_logger().info(f"Published msg: {msg}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
