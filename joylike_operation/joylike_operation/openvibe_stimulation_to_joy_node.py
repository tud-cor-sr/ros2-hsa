#!/usr/bin/env python3

import socket
import rclpy
from sensor_msgs.msg import Joy


def decode_stimulation(byte_data):
    # Decode the first byte to determine the stimulation type
    stimulation_type = int(byte_data[0])

    # Return the decoded stimulation type
    return stimulation_type


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("stimulation_receiver_node")

    node.declare_parameter("joy_control_mode", "cartesian")
    joy_control_mode = node.get_parameter(
        "joy_control_mode"
    ).value  # bending, cartesian, or cartesian_switch

    node.declare_parameter("joy_signal_topic", "joy_signal")
    joy_signal_topic = node.get_parameter("joy_signal_topic").value
    pub = node.create_publisher(Joy, joy_signal_topic, rclpy.qos.qos_profile_system_default)

    node.declare_parameter("host", "localhost")
    host = node.get_parameter("host").value
    node.declare_parameter("port", 5678)
    port = node.get_parameter("port").value

    # initialize activate direction as the y-axis
    # this only applies to the joy_control_mode == "cartesian_switch"
    active_dir = 1

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        node.get_logger().info(f"Listening on {host}:{port}")

        while rclpy.ok():
            tcp_data = s.recv(8)
            if not tcp_data:
                break

            # node.get_logger().info(f"Received raw tcp data: {tcp_data}")

            # Decode the received data
            stimulation_type = decode_stimulation(tcp_data)
            node.get_logger().info(
                f"Decoded msg to stimulation type: {stimulation_type}"
            )

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
                joy_signal = [0.0, 0.0]
                # map the stimulation type to the joy signal
                if stimulation_type == 16:
                    # no stimulation / effect
                    joy_signal = [0.0, 0.0]
                elif stimulation_type == 1:
                    # move negative (i.e. left or down)
                    joy_signal[active_dir] = -1.0
                elif stimulation_type == 2:
                    # move positive (i.e. right or up)
                    joy_signal[active_dir] = 1.0
                elif stimulation_type == 10:
                    # switch direction
                    active_dir = (active_dir + 1) % 2
                else:
                    node.get_logger().error(
                        f"Unknown stimulation type: {stimulation_type}"
                    )
                    continue
            else:
                raise ValueError(f"Unknown joy control mode: {joy_control_mode}")

            # Create an instance of your custom message
            # Assign received data to the message field
            msg = Joy(axes=joy_signal)
            msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            # node.get_logger().info(f"Published msg: {msg}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
