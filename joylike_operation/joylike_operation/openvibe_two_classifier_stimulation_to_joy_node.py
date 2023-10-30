#!/usr/bin/env python3
from ament_index_python.packages import get_package_share_directory
import numpy as np
import os
import rclpy
from pathlib import Path
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32
import socket
import threading


def decode_stimulation(byte_data):
    # Decode the first byte to determine the stimulation type
    stimulation_type = int(byte_data[0])

    # Return the decoded stimulation type
    return stimulation_type


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("stimulation_receiver_node")

    node.declare_parameter("joy_control_mode", "cartesian_switch")
    joy_control_mode = node.get_parameter(
        "joy_control_mode"
    ).value  # cartesian_switch
    assert joy_control_mode == "cartesian_switch", "This node only supports cartesian_switch mode"

    node.declare_parameter("joy_signal_topic", "joy_signal")
    joy_signal_topic = node.get_parameter("joy_signal_topic").value
    pub = node.create_publisher(Joy, joy_signal_topic, rclpy.qos.qos_profile_system_default)

    node.declare_parameter("num_axes", 2)
    node.num_axes = node.get_parameter("num_axes").value

    # initialize activate direction as the x-axis
    active_axis = 0
    cartesian_switch_state_pub = node.create_publisher(
        Int32, "cartesian_switch_state", rclpy.qos.qos_profile_system_default
    )

    node.declare_parameter("host", "localhost")
    host = node.get_parameter("host").value
    node.declare_parameter("port1", 5678)
    node.declare_parameter("port2", 5679)
    port1 = node.get_parameter("port1").value  # port for the first classifier (switching)
    port2 = node.get_parameter("port2").value  # port for the second classifier (positive / negative)
    assert port1 != port2, "Port 1 and port 2 must be different"

    # define frequency of the main loop
    node.declare_parameter("frequency", 100.0)
    freq = node.get_parameter("frequency").value
    rate = node.create_rate(freq)

    # history of samples
    lhs = 35
    t_hs = np.zeros((lhs, ))  # time associated with samples
    st1_hs = np.zeros((lhs, ))  # history of stimulations from classifier 1
    st2_hs = np.zeros((lhs, ))  # history of stimulations from classifier 2
    sw_hs = np.zeros((lhs, ))  # history of switches

    # at least 80% of samples over the specified duration need to tell us to switch
    node.declare_parameter("accuracy_for_switch", 0.8)
    node.accuracy_for_switch = node.get_parameter("accuracy_for_switch").value

    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2,
    ):
        # connect
        s1.connect((host, port1))
        s2.connect((host, port2))

        node.get_logger().info(f"Connected to {host}:{port1} and {host}:{port2}")

        # set socket to be non-blocking
        s1.setblocking(False)
        s2.setblocking(False)

        # Spin in a separate thread
        thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
        thread.start()

        try:
            while rclpy.ok():
                # sleep until it is time to listen to socket again
                rate.sleep()

                try:
                    tcp_data1 = s1.recv(8)
                    tcp_data2 = s2.recv(8)
                except BlockingIOError:
                    # no data available
                    continue
                if not tcp_data1 or not tcp_data2:
                    # we require data from both classifiers to be available
                    continue

                # Decode the received data
                stim_type1 = decode_stimulation(tcp_data1)
                stim_type2 = decode_stimulation(tcp_data2)
                node.get_logger().info(
                    f"Decoded msg to stimulation 1: {stim_type1}, and stimulation 2: {stim_type2}"
                )

                # update history
                t_hs = np.roll(t_hs, -1)
                st1_hs = np.roll(st1_hs, -1)
                st2_hs = np.roll(st2_hs, -1)
                sw_hs = np.roll(sw_hs, -1)
                t_hs[-1] = node.get_clock().now().to_msg().sec
                st1_hs[-1] = stim_type1
                st2_hs[-1] = stim_type2
                sw_hs[-1] = 0
                
                # publish message with current Cartesian switch state
                cartesian_switch_state_pub.publish(Int32(data=active_axis))

                joy_signal = [0.0 for _ in range(node.num_axes)]
                switch_stimulation = 12
                # if the buffer is not full yet, we don't want to switch
                if (t_hs != 0.0).sum() == lhs:
                    # check if we should switch the active axis
                    if (
                        # we have enough "wake" stimuluations from classifier 1 to actually switch
                        ((st1_hs == switch_stimulation).sum() / lhs >= node.accuracy_for_switch) and
                        (sw_hs == 1).sum() == 0 # we didn't switch within the available history
                    ):
                        # switch the active axis
                        active_axis = (active_axis + 1) % node.num_axes
                        sw_hs[-1] = 1
                        node.get_logger().info(f"Switched active axis to {active_axis}") 

                # map the stimulation type to the joy signal
                if stim_type1 == switch_stimulation:
                    # we received a switch stimulation, so we don't want to evaluate the second classifier
                    pass
                else:
                    if stim_type2 == 1:
                        # move negative (i.e. left or down)
                        joy_signal[active_axis] = -1.0
                    elif stim_type2 == 2:
                        # move positive (i.e. right or up)
                        joy_signal[active_axis] = 1.0
                    else:
                        node.get_logger().warn(
                            f"Unknown stimulation type for 2nd classifier: {stim_type2}"
                        )
                        continue

                # publish Joy msg
                msg = Joy(axes=joy_signal)
                msg.header.stamp = node.get_clock().now().to_msg()
                pub.publish(msg)
                # node.get_logger().info(f"Published msg: {msg}")

        except KeyboardInterrupt:
           pass

        node.destroy_node()
        rclpy.shutdown()
        thread.join()


if __name__ == "__main__":
    main()
