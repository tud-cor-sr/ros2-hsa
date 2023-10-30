#!/usr/bin/env python3

import abc
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy import qos
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from pathlib import Path

from keyboard_msgs.msg import Key
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32

import yaml


class JoyPart(abc.ABC):
    def __init__(self, init_value):
        self._value = init_value

    @abc.abstractmethod
    def down(self, code):
        pass

    @abc.abstractmethod
    def up(self, code):
        pass

    def get(self):
        return self._value


class Button(JoyPart):
    def __init__(self, key_str):
        super().__init__(0)
        self.code = getattr(Key, key_str)

    def down(self, code) -> bool:
        if code == self.code:
            self._value = 1
            return True
        return False

    def up(self, code) -> bool:
        if code == self.code:
            self._value = 0
            return True
        return False


class Axis(JoyPart):
    def __init__(self, key_neg_str, key_pos_str):
        super().__init__(0.0)
        self.code_neg = getattr(Key, key_neg_str)
        self.code_pos = getattr(Key, key_pos_str)

    def down(self, code):
        if code == self.code_neg:
            self._value -= 1.0

        elif code == self.code_pos:
            self._value += 1.0

    def up(self, code):
        if code == self.code_neg:
            self._value += 1.0

        elif code == self.code_pos:
            self._value -= 1.0


class KeyboardToJoyNode(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__("keyboard_to_joy_node")

        # Get parameters
        self.declare_parameter("joy_control_mode", "cartesian")
        self.joy_control_mode = self.get_parameter("joy_control_mode").value
        self.declare_parameter("config_filepath", "None")
        config_filepath = (
            self.get_parameter("config_filepath").get_parameter_value().string_value
        )
        if config_filepath == "None":
            config_filepath = (
                Path(get_package_share_directory("hsa_joy_control"))
                / "config"
                / "keystroke2joy_cartesian.yaml"
            )

        self.declare_parameter("sampling_frequency", 50)
        sampling_frequency = (
            self.get_parameter("sampling_frequency").get_parameter_value().integer_value
        )

        # Load config file
        with open(config_filepath, "rb") as configfile:
            self.config = yaml.load(configfile, Loader=yaml.FullLoader)

        self.buttons = [Button(key_str) for key_str in self.config.get("buttons", [])]
        self.axes = [
            Axis(key_neg_str, key_pos_str)
            for key_neg_str, key_pos_str in self.config.get("axes", [])
        ]

        # Setup publisher
        self.joy = Joy()
        self.declare_parameter("joy_signal_topic", "joy_signal")
        joy_signal_topic = self.get_parameter("joy_signal_topic").value
        self.joy_pub = self.create_publisher(
            Joy, joy_signal_topic, qos.qos_profile_system_default
        )

        # Keyboard callback
        self.keydown_sub = self.create_subscription(
            Key, "keydown", self.keydown_callback, qos.qos_profile_system_default
        )
        self.keyup_sub = self.create_subscription(
            Key, "keyup", self.keyup_callback, qos.qos_profile_system_default
        )

        if self.joy_control_mode == "cartesian_switch":
            self.declare_parameter("num_axes", 2)
            self.num_axes = self.get_parameter("num_axes").value
            
            # initialize activate direction as the x-axis
            self.active_axis = 0
            self.cartesian_switch_state_pub = self.create_publisher(
                Int32, "cartesian_switch_state", qos.qos_profile_system_default
            )
        else:
            self.num_axes = len(self.axes)

        # Start timer
        dt = 1.0 / float(sampling_frequency)
        self.create_timer(dt, self.main_loop)

    def keydown_callback(self, msg):
        for ax in self.axes:
            ax.down(msg.code)
        for but_idx, but in enumerate(self.buttons):
            is_down_event_active = but.down(msg.code)
            if self.joy_control_mode == "cartesian_switch" and but_idx == 0 and is_down_event_active:
                # switch the active axis when the first button is pressed
                self.active_axis = (self.active_axis + 1) % self.num_axes

    def keyup_callback(self, msg):
        for ax in self.axes:
            ax.up(msg.code)
        for but in self.buttons:
            but.up(msg.code)

    def main_loop(self):
        if self.joy_control_mode == "cartesian_switch":
            # publish message with current Cartesian switch state
            self.cartesian_switch_state_pub.publish(Int32(data=self.active_axis))

            joy_signal = []
            # we just consider the first keyboard axis as an input source
            ax = self.axes[0]
            for axis_idx in range(self.num_axes):
                joy_signal.append(ax.get() if axis_idx == self.active_axis else 0.0)
            msg = Joy(
                axes=joy_signal, buttons=[b.get() for b in self.buttons]
            )
        else:
            msg = Joy(
                axes=[a.get() for a in self.axes], buttons=[b.get() for b in self.buttons]
            )

        # publish Joy msg
        msg.header.stamp = self.get_clock().now().to_msg()
        self.joy_pub.publish(msg)

def main(args=None):
    # Start node, and spin
    rclpy.init(args=args)
    node = KeyboardToJoyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up and shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
