import numpy as np
import rclpy
from rclpy.node import Node

from example_interfaces.msg import Float64MultiArray
from sensor_msgs.msg import Joy


class PlanarHsaBendingJoyControlNode(Node):
    def __init__(self):
        super().__init__("planar_hsa_bending_joy_control_node")

        # offset (neural) control input [rad]
        self.declare_parameter("phi_offset", np.pi / 2)
        self.phi_offset = self.get_parameter("phi_offset").value

        # change of phi at each time step in unit [rad]
        self.declare_parameter("phi_delta", np.pi / 25)
        self.phi_delta = self.get_parameter("phi_delta").value

        # maximum magnitude of control input [rad]
        self.declare_parameter("phi_max", np.pi)
        self.phi_max = self.get_parameter("phi_max").value

        # if the robot is platform-down, the coordinates are inverted and with that we also need to invert the joy signals
        self.declare_parameter("invert_joy_signals", True)
        self.invert_joy_signals = self.get_parameter("invert_joy_signals").value

        # publisher of control input
        self.declare_parameter("control_input_topic", "control_input")
        self.control_input_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("control_input_topic").value, 10
        )

        # initialize control input
        self.phi = self.phi_offset * np.ones((2,))
        # publish initial control input
        self.control_input_pub.publish(Float64MultiArray(data=self.phi.tolist()))

        self.declare_parameter("joy_signal_topic", "joy_signal")
        self.joy_signal_sub = self.create_subscription(
            Joy,
            self.get_parameter("joy_signal_topic").value,
            self.joy_signal_callback,
            10,
        )

    def joy_signal_callback(self, msg: Joy):
        joy_signal = np.array(msg.axes).item()
        self.get_logger().info("Received joy signal: %d" % joy_signal)

        # calculate control input
        if self.invert_joy_signals:
            self.phi = self.phi + self.phi_delta * joy_signal * np.array([1.0, -1.0])
        else:
            self.phi = self.phi + self.phi_delta * joy_signal * np.array([-1.0, 1.0])

        # saturate control input to [0.0, phi_max]
        self.phi = np.clip(
            self.phi, np.zeros_like(self.phi), self.phi_max * np.ones_like(self.phi)
        )

        # publish control input
        phi_msg = Float64MultiArray(data=self.phi.tolist())
        self.control_input_pub.publish(phi_msg)


def main(args=None):
    rclpy.init(args=args)
    print("Hi from planar_hsa_bending_joy_control_node.")

    node = PlanarHsaBendingJoyControlNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
