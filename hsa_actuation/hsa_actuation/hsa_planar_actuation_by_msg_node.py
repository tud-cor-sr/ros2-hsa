from jax import Array
import jax.numpy as jnp
import rclpy

from example_interfaces.msg import Float64MultiArray
from dynamixel_control_custom_interfaces.msg import SetPosition
from dynamixel_control_custom_interfaces.srv import GetPositions
from hsa_actuation.hsa_planar_actuation_base_node import HsaPlanarActuationBaseNode


class HsaPlanarActuationByMsgNode(HsaPlanarActuationBaseNode):
    def __init__(self):
        super().__init__(
            node_name="hsa_planar_actuation_by_msg_node",
            post_present_motor_angles_receival_callback=self.map_motor_angles_to_actuation_coordinates
        )

        # create the publisher of the current actuation
        self.phi = jnp.zeros((self.control_handedness.shape[0], ))
        self.declare_parameter("present_planar_actuation_topic", "present_planar_actuation")
        self.present_planar_actuation_pub = self.create_publisher(
            Float64MultiArray,
            self.get_parameter("present_planar_actuation_topic").value,
            10,
        )

        # create the subscription to the control input
        self.declare_parameter("control_input_topic", "control_input")
        self.phi_sub = self.create_subscription(
            Float64MultiArray,
            self.get_parameter("control_input_topic").value,
            self.phi_des_callback,
            10,
        )

    def present_motor_angles_callback(self, motor_angles: Array):
        # map motor angles to actuation coordinates
        self.phi = self.map_motor_angles_to_actuation_coordinates(self.present_motor_angles)

        # publish the current actuation
        phi_msg = Float64MultiArray(data=self.phi.tolist())
        self.present_planar_actuation_pub.publish(phi_msg)

    def phi_des_callback(self, msg: Float64MultiArray):
        # demanded rod twist angles
        phi_des = jnp.array(msg.data)

        # map the actuation coordinates to motor angles
        motor_goal_angles = self.map_actuation_coordinates_to_motor_angles(phi_des)

        # send motor goal angles to dynamixel motors
        self.set_motor_goal_angles(motor_goal_angles)


def main(args=None):
    rclpy.init(args=args)
    print("Hi from hsa planar actuation by msg by msg node.")

    node = HsaPlanarActuationByMsgNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
