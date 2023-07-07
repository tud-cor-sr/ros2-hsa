import numpy as np
import rclpy

from example_interfaces.msg import Float64MultiArray
from dynamixel_control_custom_interfaces.msg import SetPosition
from dynamixel_control_custom_interfaces.srv import GetPositions
from hsa_actuation.hsa_actuation_base_node import HsaActuationBaseNode


class PresentMotorStatePubNode(HsaActuationBaseNode):
    def __init__(self):
        super().__init__("presenet_motor_state_pub_node")


def main(args=None):
    rclpy.init(args=args)
    print("Hi from present motor state publisher node.")

    node = PresentMotorStatePubNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
