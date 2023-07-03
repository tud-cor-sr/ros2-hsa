from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node


class PlanarCsIkNode(Node):
    def __init__(self):
        super().__init__("planar_cs_ik_node")
        self.get_logger().info("Hi from planar_cs_ik_node.")

        self.declare_parameter("tf_base_topic", "tf_base")
        self.declare_parameter("tf_platform_topic", "tf_platform")

        self.subscription = self.create_subscription(
            TransformStamped,
            self.get_parameter("tf_base_topic").value,
            self.listener_callback,
            10,
        )

        # transformation from base tf to the start of the proximal end of the metamaterial
        self.tf_fixed_base_offset = np.eye(4)
        # transformation from base tf to the distal end of the metamaterial to the tf of the platform
        self.tf_fixed_platform_offset = np.eye(4)

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg)


def main():
    print("Hi from hsa_ik.")


if __name__ == "__main__":
    main()
