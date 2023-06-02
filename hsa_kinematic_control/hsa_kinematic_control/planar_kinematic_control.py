import numpy as np
import rclpy
from rclpy.node import Node

from dynamixel_sdk_custom_interfaces.srv import GetPosition

class PlanarKinematicControlNode(Node):
    def __init__(self):
        super().__init__("planar_kinematic_control_node")

        self.node_frequency = 100 # Hz
        self.timer = self.create_timer(1.0/self.node_frequency, self.timer_callback)

        self.motor_pos_cli = self.create_client(GetPosition, "/get_position")
        self.motor_pos_cli.wait_for_service()

        self.motor_ids = np.array([21, 22, 23, 24])
        self.motor_neutral_position = self.get_motor_positions()

    def timer_callback(self):
        self.get_logger().info("timer_callback")

    def get_motor_positions(self) -> np.ndarray:
        motor_positions = np.zeros(len(self.motor_ids))
        for motor_idx, motor_id in enumerate(list(self.motor_ids)):
            req = GetPosition.Request()
            req.id = int(motor_id)
            
            future = self.motor_pos_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            motor_positions[motor_idx] = resp.position
            
        return motor_positions


def main(args=None):
    rclpy.init(args=args)
    print('Hi from hsa_kinematic_control.')

    node = PlanarKinematicControlNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
