import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from dynamixel_control_custom_interfaces.msg import SetPosition
from dynamixel_control_custom_interfaces.srv import GetPositions


class HsaActuationBaseNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.declare_parameter("get_motor_positions_service_name", "/get_positions")
        self.declare_parameter("present_motor_angles_topic_name", "present_motor_angles")
        self.declare_parameter("goal_motor_angles_topic_name", "goal_motor_angles")

        self.motor_pos_cli = self.create_client(GetPositions, self.get_parameter("get_motor_positions_service_name"))
        self.motor_pos_cli.wait_for_service()

        self.motor_ids = np.array([21, 22, 23, 24], dtype=np.uint8)
        self.motor_neutral_positions = self.get_motor_positions()
        self.current_motor_goal_positions = np.copy(self.motor_neutral_positions)

        self.motor_goal_pos_publishers = {}
        for motor_idx, motor_id in enumerate(list(self.motor_ids)):
            self.motor_goal_pos_publishers[motor_id] = self.create_publisher(
                SetPosition, f"/set_position_motor_{motor_id}", 10
            )

        self.present_motor_angles_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("present_motor_angles_topic_name"), 10
        )
        self.goal_motor_angles_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("goal_motor_angles_topic_name"), 10
        )
        

    def get_motor_angles(self) -> np.ndarray:
        motor_positions = self.get_motor_positions()
        motor_angles = (motor_positions - self.motor_neutral_positions) / 2048 * np.pi

        self.present_motor_angles_pub.publish(Float64MultiArray(data=motor_angles))

        return motor_angles
    
    def set_motor_goal_angles(self, goal_angles: np.ndarray):
        motor_goal_positions = self.motor_neutral_positions + goal_angles / np.pi * 2048

        self.goal_motor_angles_pub.publish(Float64MultiArray(data=goal_angles))

        return self.set_motor_goal_positions(motor_goal_positions)

    def get_motor_positions(self) -> np.ndarray:
        req = GetPositions.Request()
        req.ids = self.motor_ids

        future = self.motor_pos_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        motor_positions = resp.positions

        return motor_positions

    def set_motor_goal_positions(self, goal_positions: np.ndarray) -> bool:
        for motor_idx, motor_id in enumerate(list(self.motor_ids)):
            motor_goal_position = goal_positions[motor_idx]
            if motor_goal_position != self.current_motor_goal_positions[motor_idx]:
                msg = SetPosition()
                msg.id = int(motor_id)
                msg.position = int(goal_positions[motor_idx].item())

                self.motor_goal_pos_publishers[motor_id].publish(msg)
                self.current_motor_goal_positions[motor_idx] = motor_goal_position

        return True


def main(args=None):
    rclpy.init(args=args)
    print("Hi from hsa actuation base node.")

    node = HsaActuationBaseNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
