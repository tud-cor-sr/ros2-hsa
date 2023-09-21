import numpy as np
import rclpy
from rclpy.node import Node

from example_interfaces.msg import Float64MultiArray
from dynamixel_control_custom_interfaces.msg import SetPosition
from dynamixel_control_custom_interfaces.srv import GetPositions


class HsaActuationBaseNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.declare_parameter("get_motor_positions_service_name", "/get_positions")
        self.declare_parameter(
            "present_motor_angles_topic_name", "present_motor_angles"
        )
        self.declare_parameter("goal_motor_angles_topic_name", "goal_motor_angles")

        self.motor_pos_cli = self.create_client(
            GetPositions, self.get_parameter("get_motor_positions_service_name").value
        )
        self.motor_pos_cli.wait_for_service()

        self.motor_ids = np.array([21, 22, 23, 24], dtype=np.int32)
        self.motor_neutral_positions = self.get_present_motor_positions()
        self.rod_handedness = np.array([1.0, -1.0, 1.0, -1.0])

        self.present_motor_positions = np.copy(self.motor_neutral_positions)
        self.current_motor_goal_positions = np.copy(self.motor_neutral_positions)

        self.present_motor_angles = np.zeros_like(
            self.motor_neutral_positions, dtype=np.float64
        )

        self.motor_goal_pos_publishers = {}
        for motor_idx, motor_id in enumerate(list(self.motor_ids)):
            self.motor_goal_pos_publishers[motor_id] = self.create_publisher(
                SetPosition, f"/set_position_motor_{motor_id}", 10
            )

        self.present_motor_angles_pub = self.create_publisher(
            Float64MultiArray,
            self.get_parameter("present_motor_angles_topic_name").value,
            10,
        )
        self.goal_motor_angles_pub = self.create_publisher(
            Float64MultiArray,
            self.get_parameter("goal_motor_angles_topic_name").value,
            10,
        )

        self.declare_parameter("present_motor_angles_frequency", 200.0)
        self.present_motor_angles_frequency = self.get_parameter(
            "present_motor_angles_frequency"
        ).value

        self.present_motor_angles_timer = self.create_timer(
            1.0 / self.present_motor_angles_frequency,
            self.get_present_motor_state_async,
        )

    def get_present_motor_angles(self) -> np.ndarray:
        motor_positions = self.get_present_motor_positions()

        self.present_motor_angles = (
            (motor_positions - self.motor_neutral_positions).astype(np.float64)
            / 2048
            * np.pi
        )

        self.present_motor_angles_pub.publish(
            Float64MultiArray(data=self.present_motor_angles)
        )

        return self.present_motor_angles

    def set_motor_goal_angles(self, goal_angles: np.ndarray):
        self.goal_motor_angles_pub.publish(Float64MultiArray(data=goal_angles))

        goal_motor_positions = self.motor_neutral_positions + (
            goal_angles / np.pi * 2048
        ).astype(np.int32)

        return self.set_goal_motor_positions(goal_motor_positions)

    def get_present_motor_positions(self) -> np.ndarray:
        """
        Returns the current motor positions as a numpy array. This is a blocking call.
        """
        req = GetPositions.Request()
        req.ids = self.motor_ids

        future = self.motor_pos_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=None)
        resp = future.result()

        if resp is None:
            self.get_logger().error(
                f"Service call for {self.get_parameter('get_motor_positions_service_name').value}. timed out."
            )
            return self.present_motor_angles

        motor_positions = np.array(resp.positions, dtype=np.int32)
        self.present_motor_positions = motor_positions

        return motor_positions

    def get_present_motor_state_async(self):
        """
        Sends a request for the current motor positions and angles. This is a non-blocking call.
        """
        req = GetPositions.Request()
        req.ids = self.motor_ids

        future = self.motor_pos_cli.call_async(req)
        future.add_done_callback(self._get_present_motor_state_callback)

    def _get_present_motor_state_callback(self, future) -> np.ndarray:
        """
        Callback when the request for the current motor state was completed.
        Saves present motor positions and angles as class attributes and publishes the angles.
        """
        resp = future.result()

        if resp is None:
            self.get_logger().error(
                f"Service request for {self.get_parameter('get_motor_positions_service_name').value} failed."
            )
            return self.present_motor_angles

        motor_positions = np.array(resp.positions, dtype=np.int32)
        self.present_motor_positions = motor_positions

        self.present_motor_angles = (
            (motor_positions - self.motor_neutral_positions).astype(np.float64)
            / 2048
            * np.pi
        )
        self.present_motor_angles_pub.publish(
            Float64MultiArray(data=self.present_motor_angles)
        )

        return motor_positions

    def set_goal_motor_positions(self, goal_positions: np.ndarray) -> bool:
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

    node = HsaActuationBaseNode("hsa_actuation_base_node")

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
