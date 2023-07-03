import numpy as np
import rclpy
from rclpy.node import Node

from dynamixel_control_custom_interfaces.msg import SetPosition
from dynamixel_control_custom_interfaces.srv import GetPosition


class PlanarKinematicControlNode(Node):
    def __init__(self):
        super().__init__("planar_kinematic_control_node")

        self.motor_pos_cli = self.create_client(GetPosition, "/get_position")
        self.motor_pos_cli.wait_for_service()

        self.motor_ids = np.array([21, 22, 23, 24])
        self.motor_neutral_position = self.get_motor_positions()

        self.motor_target_pos_pub = self.create_publisher(
            SetPosition, "/set_position", 10
        )

        self.dummy_time_idx = 0

        # PID control
        self.kp = np.array([1, 0, 1])  # control bending and length, but not shear
        self.ki = np.array([0, 0, 0])
        self.kd = np.array([0, 0, 0])
        self.integral = np.zeros(3)

        self.node_frequency = 100  # Hz
        self.timer = self.create_timer(1.0 / self.node_frequency, self.timer_callback)

    def timer_callback(self, event=None):
        # print("timer_callback", type(event), event)
        self.dummy_time_idx += 1
        if self.dummy_time_idx == 500:
            print("dummy_time_idx", self.dummy_time_idx)
            new_motor_positions = self.motor_neutral_position.copy()
            new_motor_positions[0] += 20
            print("New motor positions", new_motor_positions)
            print("Neutral motor positions", self.motor_neutral_position)
            self.set_motor_goal_positions(self.motor_neutral_position)

    def get_motor_positions(self) -> np.ndarray:
        motor_positions = np.zeros(len(self.motor_ids), dtype=np.uint32)
        for motor_idx, motor_id in enumerate(list(self.motor_ids)):
            req = GetPosition.Request()
            req.id = int(motor_id)

            future = self.motor_pos_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            motor_positions[motor_idx] = resp.position

        return motor_positions

    def set_motor_goal_positions(self, goal_positions: np.ndarray):
        for motor_idx, motor_id in enumerate(list(self.motor_ids)):
            msg = SetPosition()
            msg.id = int(motor_id)
            msg.position = int(goal_positions[motor_idx].item())

            self.motor_target_pos_pub.publish(msg)

        return

    def evaluate_pid(self, q: np.ndarray, q_d: np.ndarray = None) -> np.ndarray:
        if q_d is None:
            q_d = np.zeros_like(q)

        e = q_d - q
        u = self.kp * e + self.ki * self.integral + self.kd * (e - self.prev_error)

        self.integral += e
        self.prev_error = e
        return u


def main(args=None):
    rclpy.init(args=args)
    print("Hi from hsa_kinematic_control.")

    node = PlanarKinematicControlNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
