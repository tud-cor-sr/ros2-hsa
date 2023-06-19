import numpy as np
import rclpy
from rclpy.node import Node

from dynamixel_sdk_custom_interfaces.msg import SetPosition
from dynamixel_sdk_custom_interfaces.srv import GetPosition

class PlanarMotorBabblingNode(Node):
    def __init__(self):
        super().__init__("planar_kinematic_control_node")

        self.node_frequency = 100 # Hz

        self.seed = 0
        self.mode = "gbn"
        self.duration = 45  # [s]
        self.dt = 1 / self.node_frequency
        self.phi_max = np.pi / 2  # [deg]

        if self.mode == "gbn":
            from pygbn import gbn

            gbn_ts = 2 # mean settling time of process

            # flag indicating process damping properties
            # gbn_flag = 0 if the process is over-damped (default)
            # gbn_flag = 1 if the process is oscillary (min phase)
            # gbn_flag = 2 if the process is oscillary (non min phase)
            gbn_flag = 0

            # generate the signal
            # the gbn function returns a time array and a signal array
            u_ts = np.zeros((4, int(self.duration / self.dt)))
            for motor_idx in range(4):
                u_ts[motor_idx, :] = gbn(self.dt, self.duration, self.phi_max, gbn_ts, gbn_flag, seed=self.seed + motor_idx)

                print("u", u_ts[motor_idx, :])
        
        exit()

        self.motor_pos_cli = self.create_client(GetPosition, "/get_position")
        self.motor_pos_cli.wait_for_service()

        self.motor_ids = np.array([21, 22, 23, 24])
        self.motor_neutral_position = self.get_motor_positions()
        self.rod_handedness = np.array([-1., 1., -1., 1.])

        self.motor_target_pos_pub = self.create_publisher(SetPosition, "/set_position", 10)

        self.node_frequency = 100 # Hz
        self.timer = self.create_timer(1.0/self.node_frequency, self.timer_callback)

        self.seed = 0
        self.mode = "gbn"
        self.duration = 45  # [s]
        self.dt = 1 / self.node_frequency
        self.phi_max = np.pi / 2  # [deg]

        if self.mode == "gbn":
            from pygbn import gbn

            gbn_ts = 2 # mean settling time of process

            # flag indicating process damping properties
            # gbn_flag = 0 if the process is over-damped (default)
            # gbn_flag = 1 if the process is oscillary (min phase)
            # gbn_flag = 2 if the process is oscillary (non min phase)
            gbn_flag = 0

            # generate the signal
            # the gbn function returns a time array and a signal array
            u = gbn(self.dt, self.duration, self.phi_max, gbn_ts, gbn_flag, seed=self.seed)

            print("u", u)


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


def main(args=None):
    rclpy.init(args=args)
    print('Hi from hsa_motor_babbling.')

    node = PlanarMotorBabblingNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
