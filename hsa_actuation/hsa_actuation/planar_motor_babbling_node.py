import numpy as np
import rclpy
from rclpy.node import Node

from dynamixel_control_custom_interfaces.msg import SetPosition
from dynamixel_control_custom_interfaces.srv import GetPositions

from hsa_actuation.hsa_actuation_base_node import HsaActuationBaseNode


class PlanarMotorBabblingNode(HsaActuationBaseNode):
    def __init__(self):
        super().__init__("planar_kinematic_control_node")
        self.rod_handedness = np.array([-1.0, 1.0, -1.0, 1.0])

        self.node_frequency = 25  # Hz
        self.timer = self.create_timer(1.0 / self.node_frequency, self.timer_callback)
        self.time_idx = 0

        self.seed = 0
        self.mode = "gbn"
        self.duration = 45  # [s]
        self.dt = 1 / self.node_frequency
        self.phi_max = np.pi  # [deg]

        if self.mode == "gbn":
            from pygbn import gbn

            gbn_ts = 2  # mean settling time of process

            # flag indicating process damping properties
            # gbn_flag = 0 if the process is over-damped (default)
            # gbn_flag = 1 if the process is oscillary (min phase)
            # gbn_flag = 2 if the process is oscillary (non min phase)
            gbn_flag = 0

            # generate the signal
            # the gbn function returns a time array and a signal array
            self.u_ts = np.zeros((int(self.duration / self.dt), 2))
            for motor_idx in range(2):
                u_gbn = gbn(
                    self.dt,
                    self.duration,
                    1.0,
                    gbn_ts,
                    gbn_flag,
                    seed=self.seed + motor_idx,
                )
                self.u_ts[:, motor_idx] = (1 + u_gbn) / 2 * self.phi_max
        else:
            raise ValueError("Unknown mode.")

    def timer_callback(self, event=None):
        if self.time_idx >= self.u_ts.shape[0]:
            self.get_logger().info("Finished trajectory.")
            self.destroy_timer(self.timer)
            return

        phi = np.stack([
            self.u_ts[self.time_idx, 0] * self.rod_handedness[0],
            self.u_ts[self.time_idx, 1] * self.rod_handedness[1],
            self.u_ts[self.time_idx, 1] * self.rod_handedness[2],
            self.u_ts[self.time_idx, 0] * self.rod_handedness[3],
        ])

        self.set_motor_goal_angles(phi)

        self.time_idx += 1

def main(args=None):
    rclpy.init(args=args)
    print("Hi from hsa_motor_babbling.")

    node = PlanarMotorBabblingNode()

    rclpy.spin(node)    

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
