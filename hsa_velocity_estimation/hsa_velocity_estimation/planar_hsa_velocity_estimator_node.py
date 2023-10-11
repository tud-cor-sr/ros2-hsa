from copy import deepcopy
import derivative
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import jax
from jax import Array, jit
from jax import numpy as jnp
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from pathlib import Path

from geometry_msgs.msg import Pose2D
from hsa_control_interfaces.msg import (
    Pose2DStamped,
)
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration


class PlanarHsaVelocityEstimatorNode(Node):
    def __init__(self):
        super().__init__("planar_hsa_velocity_estimator_node")

        # history of configurations
        # the longer the history, the more delays we introduce, but the less noise we get
        self.declare_parameter("history_length_for_diff", 16)
        self.tq_hs = jnp.zeros((self.get_parameter("history_length_for_diff").value,))
        self.tchiee_hs = jnp.zeros(
            (self.get_parameter("history_length_for_diff").value,)
        )

        # method for computing derivative
        self.diff_method = derivative.Spline(s=1.0, k=3)

    def configuration_listener_callback(self, msg: PlanarCsConfiguration):
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9

        # set the current configuration
        q = jnp.array([msg.kappa_b, msg.sigma_sh, msg.sigma_a])

        if self.q_hs is None:
            self.q_hs = jnp.zeros(
                (self.get_parameter("history_length_for_diff").value, q.shape[0])
            )

        # update history
        self.tq_hs = jnp.roll(self.tq_hs, shift=-1, axis=0)
        self.tq_hs = self.tq_hs.at[-1].set(t)
        self.q_hs = jnp.roll(self.q_hs, shift=-1, axis=0)
        self.q_hs = self.q_hs.at[-1].set(q)

    def end_effector_pose_listener_callback(self, msg: Pose2DStamped):
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9

        # set the current end-effector pose
        chiee = jnp.array([msg.pose.x, msg.pose.y, msg.pose.theta])

        if self.chiee_hs is None:
            self.chiee_hs = jnp.zeros(
                (self.get_parameter("history_length_for_diff").value, chiee.shape[0])
            )

        # update history
        self.tchiee_hs = jnp.roll(self.tchiee_hs, shift=-1, axis=0)
        self.tchiee_hs = self.tchiee_hs.at[-1].set(t)
        self.chiee_hs = jnp.roll(self.chiee_hs, shift=-1, axis=0)
        self.chiee_hs = self.chiee_hs.at[-1].set(chiee)

    def compute_q_d(self) -> Array:
        """
        Compute the velocity of the generalized coordinates from the history of configurations.
        """
        # if the buffer is not full yet, return the current velocity
        if jnp.any(self.tq_hs == 0.0):
            return self.q_d

        # subtract the first time stamp from all time stamps to avoid numerical issues
        t_hs = self.tq_hs - self.tq_hs[0]

        q_d = jnp.zeros_like(self.q)
        # iterate through configuration variables
        for i in range(self.q_hs.shape[-1]):
            # derivative of all time stamps for configuration variable i
            q_d_hs = self.diff_method.d(self.q_hs[:, i], t_hs)

            q_d = q_d.at[i].set(q_d_hs[-1])

        return q_d

    def compute_chiee_d(self) -> Array:
        """
        Compute the velocity of the end-effector pose from the history of end-effector poses.
        """
        # if the buffer is not full yet, return the current velocity
        if jnp.any(self.tchiee_hs == 0.0):
            return self.chiee_d

        # subtract the first time stamp from all time stamps to avoid numerical issues
        tchiee_hs = self.tchiee_hs - self.tchiee_hs[0]

        chiee_d = jnp.zeros_like(self.chiee)
        # iterate through configuration variables
        for i in range(self.chiee_hs.shape[-1]):
            # derivative of all time stamps for configuration variable i
            chiee_d_hs = self.diff_method.d(self.chiee_hs[:, i], tchiee_hs)

            chiee_d = chiee_d.at[i].set(chiee_d_hs[-1])

        return chiee_d


def main(args=None):
    # Start node, and spin
    rclpy.init(args=args)
    node = PlanarHsaVelocityEstimatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up and shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
