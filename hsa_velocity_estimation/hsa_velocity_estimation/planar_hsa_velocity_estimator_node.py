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
from typing import Tuple

from geometry_msgs.msg import Pose2D
from hsa_control_interfaces.msg import (
    Pose2DStamped,
)
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration


@jit
def update_history_array(x_hs: Array, x: Array) -> Array:
    # x_hs = jnp.roll(x_hs, shift=-1, axis=0)
    # x_ts = x_hs.at[-1].set(x)  # this operation seems to be slow
    x_ts = jnp.concatenate((x_hs[1:], jnp.expand_dims(x, axis=0)), axis=0)
    return x_ts


class PlanarHsaVelocityEstimatorNode(Node):
    def __init__(self):
        super().__init__("planar_hsa_velocity_estimator_node")

        # initialize velocity publishers
        self.declare_parameter("configuration_velocity_topic", "configuration_velocity")
        self.q_d_pub = self.create_publisher(
            PlanarCsConfiguration,
            self.get_parameter("configuration_velocity_topic").value,
            10
        )
        self.declare_parameter("end_effector_velocity_topic", "end_effector_velocity")
        self.chiee_d_pub = self.create_publisher(
            Pose2DStamped,
            self.get_parameter("end_effector_velocity_topic").value,
            10
        )

        # initialize configuration and end-effector pose velocities
        self.q_d, self.chiee_d = None, None

        # method for computing derivative
        self.declare_parameter("num_derivative_method", "numpy_gradient")
        self.num_derivative_method = self.get_parameter("num_derivative_method").value

        self.lhs4d = 4  # History length for numerical differentiation
        if self.num_derivative_method == "numpy_gradient":
            self.lhs4d = 4
            self.num_derivative_fn = jit(partial(jnp.gradient, axis=0))
        elif self.num_derivative_method == "derivative_savitzky_golay":
            self.lhs4d = 30
            # we are only interested in the last (i.e., most recent) derivative
            self.num_derivative_fn = partial(
                derivative.SavitzkyGolay(left=0.1, right=0.0, order=3).compute,
                i=-1
            )
        elif self.num_derivative_method == "derivative_spline":
            self.lhs4d = 16
            # we are only interested in the last (i.e., most recent) derivative
            self.num_derivative_fn = partial(
                derivative.Spline(s=1.0, order=3).compute,
                i=-1
            )
        else:
            raise ValueError(f"Unknown num_derivative_method: {self.num_derivative_method}")
        
        self.tq_hs = jnp.zeros((self.lhs4d,))
        self.tchiee_hs = jnp.zeros((self.lhs4d,))
        self.q_hs, self.chiee_hs = None, None

        # initialize listeners for configuration and end-effector pose
        self.declare_parameter("configuration_topic", "configuration")
        self.configuration_sub = self.create_subscription(
            PlanarCsConfiguration,
            self.get_parameter("configuration_topic").value,
            self.configuration_listener_callback,
            10,
        )

        self.declare_parameter("end_effector_pose_topic", "end_effector_pose")
        self.end_effector_pose_sub = self.create_subscription(
            Pose2DStamped,
            self.get_parameter("end_effector_pose_topic").value,
            self.end_effector_pose_listener_callback,
            10,
        )

        # timer for publishing the velocity messages
        self.declare_parameter("frequency", 200.0)
        self.create_timer(
            1.0 / self.get_parameter("frequency").value, self.timer_callback
        )

    def configuration_listener_callback(self, msg: PlanarCsConfiguration):
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9

        # set the current configuration
        q = jnp.array([msg.kappa_b, msg.sigma_sh, msg.sigma_a])

        if self.q_d is None:
            self.q_d = jnp.zeros_like(q)

        if self.q_hs is None:
            self.q_hs = jnp.zeros((self.lhs4d, q.shape[0]))

        # update history
        self.tq_hs = update_history_array(self.tq_hs, t)
        self.q_hs = update_history_array(self.q_hs, q)

    def end_effector_pose_listener_callback(self, msg: Pose2DStamped):
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9

        # set the current end-effector pose
        chiee = jnp.array([msg.pose.x, msg.pose.y, msg.pose.theta])

        if self.chiee_d is None:
            self.chiee_d = jnp.zeros_like(chiee)

        if self.chiee_hs is None:
            self.chiee_hs = jnp.zeros((self.lhs4d, chiee.shape[0]))

        # update history
        self.tchiee_hs = update_history_array(self.tchiee_hs, t)
        self.chiee_hs = update_history_array(self.chiee_hs, chiee)

    def compute_q_d(self) -> Tuple[float, Array]:
        """
        Compute the velocity of the generalized coordinates from the history of configurations.
        Returns:
            t: the time of the last configuration measurement
            q_d: velocity of the generalized coordinates
        """
        # if the buffer is not full yet, return the current velocity
        if jnp.any(self.tq_hs == 0.0):
            return 0.0, self.q_d

        # subtract the first time stamp from all time stamps to avoid numerical issues
        t_hs = jnp.repeat(jnp.expand_dims(self.tq_hs - self.tq_hs[0], axis=-1), self.q_hs.shape[-1], axis=-1)

        if self.num_derivative_method == "numpy_gradient":
            # we assume a constant time step
            dt = jnp.mean(t_hs[1:] - t_hs[:-1])
            q_d_hs = self.num_derivative_fn(self.q_hs, dt)
        elif self.num_derivative_method in ["derivative_finite_differences", "derivative_savitzky_golay", "derivative_spline"]:
            # iterate through configuration variables
            q_d_hs = []
            for i in range(self.q_hs.shape[-1]):
                # derivative of all time stamps for configuration variable i
                q_d_hs.append(self.num_derivative_fn(self.q_hs[:, i], t_hs[:, i]))
            q_d_hs = jnp.stack(q_d_hs, axis=0)
        else:
            q_d_hs = self.num_derivative_fn(self.q_hs, t_hs)

        q_d = q_d_hs[-1]

        return self.tq_hs[-1], q_d

    def compute_chiee_d(self) -> Tuple[float, Array]:
        """
        Compute the velocity of the end-effector pose from the history of end-effector poses.
        Returns:
            t: the time of the last configuration measurement
            chiee_d: velocity of the end-effector pose
        """
        # if the buffer is not full yet, return the current velocity
        if jnp.any(self.tchiee_hs == 0.0):
            return 0.0, self.chiee_d

        # subtract the first time stamp from all time stamps to avoid numerical issues
        t_hs = jnp.repeat(jnp.expand_dims(self.tchiee_hs - self.tchiee_hs[0], axis=-1), self.chiee_hs.shape[-1], axis=-1)

        if self.num_derivative_method == "numpy_gradient":
            # we assume a constant time step
            dt = jnp.mean(t_hs[1:] - t_hs[:-1])
            chiee_d_hs = self.num_derivative_fn(self.chiee_hs, dt)
            chiee_d = chiee_d_hs[-1]
        elif self.num_derivative_method in ["derivative_finite_differences", "derivative_savitzky_golay", "derivative_spline"]:
            # iterate through configuration variables
            chiee_d = []
            for chiee_idx in range(self.chiee_hs.shape[-1]):
                # derivative for the last (i.e., most recent) time step
                chiee_d.append(self.num_derivative_fn(self.chiee_hs[:, chiee_idx], t_hs[:, chiee_idx]))
            chiee_d = jnp.stack(chiee_d, axis=0)
        else:
            chiee_d_hs = self.num_derivative_fn(self.chiee_hs, t_hs)
            chiee_d = chiee_d_hs[-1]

        return self.tchiee_hs[-1], chiee_d
    
    def timer_callback(self):
        tq, self.q_d = self.compute_q_d()
        tchiee, self.chiee_d = self.compute_chiee_d()

        # if there is no data available, do not publish anything
        if self.q_d is None or self.chiee_d is None:
            return

        # publish the velocity of the generalized coordinates
        msg = PlanarCsConfiguration()
        msg.header.stamp = Time(seconds=tq).to_msg()
        msg.kappa_b = self.q_d[0].item()
        msg.sigma_sh = self.q_d[1].item()
        msg.sigma_a = self.q_d[2].item()
        self.q_d_pub.publish(msg)

        # publish the velocity of the end-effector pose
        msg = Pose2DStamped()
        msg.header.stamp = Time(seconds=tchiee).to_msg()
        msg.pose.x = self.chiee_d[0].item()
        msg.pose.y = self.chiee_d[1].item()
        msg.pose.theta = self.chiee_d[2].item()
        self.chiee_d_pub.publish(msg)

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
