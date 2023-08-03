import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, jit
from jax import numpy as jnp
import rclpy
from rclpy.node import Node
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from example_interfaces.msg import Float64MultiArray
from geometry_msgs.msg import Pose2D
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration

import jsrm
from jsrm.parameters.hsa_params import PARAMS_CONTROL
from jsrm.systems import planar_hsa


class PlanarSimNode(Node):
    def __init__(self):
        super().__init__("planar_sim_node")

        # filepath to symbolic expressions
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_hsa_ns-1_nrs-2.dill"
        )
        # call factory for the planar HSA kinematics and dynamics
        (
            _,
            _,
            _,
            _,
            dynamical_matrices_fn,
            sys_helpers,
        ) = planar_hsa.factory(sym_exp_filepath)

        self.params = PARAMS_CONTROL

        # parameter for specifying a different axial rest strain
        self.declare_parameter("sigma_a_eq", self.params["sigma_a_eq"].mean().item())
        sigma_a_eq = self.get_parameter("sigma_a_eq").value
        self.params["sigma_a_eq"] = sigma_a_eq * jnp.ones_like(self.params["sigma_a_eq"])

        # initialize state
        self.q = jnp.zeros_like(self.xi_eq)  # generalized coordinates
        self.n_q = self.q.shape[0]  # number of generalized coordinates
        self.q_d = jnp.zeros_like(self.q)  # velocity of generalized coordinates

        # initialize ODE solver
        self.declare_parameter("sim_dt", 1e-3)
        self.sim_dt = jnp.array(self.get_parameter("sim_dt").value)
        self.ode_fn = jit(planar_hsa.ode_factory(dynamical_matrices_fn, self.params))
        self.declare_parameter("ode_solver_class", "Euler")
        self.ode_solver = getattr(diffrax, self.get_parameter("ode_solver_class").value)()
        ode_term = ODETerm(partial(ode_fn, u=phi))

        # jit the ode fn
        x_dummy = jnp.zeros((2 * self.n_q,))
        phi_dummy = jnp.zeros_like(self.params["roff"].flatten())
        print("phi dummy", phi_dummy)
        x_d_dummy = ode_fn(0.0, x_dummy, u=phi_dummy)

        # initialize time
        self.clock_start_time = self.get_clock().now().nanoseconds * 1e-9  # time in seconds
        self.clock_time = self.clock_start_time

        # create a publisher for the configuration and its velocity
        self.declare_parameter("configuration_topic", "configuration")
        self.configuration_pub = self.create_publisher(
            PlanarCsConfiguration, self.get_parameter("configuration_topic").value, 10
        )
        self.declare_parameter("configuration_velocity_topic", "configuration_velocity")
        self.configuration_velocity_pub = self.create_publisher(
            Float64MultiArray,
            self.get_parameter("configuration_velocity_topic").value,
            10,
        )

        # create the subscription to the control input
        self.declare_parameter(
            "control_input_topic", "control_input"
        )
        self.phi_sub = self.create_subscription(
            Float64MultiArray,
            self.get_parameter("control_input_topic").value,
            self.phi_callback,
            10,
        )

    def phi_callback(self, msg: Float64MultiArray):
        # demanded rod twist angles
        phi = jnp.array(msg.data)

        # the current clock time
        clock_current_time = self.get_clock().now().nanoseconds * 1e-9
        # compute the relative time to the start of the simulation
        t0 = self.clock_time - self.clock_start_time
        t1 = clock_current_time - self.clock_start_time

        # current state of the system
        x0 = jnp.concatenate((self.q, self.q_d))

        # simulate the system
        ode_term = ODETerm(partial(ode_fn, u=phi))
        sol = diffeqsolve(
            ode_term,
            solver=self.ode_solver,
            t0=jnp.array(t0),
            t1=jnp.array(t1),
            dt0=self.sim_dt,
            y0=x0,
            max_steps=None,
            # saveat=SaveAt(ts=video_ts),
        )

        # update the state of the system
        x1 = sol.y[-1, :]  # final state of the simulation
        self.q, self.q_d = x1[:self.n_q], x1[self.n_q:]

        # publish the configuration
        configuration_msg = PlanarCsConfiguration(
            kappa_b=self.q.tolist(),
            sigma_sh=self.q_d.tolist(),
            sigma_a=self.params["sigma_a_eq"].tolist(),
        )
        configuration_msg.header.stamp = self.get_clock().now().to_msg()
        self.configuration_pub.publish(configuration_msg)

        # publish configuration velocity
        configuration_velocity_msg = Float64MultiArray(
            data=self.q_d.tolist()
        )
        self.configuration_velocity_pub.publish(configuration_velocity_msg)

        # update the clock time
        self.clock_time = clock_current_time


def main():
    rclpy.init(args=args)
    print("Hi from planar_sim_node.")

    node = PlanarSimNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
