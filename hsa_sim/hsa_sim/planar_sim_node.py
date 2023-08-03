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
        # actual rest strain
        self.xi_eq = sys_helpers["rest_strains_fn"](self.params)  # rest strains

        # initialize state and control input
        self.q = jnp.zeros_like(self.xi_eq)  # generalized coordinates
        self.n_q = self.q.shape[0]  # number of generalized coordinates
        self.q_d = jnp.zeros_like(self.q)  # velocity of generalized coordinates
        self.phi = jnp.zeros_like(self.params["roff"].flatten())

        # initialize ODE solver
        self.declare_parameter("sim_dt", 1e-4)
        self.sim_dt = jnp.array(self.get_parameter("sim_dt").value)
        self.declare_parameter("control_frequency", 100)
        self.control_frequency = self.get_parameter("control_frequency").value
        self.control_dt = 1 / self.control_frequency

        self.ode_fn = planar_hsa.ode_factory(dynamical_matrices_fn, self.params)
        self.declare_parameter("ode_solver_class", "Dopri5")
        self.ode_solver = getattr(diffrax, self.get_parameter("ode_solver_class").value)()

        @jit
        def simulation_fn(_t0: Array, _t1: Array, _x0: Array, _phi: Array = jnp.zeros_like(self.phi)) -> Array:
            """
            Simulate the system for a given control input.
            Args:
                _t0: initial time
                _x0: initial state
                _phi: control input
            Returns:
                _x1: final state
            """
            ode_term = ODETerm(partial(self.ode_fn, u=_phi))
            sol = diffeqsolve(
                ode_term,
                solver=self.ode_solver,
                t0=_t0,
                t1=_t1,
                dt0=self.sim_dt,
                y0=_x0,
                max_steps=None,
                # saveat=SaveAt(ts=video_ts),
            )
            _x1 = sol.ys[-1, :]  # final state of the simulation
            return _x1

        self.simulation_fn = simulation_fn
        # jit the simulation function
        x0_dummy = jnp.zeros((2 * self.n_q,))
        phi_dummy = jnp.zeros_like(self.params["roff"].flatten())
        x1_dummy = self.simulation_fn(jnp.array(0.0), jnp.array(self.control_dt), x0_dummy, phi_dummy)

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

        # initialize timer for the control loop
        self.control_timer = self.create_timer(self.control_dt, self.call_controller)

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

        self.get_logger().info("Finished initializing planar_sim_node.")

    def phi_callback(self, msg: Float64MultiArray):
        # demanded rod twist angles
        self.phi = jnp.array(msg.data)    

    def call_controller(self):
        # the current clock time
        clock_current_time_obj = self.get_clock().now()
        clock_current_time = clock_current_time_obj.nanoseconds * 1e-9
        # compute the relative time to the start of the simulation
        t0 = self.clock_time - self.clock_start_time
        t1 = clock_current_time - self.clock_start_time

        # current state of the system
        x0 = jnp.concatenate((self.q, self.q_d))

        # simulate the system
        x1 = self.simulation_fn(t0, t1, x0, self.phi)

        # update the state of the system
        self.q, self.q_d = x1[:self.n_q], x1[self.n_q:]

        # publish the configuration
        configuration_msg = PlanarCsConfiguration(
            kappa_b=self.q[0].item(),
            sigma_sh=self.q[1].item(),
            sigma_a=self.q[2].item(),
        )
        configuration_msg.header.stamp = clock_current_time_obj.to_msg()
        self.configuration_pub.publish(configuration_msg)

        # publish configuration velocity
        configuration_velocity_msg = Float64MultiArray(
            data=self.q_d.tolist()
        )
        self.configuration_velocity_pub.publish(configuration_velocity_msg)

        # update the clock time
        self.clock_time = clock_current_time


def main(args=None):
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
