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

from example_interfaces.msg import Float64MultiArray
from geometry_msgs.msg import Pose2D
from hsa_control_interfaces.msg import Pose2DStamped
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration

import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
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
            forward_kinematics_virtual_backbone_fn,
            forward_kinematics_end_effector_fn,
            jacobian_end_effector_fn,
            inverse_kinematics_end_effector_fn,
            dynamical_matrices_fn,
            sys_helpers,
        ) = planar_hsa.factory(sym_exp_filepath)

        self.declare_parameter("hsa_material", "fpu")
        hsa_material = self.get_parameter("hsa_material").value
        if hsa_material == "fpu":
            self.params = PARAMS_FPU_CONTROL.copy()
        elif hsa_material == "epu":
            self.params = PARAMS_EPU_CONTROL.copy()
        else:
            raise ValueError(f"Unknown HSA material: {hsa_material}")

        # parameters for specifying different rest strains
        self.declare_parameter("kappa_b_eq", self.params["kappa_b_eq"].mean().item())
        self.declare_parameter("sigma_sh_eq", self.params["sigma_sh_eq"].mean().item())
        self.declare_parameter("sigma_a_eq1", self.params["sigma_a_eq"][0, 0].item())
        self.declare_parameter("sigma_a_eq2", self.params["sigma_a_eq"][0, 1].item())
        kappa_b_eq = self.get_parameter("kappa_b_eq").value
        sigma_sh_eq = self.get_parameter("sigma_sh_eq").value
        sigma_a_eq1 = self.get_parameter("sigma_a_eq1").value
        sigma_a_eq2 = self.get_parameter("sigma_a_eq2").value
        self.params["kappa_b_eq"] = kappa_b_eq * jnp.ones_like(
            self.params["kappa_b_eq"]
        )
        self.params["sigma_sh_eq"] = sigma_sh_eq * jnp.ones_like(
            self.params["sigma_sh_eq"]
        )
        self.params["sigma_a_eq"] = jnp.array([[sigma_a_eq1, sigma_a_eq2]])
        # actual rest strain
        self.xi_eq = sys_helpers["rest_strains_fn"](self.params)  # rest strains

        # initialize forward kinematic functions
        self.forward_kinematics_end_effector_fn = jit(
            partial(forward_kinematics_end_effector_fn, self.params)
        )
        self.jacobian_end_effector_fn = jit(
            partial(jacobian_end_effector_fn, self.params)
        )

        # initialize state and control input
        self.q = jnp.zeros_like(self.xi_eq)  # generalized coordinates
        self.n_q = self.q.shape[0]  # number of generalized coordinates
        self.q_d = jnp.zeros_like(self.q)  # velocity of generalized coordinates
        self.phi = jnp.zeros_like(self.params["roff"].flatten())

        # initialize ODE solver
        self.declare_parameter("sim_dt", 1e-4)
        self.sim_dt = jnp.array(self.get_parameter("sim_dt").value)
        self.declare_parameter("control_frequency", 100.0)
        self.control_frequency = self.get_parameter("control_frequency").value
        self.control_dt = 1 / self.control_frequency

        self.ode_fn = planar_hsa.ode_factory(dynamical_matrices_fn, self.params)
        self.declare_parameter("ode_solver_class", "Dopri5")
        self.ode_solver = getattr(
            diffrax, self.get_parameter("ode_solver_class").value
        )()

        @jit
        def simulation_fn(
            _t0: Array, _t1: Array, _x0: Array, _phi: Array = jnp.zeros_like(self.phi)
        ) -> Array:
            """
            Simulate the system for a given control input.
            Args:
                _t0: initial time
                _x0: initial state
                _phi: control input
            Returns:
                _x1: final state
            """
            ode_term = ODETerm(self.ode_fn)
            sol = diffeqsolve(
                ode_term,
                solver=self.ode_solver,
                t0=_t0,
                t1=_t1,
                dt0=self.sim_dt,
                y0=_x0,
                args=_phi,
                max_steps=None,
                # saveat=SaveAt(ts=video_ts),
            )
            _x1 = sol.ys[-1, :]  # final state of the simulation
            return _x1

        self.simulation_fn = simulation_fn
        # jit the simulation function
        x0_dummy = jnp.zeros((2 * self.n_q,))
        phi_dummy = jnp.zeros_like(self.params["roff"].flatten())
        x1_dummy = self.simulation_fn(
            jnp.array(0.0), jnp.array(self.control_dt), x0_dummy, phi_dummy
        )
        self.get_logger().info("Finished jitting the simulation_fn.")

        # initialize time
        self.clock_start_time = (
            self.get_clock().now().nanoseconds * 1e-9
        )  # time in seconds
        self.clock_time = self.clock_start_time

        # create a publisher for the configuration and its velocity
        self.declare_parameter("configuration_topic", "configuration")
        self.configuration_pub = self.create_publisher(
            PlanarCsConfiguration, self.get_parameter("configuration_topic").value, 10
        )
        self.declare_parameter("configuration_velocity_topic", "configuration_velocity")
        self.configuration_velocity_pub = self.create_publisher(
            PlanarCsConfiguration,
            self.get_parameter("configuration_velocity_topic").value,
            10,
        )

        # create a publisher for the end-effector pose and its velocity
        self.declare_parameter("end_effector_pose_topic", "end_effector_pose")
        self.end_effector_pose_pub = self.create_publisher(
            Pose2DStamped, self.get_parameter("end_effector_pose_topic").value, 10
        )
        self.declare_parameter("end_effector_velocity_topic", "end_effector_velocity")
        self.end_effector_velocity_pub = self.create_publisher(
            Pose2DStamped,
            self.get_parameter("end_effector_velocity_topic").value,
            10,
        )

        # initialize timer for the simulation loop
        self.sim_timer = self.create_timer(self.control_dt, self.simulate_system)

        # create the subscription to the control input
        self.declare_parameter("control_input_topic", "control_input")
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

    def simulate_system(self):
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
        self.q, self.q_d = x1[: self.n_q], x1[self.n_q :]

        # compute the end-effector pose
        chiee = self.forward_kinematics_end_effector_fn(self.q)
        # compute the end-effector velocity
        Jee = self.jacobian_end_effector_fn(self.q)
        chiee_d = Jee @ self.q_d

        # publish the configuration
        configuration_msg = PlanarCsConfiguration(
            kappa_b=self.q[0].item(),
            sigma_sh=self.q[1].item(),
            sigma_a=self.q[2].item(),
        )
        configuration_msg.header.stamp = clock_current_time_obj.to_msg()
        self.configuration_pub.publish(configuration_msg)

        # publish configuration velocity
        configuration_velocity_msg = PlanarCsConfiguration(
            kappa_b=self.q_d[0].item(),
            sigma_sh=self.q_d[1].item(),
            sigma_a=self.q_d[2].item(),
        )
        configuration_velocity_msg.header.stamp = clock_current_time_obj.to_msg()
        self.configuration_velocity_pub.publish(configuration_velocity_msg)

        # publish the end-effector pose
        end_effector_pose_msg = Pose2DStamped(
            pose=Pose2D(
                x=chiee[0].item(),
                y=chiee[1].item(),
                theta=chiee[2].item(),
            )
        )
        end_effector_pose_msg.header.stamp = clock_current_time_obj.to_msg()
        self.end_effector_pose_pub.publish(end_effector_pose_msg)

        # publish the end-effector velocity
        end_effector_velocity_msg = Pose2DStamped(
            pose=Pose2D(
                x=chiee_d[0].item(),
                y=chiee_d[1].item(),
                theta=chiee_d[2].item(),
            )
        )
        end_effector_velocity_msg.header.stamp = clock_current_time_obj.to_msg()
        self.end_effector_velocity_pub.publish(end_effector_velocity_msg)

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


if __name__ == "__main__":
    main()
