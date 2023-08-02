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
from mocap_optitrack_interfaces.msg import RigidBodyArray, PlanarCsConfiguration

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
        (_, _, _, inverse_kinematics_end_effector_fn, _, _) = planar_hsa.factory(
            sym_exp_filepath
        )

        self.params = PARAMS_CONTROL

        # parameter for specifying a different axial rest strain
        self.declare_parameter("sigma_a_eq", self.params["sigma_a_eq"].mean().item())
        sigma_a_eq = self.get_parameter("sigma_a_eq").value
        self.params["sigma_a_eq"] = sigma_a_eq * jnp.ones_like(self.params["sigma_a_eq"])
        self.get_logger().info(f"sigma_a_eq: {self.params['sigma_a_eq']}")

        self.phi_sub = self.create_subscription(
            Float64MultiArray,
            "/topic",
            self.phi_callback,
            10,
        )

    def phi_callback(self, msg: Float64MultiArray):
        
        print(msg.data)


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
