import cv_bridge
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
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration
from sensor_msgs.msg import Image

import jsrm
from jsrm.parameters.hsa_params import PARAMS_CONTROL
from jsrm.systems import planar_hsa

from hsa_visualization.planar_opencv_renderer import draw_robot


class PlanarVizNode(Node):
    def __init__(self):
        super().__init__("planar_viz_node")

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

        self.params = PARAMS_CONTROL

        # parameter for specifying a different axial rest strain
        self.declare_parameter("sigma_a_eq", self.params["sigma_a_eq"].mean().item())
        sigma_a_eq = self.get_parameter("sigma_a_eq").value
        self.params["sigma_a_eq"] = sigma_a_eq * jnp.ones_like(
            self.params["sigma_a_eq"]
        )
        # actual rest strain
        self.xi_eq = sys_helpers["rest_strains_fn"](self.params)  # rest strains

        # initialize state and control input
        self.q = jnp.zeros_like(self.xi_eq)  # generalized coordinates
        self.n_q = self.q.shape[0]  # number of generalized coordinates

        # initialize the rendering function
        self.declare_parameter("image_width", 700)
        self.declare_parameter("image_height", 700)
        self.rendering_fn = partial(
            draw_robot,
            forward_kinematics_virtual_backbone_fn,
            sys_helpers["forward_kinematics_rod_fn"],
            sys_helpers["forward_kinematics_platform_fn"],
            self.params,
            width=self.get_parameter("image_width").value,
            height=self.get_parameter("image_height").value,
        )

        # initialize bridge for converting between ROS and OpenCV images
        self.ros_opencv_bridge = cv_bridge.CvBridge()

        self.declare_parameter("rendering_topic", "rendering")
        self.rendering_pub = self.create_publisher(
            Image, self.get_parameter("rendering_topic").value, 10
        )

        self.declare_parameter("configuration_topic", "configuration")
        self.configuration_sub = self.create_subscription(
            PlanarCsConfiguration,
            self.get_parameter("configuration_topic").value,
            self.configuration_callback,
            10,
        )

    def configuration_callback(self, msg: PlanarCsConfiguration):
        """
        Callback for the configuration topic.
        Now, we need to render the robot for the given configuration.
        """
        # set the current configuration
        self.q = jnp.array([msg.kappa_b, msg.sigma_sh, msg.sigma_a])
        img = self.rendering_fn(self.q)

        img_msg = self.ros_opencv_bridge.cv2_to_imgmsg(img)
        img_msg.header = msg.header
        self.rendering_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    print("Hi from planar_viz_node.")

    node = PlanarVizNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
