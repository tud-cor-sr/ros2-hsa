import cv2  # importing cv2
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

from hsa_control_interfaces.msg import PlanarSetpoint
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration
from sensor_msgs.msg import Image

import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
from jsrm.systems import planar_hsa

from hsa_visualization.planar_opencv_renderer import robot_rendering_factory


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
            _,
            _,
            _,
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

        # initialize state and control input
        self.q = jnp.zeros_like(self.xi_eq)  # generalized coordinates
        self.q_msg = PlanarCsConfiguration()
        self.n_q = self.q.shape[0]  # number of generalized coordinates

        # initialize the rendering function
        self.declare_parameter("open_cv2_window", True)
        self.open_cv2_window = self.get_parameter("open_cv2_window").value
        self.declare_parameter("image_width", 400)
        self.declare_parameter("image_height", 400)
        self.declare_parameter("invert_colors", False)
        # x becomes -x and y becomes -y when rendering
        self.declare_parameter("invert_coordinates", True)
        self.rendering_fn = robot_rendering_factory(
            forward_kinematics_end_effector_fn,
            forward_kinematics_virtual_backbone_fn,
            sys_helpers["forward_kinematics_rod_fn"],
            sys_helpers["forward_kinematics_platform_fn"],
            params=self.params,
            width=self.get_parameter("image_width").value,
            height=self.get_parameter("image_height").value,
            num_points=25,
            inverted_coordinates=self.get_parameter("invert_coordinates"),
            invert_colors=self.get_parameter("invert_colors"),
        )
        if self.open_cv2_window:
            cv2.namedWindow("Planar HSA rendering", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                "Planar HSA rendering",
                self.get_parameter("image_width").value,
                self.get_parameter("image_height").value,
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

        self.chiee_des = None
        self.declare_parameter("setpoint_topic", "setpoint")
        self.setpoint_sub = self.create_subscription(
            PlanarSetpoint,
            self.get_parameter("setpoint_topic").value,
            self.setpoint_callback,
            10,
        )

        self.chiee_at = None
        self.declare_parameter("attractor_topic", "attractor")
        self.attractor_sub = self.create_subscription(
            PlanarSetpoint,
            self.get_parameter("attractor_topic").value,
            self.attractor_callback,
            10,
        )

        self.declare_parameter("rendering_frequency", 10.0)
        self.rendering_timer = self.create_timer(
            1 / self.get_parameter("rendering_frequency").value, self.render_robot
        )

    def configuration_callback(self, msg: PlanarCsConfiguration):
        """
        Callback for the configuration topic.
        Now, we need to render the robot for the given configuration.
        """
        # set the current configuration
        self.q = jnp.array([msg.kappa_b, msg.sigma_sh, msg.sigma_a])
        self.q_msg = msg

    def setpoint_callback(self, msg: PlanarSetpoint):
        self.chiee_des = jnp.array(
            [msg.chiee_des.x, msg.chiee_des.y, msg.chiee_des.theta]
        )

    def attractor_callback(self, msg: PlanarSetpoint):
        self.chiee_at = jnp.array(
            [msg.chiee_des.x, msg.chiee_des.y, msg.chiee_des.theta]
        )

    def render_robot(self):
        # self.get_logger().info(f"Rendering robot for configuration: {self.q}")
        img = self.rendering_fn(self.q, self.chiee_des, self.chiee_at)

        img_msg = self.ros_opencv_bridge.cv2_to_imgmsg(img)
        img_msg.header = self.q_msg.header
        self.rendering_pub.publish(img_msg)

        if self.open_cv2_window:
            cv2.imshow("Planar HSA rendering", img)
            cv2.waitKey(1)  # wait 1ms to prevent waiting for key press


def main(args=None):
    rclpy.init(args=args)
    print("Hi from planar_viz_node.")

    node = PlanarVizNode()

    rclpy.spin(node)

    # destroy all CV2 windows
    cv2.destroyAllWindows()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
