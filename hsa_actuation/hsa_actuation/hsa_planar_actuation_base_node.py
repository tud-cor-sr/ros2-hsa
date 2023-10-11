from jax import Array
import jax.numpy as jnp
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
import numpy as np
import rclpy

from hsa_actuation.hsa_actuation_base_node import HsaActuationBaseNode


class HsaPlanarActuationBaseNode(HsaActuationBaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.declare_parameter("hsa_material", "fpu")
        hsa_material = self.get_parameter("hsa_material").value
        if hsa_material == "fpu":
            self.params = PARAMS_FPU_CONTROL.copy()
        elif hsa_material == "epu":
            self.params = PARAMS_EPU_CONTROL.copy()
        else:
            raise ValueError(f"Unknown HSA material: {hsa_material}")

        self.control_handedness = self.params["h"][
            0
        ]  # handedness of rods in first segment in control model

    def map_motor_angles_to_actuation_coordinates(self, motor_angles: Array) -> Array:
        """
        Map the motor angles to the actuation coordinates. The actuation coordinates are defined as twist angle
        of an imagined rod on the left and right respectively.
        """

        phi = jnp.stack(
            [
                (
                    motor_angles[2] * self.rod_handedness[2]
                    + motor_angles[3] * self.rod_handedness[3]
                )
                * self.control_handedness[0]
                / 2,
                (
                    motor_angles[0] * self.rod_handedness[0]
                    + motor_angles[1] * self.rod_handedness[1]
                )
                * self.control_handedness[1]
                / 2,
            ]
        )
        return phi

    def map_actuation_coordinates_to_motor_angles(self, phi: Array) -> Array:
        """
        We devise the control input in positive actuation coordinates of shape (2, ). However, we need to actuate
        four motors. This function maps the two actuation coordinates to the four motor angles.
        """

        motor_angles = jnp.stack(
            [
                phi[1] * self.control_handedness[1] * self.rod_handedness[0],
                phi[1] * self.control_handedness[1] * self.rod_handedness[1],
                phi[0] * self.control_handedness[0] * self.rod_handedness[2],
                phi[0] * self.control_handedness[0] * self.rod_handedness[3],
            ]
        )
        return motor_angles
