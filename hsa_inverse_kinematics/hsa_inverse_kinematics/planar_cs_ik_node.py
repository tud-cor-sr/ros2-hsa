from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Pose2D
from mocap_optitrack_interfaces.msg import RigidBodyArray


class PlanarCsIkNode(Node):
    def __init__(self):
        super().__init__("planar_cs_ik_node")
        self.get_logger().info("Hi from planar_cs_ik_node.")

        self.declare_parameter("baseframe_rigid_bodies_topic", "baseframe_rigid_bodies")
        self.base_frame_rigid_bodies_sub = self.create_subscription(
            RigidBodyArray,
            self.get_parameter("baseframe_rigid_bodies_topic").value,
            self.listener_callback,
            10,
        )

        self.declare_parameter("mocap_platform_id", 4)
        self.mocap_platform_id = self.get_parameter("mocap_platform_id").value

        self.declare_parameter("end_effector_pose_topic", "end_effector_pose")
        self.end_effector_pose_pub = self.create_publisher(
            Pose2D, self.get_parameter("end_effector_pose_topic").value, 10
        )



        # self.declare_parameter("tf_base_topic", "tf_base")
        # self.declare_parameter("tf_platform_topic", "tf_platform")
        # self.subscription = self.create_subscription(
        #     TransformStamped,
        #     self.get_parameter("tf_base_topic").value,
        #     self.listener_callback,
        #     10,
        # )
        # # transformation from base tf to the start of the proximal end of the metamaterial
        # self.tf_fixed_base_offset = np.eye(4)
        # # transformation from base tf to the distal end of the metamaterial to the tf of the platform
        # self.tf_fixed_platform_offset = np.eye(4)

    def listener_callback(self, msg):
        for rigid_body_msg in msg.rigid_bodies:
            # self.get_logger().info('Rigid body: "%s"' % rigid_body_msg)
            if rigid_body_msg.id == self.mocap_platform_id:
                self.process_platform_msg(rigid_body_msg)
                

    def process_platform_msg(self, msg):
        position_msg = msg.pose_stamped.pose.position
        orientation_msg = msg.pose_stamped.pose.orientation

        position = np.array([position_msg.x, position_msg.y, position_msg.z])
        quat = np.array(
            [orientation_msg.x, orientation_msg.y, orientation_msg.z, orientation_msg.w]
        )
        rot = R.from_quat(quat)
        rotmat = rot.as_matrix()
        euler_xyz = rot.as_euler("xyz", degrees=False)

        # subtract distance from the platform markers to the top surface of the platform
        # the MoCap markers of the platform are roughly 7 mm above the end-effector frame (e.g. top surface of the platform)
        position = position - rotmat @ np.array([0, 0, 0.007])

        # define the SE(2) pose of the end-effector frame\
        # y-axis of the world frame becomes the negative x-axis of the end-effector frame
        # z-axis of the world frame becomes the y-axis of the end-effector frame
        # the rotation around the x-axis of the world frame is the same as the rotation around the negative z-axis of the end-effector frame
        chiee = np.array([
            -position[1], position[2], -euler_xyz[0],
        ])

        end_effector_pose_msg = Pose2D(x=chiee[0], y=chiee[1], theta=chiee[2])
        self.end_effector_pose_pub.publish(end_effector_pose_msg)

def main(args=None):
    rclpy.init(args=args)
    print("Hi from hsa_inverse_kinematics.")

    node = PlanarCsIkNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
