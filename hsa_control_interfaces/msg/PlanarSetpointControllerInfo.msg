# Messages
std_msgs/Header header
hsa_control_interfaces/PlanarSetpoint planar_setpoint  # desired setpoint 
geometry_msgs/Pose2D chiee  # current pose of the end-effector
mocap_optitrack_interfaces/PlanarCsConfiguration q  # current configuration
mocap_optitrack_interfaces/PlanarCsConfiguration q_d  # current velocity of configuration
float64[] e_int  # integrated error
float64[] varphi_des  # desired actuation coordinates
float64[] varphi  # actuation coordinates
float64[] phi_des_unsat  # unsaturated control input
float64[] phi_des_sat  # saturated control input
float64[] motor_goal_angles  # desired motor angles
