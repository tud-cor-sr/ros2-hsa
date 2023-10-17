# Messages
std_msgs/Header header
hsa_control_interfaces/PlanarSetpoint planar_setpoint  # desired setpoint 
hsa_control_interfaces/Pose2DStamped chiee  # current pose of the end-effector
hsa_control_interfaces/Pose2DStamped chiee_d  # current velocity / twist of the end-effector
mocap_optitrack_interfaces/PlanarCsConfiguration q  # current configuration
mocap_optitrack_interfaces/PlanarCsConfiguration q_d  # current velocity of configuration
float64[] e_int  # integrated error
float64[] f  # force in operational space
float64[] tau  # torque in configuration space
float64[] varphi_des  # desired actuation coordinates
float64[] varphi  # actuation coordinates
float64[] phi_des_unsat  # unsaturated control input
float64[] phi_des_sat  # saturated control input
float64[] motor_goal_angles  # desired motor angles
float64 actuation_optimality_error  # optimality error for generating the demanded configuration space torque
