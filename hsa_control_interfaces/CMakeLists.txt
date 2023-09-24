cmake_minimum_required(VERSION 3.5)
project(hsa_control_interfaces)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(common_interfaces REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mocap_optitrack_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  "msg/PlanarSetpoint.msg"
  "msg/PlanarSetpointControllerInfo.msg"
  "msg/Pose2DStamped.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces common_interfaces example_interfaces geometry_msgs mocap_optitrack_interfaces std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
