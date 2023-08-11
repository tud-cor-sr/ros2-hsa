# ROS2 wrapper for planar HSA simulator

## Usage

### Start the simulator usingg launch file

```bash
ros2 launch ../launch/hsa_planar_sim_launch.py
```

### Manually send control commands

```bash
ros2 topic pub /control_input example_interfaces/Float64MultiArray 'data: [3.0, 0.0]' -1
```
