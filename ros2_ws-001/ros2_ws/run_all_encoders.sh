#!/bin/bash

source ~/ros2_ws/install/setup.bash

ros2 run hospital_attention lidar_listener &
ros2 run hospital_attention depth_listener &
ros2 run hospital_attention imu_listener &
ros2 run hospital_attention wheel_listener

wait
