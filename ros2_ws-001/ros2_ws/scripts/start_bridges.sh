#!/bin/bash

echo "Starting Gazebo-ROS2 Bridges..."

# Clock
ros2 run ros_gz_bridge parameter_bridge \
/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &

# LiDAR
ros2 run ros_gz_bridge parameter_bridge \
/velodyne_points/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked &

# IMU
ros2 run ros_gz_bridge parameter_bridge \
/imu@sensor_msgs/msg/Imu@gz.msgs.IMU &

# cmd_vel bridge
ros2 run ros_gz_bridge parameter_bridge \
/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist &

# Odometry
ros2 run ros_gz_bridge parameter_bridge \
/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry &

# Joint States
ros2 run ros_gz_bridge parameter_bridge \
/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model &

# Depth Camera
ros2 run ros_gz_bridge parameter_bridge \
/world/smart_hospital/model/tb3_1/link/base_link/sensor/depth_camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image &

# RGB Camera
ros2 run ros_gz_bridge parameter_bridge \
/rgb_image@sensor_msgs/msg/Image@gz.msgs.Image &

# Camera Info
ros2 run ros_gz_bridge parameter_bridge \
/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo &

echo "All bridges started."

wait
