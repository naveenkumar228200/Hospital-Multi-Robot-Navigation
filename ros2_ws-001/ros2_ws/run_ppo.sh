#!/bin/bash

echo "Cleaning old PPO processes..."

pkill -f state_aggregator
pkill -f trainer
pkill -f action_interface
pkill -f episode_manager

sleep 2

cd ~/ros2_ws

source /opt/ros/jazzy/setup.bash
source install/setup.bash

################################################
# Background nodes
################################################

echo "Starting state_aggregator..."

ros2 run hospital_attention state_aggregator &
sleep 2

echo "Starting trainer..."

ros2 run hospital_attention trainer &
sleep 2

echo "Starting action_interface..."

ros2 run hospital_attention action_interface &
sleep 2

################################################
# Episode manager terminal
################################################

gnome-terminal -- bash -c "

cd ~/ros2_ws

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

echo 'Episode Manager Started'

ros2 run hospital_attention episode_manager

exec bash
"

################################################
# Monitor terminal
################################################

gnome-terminal -- bash -c "

cd ~/ros2_ws

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

echo
echo '===== PPO ACTION ====='
echo

ros2 topic echo /ppo_action &

sleep 1

echo
echo '===== CMD VEL ====='
echo

ros2 topic echo /cmd_vel

exec bash
"

################################################

echo
echo "======================================"
echo "PPO SYSTEM STARTED"
echo "======================================"

echo "Background Nodes:"
echo "  state_aggregator"
echo "  trainer"
echo "  action_interface"

echo
echo "Terminal 1:"
echo "  episode_manager"

echo
echo "Terminal 2:"
echo "  /ppo_action"
echo "  /cmd_vel"

echo "======================================"
