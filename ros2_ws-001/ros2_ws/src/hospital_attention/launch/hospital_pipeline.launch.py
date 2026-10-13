from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        Node(
            package='hospital_attention',
            executable='human_detector',
            name='human_detector',
            output='screen'
        ),

        Node(
            package='hospital_attention',
            executable='human_localizer',
            name='human_localizer',
            output='screen'
        ),

        Node(
            package='hospital_attention',
            executable='human_tracker',
            name='human_tracker',
            output='screen'
        ),

        Node(
            package='hospital_attention',
            executable='kalman_tracker',
            name='kalman_predictor',
            output='screen'
        ),

        # Future trajectory prediction
        Node(
            package='hospital_attention',
            executable='intention_predictor',
            name='intention_predictor',
            output='screen'
        ),

        # One-hot human encoding
        Node(
            package='hospital_attention',
            executable='human_intention_encoder',
            name='human_intention_encoder',
            output='screen'
        ),

        #Node(
         #   package='hospital_attention',
          #  executable='trajectory_visualizer',
           # name='trajectory_visualizer',
            #output='screen'
        #),

        
        Node(
            package='hospital_attention',
            executable='goal_encoder',
            name='goal_encoder',
            output='screen'
        ),
        Node(
            package='hospital_attention',
            executable='lidar_listener',
            name='lidar_listener',
            output='screen'
        ),
        Node(
            package='hospital_attention',
            executable='state_aggregator',
            name='state_aggregator',
            output='screen'
            ),

        Node(
            package='hospital_attention',
            executable='imu_listener',
            name='imu_listener',
            output='screen'
        ),
        Node(
            package='hospital_attention',
            executable='depth_listener',
            name='depth_listener',
            output='screen'
            ),

        Node(
            package='hospital_attention',
            executable='wheel_listener',
            name='wheel_listener',
            output='screen'
        ),

        Node(
            package='hospital_attention',
            executable='fusion_node',
            name='fusion_node',
            output='screen'
        ),

    ])
