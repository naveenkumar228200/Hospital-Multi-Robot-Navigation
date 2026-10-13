from setuptools import find_packages, setup

package_name = 'hospital_attention'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/hospital_attention/launch',
            ['launch/hospital_pipeline.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lucifer',
    maintainer_email='lucifer@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                      'lidar_listener = hospital_attention.lidar_listener:main',
                      'depth_listener = hospital_attention.depth_listener:main',
                      'imu_listener = hospital_attention.imu_listener:main',
                      'wheel_listener = hospital_attention.wheel_listener:main',
                      'fusion_node = hospital_attention.fusion_node:main',
                      'human_detector = hospital_attention.human_detector:main',
                      'human_localizer = hospital_attention.human_localizer:main',
                      'human_tracker = hospital_attention.human_tracker:main',
                      'kalman_tracker = hospital_attention.kalman_tracker:main',
                      'trajectory_visualizer = hospital_attention.trajectory_visualizer:main',
                      'intention_predictor = hospital_attention.intention_predictor:main',
                      'human_intention_encoder = hospital_attention.human_intention_encoder:main',
                      'goal_encoder = hospital_attention.goal_encoder:main',
                      'state_aggregator = hospital_attention.state_aggregator:main',
                      'trainer = hospital_attention.ppo.trainer:main',
                      'policy_node = hospital_attention.ppo.policy_node:main',
                      'episode_manager = hospital_attention.ppo.episode_manager:main',
                      'action_interface = hospital_attention.ppo.action_interface:main',

        ],
    },
)
