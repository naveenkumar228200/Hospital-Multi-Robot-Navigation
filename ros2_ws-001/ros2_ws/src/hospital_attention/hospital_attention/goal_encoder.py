import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

import math


class GoalEncoder(Node):

    def __init__(self):

        super().__init__("goal_encoder")

        self.robot_x = None
        self.robot_y = None

        self.goal_x = None
        self.goal_y = None

        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_callback,
            10
        )

        self.goal_pub = self.create_publisher(
            Float32MultiArray,
            "/goal_feature",
            10
        )

        self.get_logger().info(
            "Goal Encoder Started"
        )

    # ---------------------------------
    # Robot Position
    # ---------------------------------

    def odom_callback(self, msg):

        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        self.publish_goal_feature()

    # ---------------------------------
    # Goal Position
    # ---------------------------------

    def goal_callback(self, msg):

        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

        self.publish_goal_feature()

    # ---------------------------------
    # Goal Encoding
    # ---------------------------------

    def publish_goal_feature(self):

        if (
            self.robot_x is None or
            self.goal_x is None
        ):
            return

        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y

        goal_distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        goal_angle = math.atan2(
            dy,
            dx
        )

        msg = Float32MultiArray()

        msg.data = [
            float(goal_distance),
            float(goal_angle)
        ]

        self.goal_pub.publish(msg)

        self.get_logger().info(
            f"Distance={goal_distance:.2f} "
            f"Angle={math.degrees(goal_angle):.2f}"
        )


def main(args=None):

    rclpy.init(args=args)

    node = GoalEncoder()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
