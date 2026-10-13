import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

import torch

from hospital_attention.models.wheel_encoder import WheelEncoder


class WheelListener(Node):

    def __init__(self):

        super().__init__("wheel_listener")

        self.subscription = self.create_subscription(
            Odometry,
            "/odom",
            self.callback,
            10
        )

        self.encoder = WheelEncoder()
        self.encoder.eval()

        self.feature_pub = self.create_publisher(
            Float32MultiArray,
            "/wheel_feature",
            10
        )

        self.get_logger().info(
            "Wheel Listener Started"
        )

    def callback(self, msg):

        try:

            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y

            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y

            wz = msg.twist.twist.angular.z

            wheel_tensor = torch.tensor(
                [[px, py, vx, vy, wz]],
                dtype=torch.float32
            )

            self.get_logger().info(
                f"Wheel input shape: {tuple(wheel_tensor.shape)}"
            )

            with torch.no_grad():

                features = self.encoder(
                    wheel_tensor
                )

            feature_msg = Float32MultiArray()

            feature_msg.data = (
                features
                .squeeze(0)
                .tolist()
            )

            self.feature_pub.publish(
                feature_msg
            )

            self.get_logger().info(
                f"Wheel feature shape: {tuple(features.shape)}"
            )

            self.get_logger().info(
                f"First feature value: "
                f"{features[0][0].item():.4f}"
            )

        except Exception as e:

            self.get_logger().error(
                f"Wheel encoding failed: {e}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = WheelListener()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
