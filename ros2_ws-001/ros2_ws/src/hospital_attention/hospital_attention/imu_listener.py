import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

import torch

from hospital_attention.models.imu_encoder import IMUEncoder


class IMUListener(Node):

    def __init__(self):

        super().__init__("imu_listener")

        self.subscription = self.create_subscription(
            Imu,
            "/imu",
            self.callback,
            10
        )

        self.encoder = IMUEncoder()
        self.encoder.eval()

        self.feature_pub = self.create_publisher(
            Float32MultiArray,
            "/imu_feature",
            10
        )

        self.get_logger().info(
            "IMU Listener Started"
        )

    def callback(self, msg):

        try:

            imu_vector = [

                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,

                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,

                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w

            ]

            imu_tensor = torch.tensor(
                imu_vector,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():

                features = self.encoder(
                    imu_tensor
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
                f"IMU input shape: {tuple(imu_tensor.shape)}"
            )

            self.get_logger().info(
                f"IMU feature shape: {tuple(features.shape)}"
            )

            self.get_logger().info(
                f"First feature value: "
                f"{features[0][0].item():.4f}"
            )

        except Exception as e:

            self.get_logger().error(
                f"IMU processing failed: {e}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = IMUListener()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
