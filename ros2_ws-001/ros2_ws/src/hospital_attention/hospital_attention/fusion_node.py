import rclpy

from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

from hospital_attention.models.attention_fusion import AttentionFusion

import torch


class FusionNode(Node):

    def __init__(self):

        super().__init__("fusion_node")

        self.lidar_feature = None
        self.depth_feature = None
        self.imu_feature = None
        self.wheel_feature = None

        self.fusion = AttentionFusion()

        self.fusion.eval()

        ############################################

        self.create_subscription(

            Float32MultiArray,

            "/lidar_feature",

            self.lidar_callback,

            10

        )

        self.create_subscription(

            Float32MultiArray,

            "/depth_feature",

            self.depth_callback,

            10

        )

        self.create_subscription(

            Float32MultiArray,

            "/imu_feature",

            self.imu_callback,

            10

        )

        self.create_subscription(

            Float32MultiArray,

            "/wheel_feature",

            self.wheel_callback,

            10

        )

        ############################################

        self.fused_pub = self.create_publisher(

            Float32MultiArray,

            "/fused_feature",

            10

        )

        ############################################

        self.create_timer(

            0.1,

            self.try_fusion

        )

        ############################################

        self.get_logger().info(

            "Fusion Node Started"

        )

    #####################################################

    def lidar_callback(self, msg):

        self.get_logger().info(

            "LiDAR received"

        )

        self.lidar_feature = torch.tensor(

            msg.data,

            dtype=torch.float32

        ).unsqueeze(0)

    #####################################################

    def depth_callback(self, msg):

        self.get_logger().info(

            "Depth received"

        )

        self.depth_feature = torch.tensor(

            msg.data,

            dtype=torch.float32

        ).unsqueeze(0)

    #####################################################

    def imu_callback(self, msg):

        self.get_logger().info(

            "IMU received"

        )

        self.imu_feature = torch.tensor(

            msg.data,

            dtype=torch.float32

        ).unsqueeze(0)

    #####################################################

    def wheel_callback(self, msg):

        self.get_logger().info(

            "Wheel received"

        )

        self.wheel_feature = torch.tensor(

            msg.data,

            dtype=torch.float32

        ).unsqueeze(0)

    #####################################################

    def try_fusion(self):

        self.get_logger().info(

            "Timer running"

        )

        self.get_logger().info(

            f"{self.lidar_feature is not None} "

            f"{self.depth_feature is not None} "

            f"{self.imu_feature is not None} "

            f"{self.wheel_feature is not None}"

        )

        if (

            self.lidar_feature is None

            or

            self.depth_feature is None

            or

            self.imu_feature is None

            or

            self.wheel_feature is None

        ):

            return

        self.get_logger().info(

            "All modalities available"

        )

        try:

            print(

                "LiDAR:",

                self.lidar_feature.shape

            )

            print(

                "Depth:",

                self.depth_feature.shape

            )

            print(

                "IMU:",

                self.imu_feature.shape

            )

            print(

                "Wheel:",

                self.wheel_feature.shape

            )

            with torch.no_grad():

                fused = self.fusion(

                    self.lidar_feature,

                    self.depth_feature,

                    self.imu_feature,

                    self.wheel_feature

                )

            print(

                "Fused:",

                fused.shape

            )

            msg = Float32MultiArray()

            msg.data = fused.squeeze().tolist()

            self.fused_pub.publish(

                msg

            )

            self.get_logger().info(

                "Published fused feature"

            )

        except Exception as e:

            self.get_logger().error(

                str(e)

            )


#########################################################

def main(args=None):

    rclpy.init(args=args)

    node = FusionNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()
