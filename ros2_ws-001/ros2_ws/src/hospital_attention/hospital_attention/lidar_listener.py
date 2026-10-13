import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray

import numpy as np
import torch

from hospital_attention.models.lidar_encoder_v2 import (
    LidarEncoderV2
)


class LidarListener(Node):

    def __init__(self):

        super().__init__(

            "lidar_listener"

        )

        self.subscription = self.create_subscription(

            PointCloud2,

            "/velodyne_points/points",

            self.lidar_callback,

            10

        )

        self.device = torch.device(

            "cpu"

        )

        self.encoder = LidarEncoderV2()

        self.encoder.to(

            self.device

        )

        self.encoder.eval()

        for p in self.encoder.parameters():

            p.requires_grad = False

        self.feature_pub = self.create_publisher(

            Float32MultiArray,

            "/lidar_feature",

            10

        )

        self.get_logger().info(

            "LiDAR Listener Started"

        )

    ###################################################

    def lidar_callback(

            self,

            msg

    ):

        try:

            cloud = point_cloud2.read_points(

                msg,

                field_names=(

                    "x",

                    "y",

                    "z"

                ),

                skip_nans=True

            )

            points = np.array(

                [

                    [

                        p[0],

                        p[1],

                        p[2]

                    ]

                    for p in cloud

                ],

                dtype=np.float32

            )

            if len(points) == 0:

                return

            points = points[

                np.isfinite(

                    points

                ).all(

                    axis=1

                )

            ]

            if len(points) < 1024:

                return

            ################################################

            idx = np.random.choice(

                len(points),

                1024,

                replace=False

            )

            sampled = points[

                idx

            ]

            ################################################

            tensor = torch.tensor(

                sampled,

                dtype=torch.float32,

                device=self.device

            )

            ################################################
            # Shape = (1,1024,3)
            ################################################

            tensor = tensor.unsqueeze(

                0

            )

            ################################################

            self.get_logger().info(

                f"{tensor.shape}"

            )

            with torch.no_grad():

                feat = self.encoder(

                    tensor

                )

            ################################################

            msg_out = Float32MultiArray()

            msg_out.data = (

                feat

                .squeeze()

                .cpu()

                .tolist()

            )

            self.feature_pub.publish(

                msg_out

            )

        except Exception as e:

            self.get_logger().error(

                f"LiDAR failed: {e}"

            )


#####################################################

def main(args=None):

    rclpy.init(

        args=args

    )

    node = LidarListener()

    try:

        rclpy.spin(

            node

        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
