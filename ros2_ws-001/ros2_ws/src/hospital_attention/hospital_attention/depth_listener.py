import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from hospital_attention.models.depth_encoder import DepthEncoder

import numpy as np
import torch
import torch.nn.functional as F
import time


class DepthListener(Node):

    def __init__(self):

        super().__init__(

            "depth_listener"

        )

        ############################################

        self.subscription = self.create_subscription(

            Image,

            "/world/smart_hospital/model/tb3_1/link/base_link/sensor/depth_camera/depth_image",

            self.callback,

            10

        )

        ############################################

        self.device = torch.device(

            "cpu"

        )

        ############################################

        self.encoder = DepthEncoder()

        self.encoder.to(

            self.device

        )

        self.encoder.eval()

        for p in self.encoder.parameters():

            p.requires_grad = False

        ############################################

        self.feature_pub = self.create_publisher(

            Float32MultiArray,

            "/depth_feature",

            10

        )

        ############################################

        self.last_time = time.time()

        self.target_rate = 0.2

        ############################################

        self.get_logger().info(

            "Depth Listener Started"

        )

    ################################################

    def callback(

            self,

            msg

    ):

        current = time.time()

        if current - self.last_time < self.target_rate:

            return

        self.last_time = current

        try:

            depth = np.frombuffer(

                msg.data,

                dtype=np.float32

            ).reshape(

                msg.height,

                msg.width

            )

            ################################################

            depth = np.nan_to_num(

                depth,

                nan=0.0,

                posinf=10.0,

                neginf=0.0

            )

            ################################################

            depth_tensor = torch.from_numpy(

                depth

            )

            depth_tensor = depth_tensor.unsqueeze(

                0

            ).unsqueeze(

                0

            )

            ################################################

            depth_tensor = F.interpolate(

                depth_tensor,

                size=(96,96),

                mode='nearest'

            )

            ################################################

            depth_tensor = depth_tensor.to(

                self.device

            )

            ################################################

            with torch.no_grad():

                features = self.encoder(

                    depth_tensor

                )

            ################################################

            msg_out = Float32MultiArray()

            msg_out.data = (

                features

                .squeeze(

                    0

                )

                .cpu()

                .tolist()

            )

            self.feature_pub.publish(

                msg_out

            )

        except Exception as e:

            self.get_logger().error(

                f"Depth failed: {e}"

            )


####################################################

def main(args=None):

    rclpy.init(

        args=args

    )

    node = DepthListener()

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
