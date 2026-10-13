import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import numpy as np
import math


def quaternion_to_yaw(x, y, z, w):

    siny_cosp = 2.0 * (

        w * z + x * y

    )

    cosy_cosp = 1.0 - 2.0 * (

        y * y + z * z

    )

    return math.atan2(

        siny_cosp,

        cosy_cosp

    )


class HumanLocalizer(Node):

    def __init__(self):

        super().__init__(

            "human_localizer"

        )

        #######################################

        # 320x240 camera intrinsics

        #######################################

        self.fx = 277.19
        self.fy = 277.19

        self.cx = 160.0
        self.cy = 120.0

        #######################################

        self.depth_image = None

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        #######################################

        self.create_subscription(

            Float32MultiArray,

            "/people_detection",

            self.people_callback,

            5

        )

        self.create_subscription(

            Image,

            "/world/smart_hospital/model/tb3_1/link/base_link/sensor/depth_camera/depth_image",

            self.depth_callback,

            5

        )

        self.create_subscription(

            Odometry,

            "/odom",

            self.odom_callback,

            5

        )

        #######################################

        self.pub = self.create_publisher(

            Float32MultiArray,

            "/people_world_position",

            5

        )

        #######################################

        self.get_logger().info(

            "Human Localizer Started"

        )

    #######################################

    def depth_callback(

            self,

            msg

    ):

        try:

            depth = np.frombuffer(

                msg.data,

                dtype=np.float32

            )

            self.depth_image = depth.reshape(

                msg.height,

                msg.width

            )

        except Exception:

            pass

    #######################################

    def odom_callback(

            self,

            msg

    ):

        self.robot_x = (

            msg.pose.pose.position.x

        )

        self.robot_y = (

            msg.pose.pose.position.y

        )

        q = msg.pose.pose.orientation

        self.robot_yaw = quaternion_to_yaw(

            q.x,

            q.y,

            q.z,

            q.w

        )

    #######################################

    def people_callback(

            self,

            msg

    ):

        if self.depth_image is None:

            return

        detections = msg.data

        output = []

        try:

            for i in range(

                    0,

                    len(detections),

                    3

            ):

                u = int(

                    detections[i]

                )

                v = int(

                    detections[i + 1]

                )

                conf = float(

                    detections[i + 2]

                )

                if (

                    u < 0 or

                    v < 0 or

                    u >= self.depth_image.shape[1] or

                    v >= self.depth_image.shape[0]

                ):

                    continue

                ###################################

                x1 = max(

                    0,

                    u - 2

                )

                x2 = min(

                    self.depth_image.shape[1],

                    u + 3

                )

                y1 = max(

                    0,

                    v - 2

                )

                y2 = min(

                    self.depth_image.shape[0],

                    v + 3

                )

                ###################################

                window = self.depth_image[

                    y1:y2,

                    x1:x2

                ]

                valid = window[

                    np.isfinite(

                        window

                    )

                ]

                if len(valid) == 0:

                    continue

                depth = float(

                    np.median(

                        valid

                    )

                )

                if (

                    not np.isfinite(

                        depth

                    )

                ):

                    continue

                if depth < 0.1:

                    continue

                ###################################

                X = (

                    (u - self.cx)

                    * depth

                    / self.fx

                )

                Z = depth

                ###################################

                world_x = (

                    self.robot_x

                    + Z * math.cos(

                        self.robot_yaw

                    )

                    - X * math.sin(

                        self.robot_yaw

                    )

                )

                world_y = (

                    self.robot_y

                    + Z * math.sin(

                        self.robot_yaw

                    )

                    + X * math.cos(

                        self.robot_yaw

                    )

                )

                ###################################

                output.extend(

                    [

                        world_x,

                        world_y,

                        conf

                    ]

                )

            ###################################

            if len(output) > 0:

                out_msg = Float32MultiArray()

                out_msg.data = output

                self.pub.publish(

                    out_msg

                )

        except Exception as e:

            self.get_logger().error(

                f"Localization failed: {e}"

            )


#######################################


def main(args=None):

    rclpy.init(

        args=args

    )

    node = HumanLocalizer()

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


