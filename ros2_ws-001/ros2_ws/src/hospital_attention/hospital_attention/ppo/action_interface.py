import rclpy

from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

import numpy as np


class ActionInterface(Node):

    def __init__(self):

        super().__init__(

            "action_interface"

        )

        ##################################

        self.max_linear = 0.5

        self.max_angular = 1.5

        self.counter = 0

        ##################################

        self.cmd_pub = self.create_publisher(

            Twist,

            "/cmd_vel",

            10

        )

        ##################################

        self.create_subscription(

            Float32MultiArray,

            "/ppo_action",

            self.action_callback,

            10

        )

        ##################################

        self.get_logger().info(

            "Action Interface Started"

        )

    ################################################

    def action_callback(

            self,

            msg

    ):

        action = np.array(

            msg.data,

            dtype=np.float32

        )

        ################################################

        if len(action) != 2:

            return

        ################################################

        linear = float(

            action[0]

        )

        angular = float(

            action[1]

        )

        ################################################
        # Safety checks
        ################################################

        if not np.isfinite(

                linear

        ):

            linear = 0.0

        if not np.isfinite(

                angular

        ):

            angular = 0.0

        ################################################

        linear = np.clip(

            linear,

            -1.0,

            1.0

        )

        angular = np.clip(

            angular,

            -1.0,

            1.0

        )

        ################################################

        linear *= self.max_linear

        angular *= self.max_angular

        ################################################

        cmd = Twist()

        cmd.linear.x = linear

        cmd.angular.z = angular

        self.cmd_pub.publish(

            cmd

        )

        ################################################

        self.counter += 1

        if self.counter % 20 == 0:

            self.get_logger().info(

                f"cmd_vel "

                f"v={linear:.2f} "

                f"w={angular:.2f}"

            )

    ################################################

    def stop_robot(

            self

    ):

        stop = Twist()

        self.cmd_pub.publish(

            stop

        )


#####################################################


def main(args=None):

    rclpy.init(

        args=args

    )

    node = ActionInterface()

    try:

        rclpy.spin(

            node

        )

    except KeyboardInterrupt:

        pass

    finally:

        node.stop_robot()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
