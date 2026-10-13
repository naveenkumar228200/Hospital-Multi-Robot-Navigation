
import rclpy
from rclpy.node import Node

import numpy as np
import torch

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

from hospital_attention.ppo.ppo_network import PPO


class PolicyNode(Node):

    def __init__(self):

        super().__init__("policy_node")

        ########################################

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        ########################################

        self.model = PPO().to(

            self.device

        )

        checkpoint = (

            "/home/lucifer/ros2_ws/"

            "best_model.pth"

        )

        try:

            ckpt = torch.load(

                checkpoint,

                map_location=self.device,

                weights_only=False

            )

            self.model.load_state_dict(

                ckpt["model"]

            )

            self.get_logger().info(

                "Loaded best_model.pth"

            )

        except Exception as e:

            self.get_logger().error(

                f"Model loading failed : {e}"

            )

        ########################################

        self.model.eval()

        ########################################

        self.create_subscription(

            Float32MultiArray,

            "/ppo_state",

            self.state_callback,

            10

        )

        ########################################

        self.cmd_pub = self.create_publisher(

            Twist,

            "/cmd_vel",

            10

        )

        ########################################

        self.counter = 0

        self.get_logger().info(

            "Policy Node Started"

        )

    #################################################

    def state_callback(

            self,

            msg

    ):

        state = np.array(

            msg.data,

            dtype=np.float32

        )

        ########################################

        if len(state) != 160:

            self.get_logger().warn(

                f"Invalid State Size : {len(state)}"

            )

            return

        ########################################

        state = torch.tensor(

            state,

            dtype=torch.float32

        ).unsqueeze(

            0

        ).to(

            self.device

        )

        ########################################

        with torch.no_grad():

            mean, value = self.model(

                state

            )

        ########################################

        action = mean.squeeze(

            0

        ).cpu().numpy()

        ########################################

        raw_linear = float(

            action[0]

        )

        raw_angular = float(

            action[1]

        )

        ########################################

        linear = np.tanh(

            raw_linear

        )

        linear = (

            linear + 1.0

        ) / 2.0

        linear *= 0.6

        ########################################

        angular = np.tanh(

            raw_angular

        )

        angular *= 1.5

        ########################################

        cmd = Twist()

        cmd.linear.x = float(

            linear

        )

        cmd.angular.z = float(

            angular

        )

        self.cmd_pub.publish(

            cmd

        )

        ########################################

        self.counter += 1

        ########################################

        if self.counter % 20 == 0:

            self.get_logger().info(

                f"RAW : "

                f"{raw_linear:.3f} "

                f"{raw_angular:.3f}"

            )

            self.get_logger().info(

                f"CMD : "

                f"vx={linear:.3f} "

                f"wz={angular:.3f}"

            )


########################################################


def main(args=None):

    rclpy.init(

        args=args

    )

    node = PolicyNode()

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


