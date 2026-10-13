import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray


class StateAggregator(Node):

    def __init__(self):

        super().__init__(

            "state_aggregator"

        )

        self.fused_feature = None
        self.human_feature = None
        self.goal_feature = None

        ##################################################

        self.create_subscription(

            Float32MultiArray,

            "/fused_feature",

            self.fused_callback,

            10

        )

        self.create_subscription(

            Float32MultiArray,

            "/human_feature",

            self.human_callback,

            10

        )

        self.create_subscription(

            Float32MultiArray,

            "/goal_feature",

            self.goal_callback,

            10

        )

        ##################################################

        self.pub = self.create_publisher(

            Float32MultiArray,

            "/ppo_state",

            10

        )

        ##################################################

        self.get_logger().info(

            "State Aggregator Started"

        )

    ##################################################

    def fused_callback(

            self,

            msg

    ):

        self.fused_feature = list(

            msg.data

        )

        self.publish_state()

    ##################################################

    def human_callback(

            self,

            msg

    ):

        self.human_feature = list(

            msg.data

        )

        self.publish_state()

    ##################################################

    def goal_callback(

            self,

            msg

    ):

        self.goal_feature = list(

            msg.data

        )

        self.publish_state()

    ##################################################

    def publish_state(

            self

    ):

        if (

                self.fused_feature is None

                or

                self.human_feature is None

                or

                self.goal_feature is None

        ):

            return

        ##################################################

        print()

        print(

            "Fused size :",

            len(

                self.fused_feature

            )

        )

        print(

            "Human size :",

            len(

                self.human_feature

            )

        )

        print(

            "Goal size :",

            len(

                self.goal_feature

            )

        )

        ##################################################

        state = (

                self.fused_feature

                +

                self.human_feature

                +

                self.goal_feature

        )

        ##################################################

        print(

            "TOTAL PPO STATE :",

            len(

                state

            )

        )

        print()

        ##################################################

        msg = Float32MultiArray()

        msg.data = state

        self.pub.publish(

            msg

        )

        ##################################################

        self.get_logger().info(

            f"PPO State Shape = ({len(state)},)"

        )

##################################################


def main(args=None):

    rclpy.init(

        args=args

    )

    node = StateAggregator()

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
