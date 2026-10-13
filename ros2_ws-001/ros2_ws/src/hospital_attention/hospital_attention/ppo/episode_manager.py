import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool

import random
import subprocess
import time


class EpisodeManager(Node):

    def __init__(self):

        super().__init__(

            "episode_manager"

        )

        self.episode = 0

        self.max_steps = 500

        self.goal_threshold = 0.5

        self.step_count = 0

        self.goal_distance = 999.0

        self.collision = False

        self.resetting = False

        #################################

        self.goals = [

            (2.0, 1.0),

            (4.0, 2.0),

            (6.0, -1.0),

            (8.0, 3.0),

            (10.0, -2.0),

            (-10.0, 2.0),

            (-4.0, 2.0),

            (-6.0, 1.0)

        ]

        #################################

        self.goal_pub = self.create_publisher(

            PoseStamped,

            "/goal_pose",

            10

        )

        #################################

        self.done_pub = self.create_publisher(

            Bool,

            "/episode_done",

            10

        )

        #################################

        self.create_subscription(

            Float32MultiArray,

            "/ppo_state",

            self.state_callback,

            10

        )

        #################################

        self.get_logger().info(

            "Episode Manager Started"

        )

        self.reset_episode()

    ##################################################

    def state_callback(

            self,

            msg

    ):

        if self.resetting:

            return

        state = list(msg.data)

        if len(state) < 2:

            return

        self.step_count += 1

        self.goal_distance = float(

            state[-2]

        )

        self.collision = (

            state[-1] > 0.5

        )

        if self.is_done():

            done = Bool()

            done.data = True

            self.done_pub.publish(

                done

            )

            self.reset_episode()

    ##################################################

    def is_done(

            self

    ):

        if (

                self.goal_distance

                <

                self.goal_threshold

        ):

            self.get_logger().info(

                "Goal Reached"

            )

            return True

        if self.collision:

            self.get_logger().info(

                "Collision"

            )

            return True

        if (

                self.step_count

                >=

                self.max_steps

        ):

            self.get_logger().info(

                "Episode Timeout"

            )

            return True

        return False

    ##################################################

    def publish_goal(

            self,

            x,

            y

    ):

        goal = PoseStamped()

        goal.header.frame_id = "map"

        goal.pose.position.x = x

        goal.pose.position.y = y

        goal.pose.position.z = 0.0

        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(

            goal

        )

        self.get_logger().info(

            f"Goal Published : ({x:.2f}, {y:.2f})"

        )

    ##################################################

    def sample_goal(

            self

    ):

        return random.choice(

            self.goals

        )

    ##################################################

    def reset_robot(

            self

    ):

        try:

            subprocess.run(

                [

                    "gz",

                    "service",

                    "-s",

                    "/world/smart_hospital/control",

                    "--reqtype",

                    "gz.msgs.WorldControl",

                    "--reptype",

                    "gz.msgs.Boolean",

                    "--timeout",

                    "3000",

                    "--req",

                    "reset:{all:true}"

                ],

                check=True

            )

            self.get_logger().info(

                "Gazebo Reset"

            )

        except Exception as e:

            self.get_logger().error(

                f"Reset Failed : {e}"

            )

    ##################################################

    def reset_humans(

            self

    ):

        self.get_logger().info(

            "Human Reset"

        )

    ##################################################

    def reset_episode(

            self

    ):

        self.resetting = True

        self.episode += 1

        self.step_count = 0

        self.goal_distance = 999.0

        self.collision = False

        self.reset_robot()

        time.sleep(

            1.0

        )

        self.reset_humans()

        gx, gy = self.sample_goal()

        self.publish_goal(

            gx,

            gy

        )

        self.get_logger().info(

            f"Episode {self.episode}"

        )

        self.get_logger().info(

            "Episode Reset Complete"

        )

        self.resetting = False


########################################################

def main(args=None):

    rclpy.init(args=args)

    node = EpisodeManager()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
