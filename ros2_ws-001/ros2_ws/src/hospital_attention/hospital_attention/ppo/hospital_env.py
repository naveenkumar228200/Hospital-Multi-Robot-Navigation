
import rclpy

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from hospital_attention.ppo.reward_function import RewardFunction


class HospitalEnv:

    def __init__(

            self,

            node

    ):

        self.node = node

        self.state = None

        self.done = False

        self.prev_goal = None

        self.prev_action = None

        self.reward_fn = RewardFunction()

        #################################

        self.cmd_pub = (

            self.node.create_publisher(

                Twist,

                "/cmd_vel",

                10

            )

        )

        #################################

        self.node.create_subscription(

            Float32MultiArray,

            "/ppo_state",

            self.state_callback,

            10

        )

    ################################################

    def state_callback(

            self,

            msg

    ):

        self.state = list(

            msg.data

        )

    ################################################

    def reset(

            self

    ):

        self.done = False

        self.state = None

        self.prev_goal = None

        self.prev_action = None

        timeout = 100

        count = 0

        while (

                self.state is None

                and

                count < timeout

        ):

            rclpy.spin_once(

                self.node,

                timeout_sec=0.1

            )

            count += 1

        return self.state

    ################################################

    def step(

            self,

            action

    ):

        cmd = Twist()

        cmd.linear.x = float(

            action[0]

        )

        cmd.angular.z = float(

            action[1]

        )

        self.cmd_pub.publish(

            cmd

        )

        #################################

        rclpy.spin_once(

            self.node,

            timeout_sec=0.05

        )

        #################################

        next_state = self.state

        goal_dist = next_state[-2]

        collision = (

                next_state[-1]

                >

                0.5

        )

        #################################

        if self.prev_goal is None:

            self.prev_goal = goal_dist

        #################################

        reward = self.reward_fn.compute_reward(

            prev_goal_dist=

            self.prev_goal,

            current_goal_dist=

            goal_dist,

            humans=[],

            linear_vel=

            float(

                action[0]

            ),

            angular_vel=

            float(

                action[1]

            ),

            previous_action=

            self.prev_action,

            collision=

            collision

        )

        #################################

        done = self.is_done(

            next_state

        )

        #################################

        self.prev_goal = goal_dist

        self.prev_action = action

        #################################

        info = {

            "goal_distance":

                goal_dist,

            "collision":

                collision

        }

        return (

            next_state,

            reward,

            done,

            info

        )

    ################################################

    def is_done(

            self,

            state

    ):

        if state is None:

            return False

        goal_distance = state[-2]

        collision = state[-1]

        if goal_distance < 0.5:

            return True

        if collision > 0.5:

            return True

        return False

    ################################################

    def stop_robot(

            self

    ):

        cmd = Twist()

        self.cmd_pub.publish(

            cmd

        )


