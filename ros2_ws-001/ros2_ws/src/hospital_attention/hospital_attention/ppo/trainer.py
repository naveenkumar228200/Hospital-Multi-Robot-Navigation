import rclpy
from rclpy.node import Node

import numpy as np
import time
import torch

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool

from hospital_attention.ppo.ppo_agent import PPOAgent
from hospital_attention.ppo.reward_function import RewardFunction


class PPOTrainer(Node):

    def __init__(self):

        super().__init__(

            "ppo_collector"

        )

        ################################################

        self.agent = PPOAgent()

        self.reward_fn = RewardFunction()

        ################################################

        self.step = 0

        self.episode = 0

        self.step_in_episode = 0

        ################################################

        self.save_episode_interval = 10

        self.dataset_id = 0

        ################################################

        self.episode_reward = 0.0

        ################################################

        self.prev_goal_distance = None

        self.prev_action = None

        ################################################

        self.action_interval = 0.1

        self.last_action_time = time.time()

        ################################################

        self.action_pub = self.create_publisher(

            Float32MultiArray,

            "/ppo_action",

            10

        )

        ################################################

        self.create_subscription(

            Float32MultiArray,

            "/ppo_state",

            self.state_callback,

            10

        )

        ################################################

        self.create_subscription(

            Bool,

            "/episode_done",

            self.done_callback,

            10

        )

        ################################################

        self.get_logger().info(

            "Offline PPO Collector Started"

        )

    ################################################

    def state_callback(

            self,

            msg

    ):

        now = time.time()

        if (

                now -

                self.last_action_time

                <

                self.action_interval

        ):

            return

        self.last_action_time = now

        ################################################

        state = np.array(

            msg.data,

            dtype=np.float32

        )

        if len(state) != 160:

            return

        ################################################

        action, log_prob, value = (

            self.agent.select_action(

                state

            )

        )

        ################################################

        action_msg = Float32MultiArray()

        action_msg.data = (

            action.tolist()

        )

        self.action_pub.publish(

            action_msg

        )

        ################################################

        goal_distance = state[-2]

        collision = (

                state[-1]

                >

                0.5

        )

        ################################################

        if self.prev_goal_distance is None:

            self.prev_goal_distance = (

                goal_distance

            )

        ################################################

        reward = self.reward_fn.compute_reward(

            prev_goal_dist=

            self.prev_goal_distance,

            current_goal_dist=

            goal_distance,

            humans=[],

            linear_vel=

            float(action[0]),

            angular_vel=

            float(action[1]),

            previous_action=

            self.prev_action,

            collision=

            collision

        )

        ################################################

        self.agent.buffer.store(

            state,

            action,

            reward,

            value,

            log_prob,

            False

        )

        ################################################

        self.step += 1

        self.step_in_episode += 1

        self.episode_reward += reward

        ################################################

        self.prev_goal_distance = (

            goal_distance

        )

        self.prev_action = action

        ################################################

        self.get_logger().info(

            f"Episode : {self.episode}"

        )

        self.get_logger().info(

            f"Step : {self.step}"

        )

        self.get_logger().info(

            f"Reward : {reward:.3f}"

        )

    ################################################

    def done_callback(

            self,

            msg

    ):

        if msg.data:

            if len(

                    self.agent.buffer.dones

            ) > 0:

                self.agent.buffer.dones[-1] = True

            self.finish_episode()

    ################################################

    def finish_episode(

            self

    ):

        self.episode += 1

        self.step_in_episode = 0

        ################################################

        self.get_logger().info(

            f"Episode {self.episode}"

        )

        self.get_logger().info(

            f"Return = "

            f"{self.episode_reward:.2f}"

        )

        ################################################

        if (

                self.episode %

                self.save_episode_interval

                ==

                0

        ):

            self.save_dataset()

        ################################################

        stop = Float32MultiArray()

        stop.data = [

            0.0,

            0.0

        ]

        self.action_pub.publish(

            stop

        )

        ################################################

        self.prev_goal_distance = None

        self.prev_action = None

        self.episode_reward = 0.0

    ################################################

    def save_dataset(

            self

    ):

        if len(

                self.agent.buffer.states

        ) == 0:

            return

        ################################################

        filename = (

            "/home/lucifer/ros2_ws/"

            f"dataset_{self.dataset_id}.pth"

        )

        ################################################

        torch.save(

            {

                "states":

                    self.agent.buffer.states,

                "actions":

                    self.agent.buffer.actions,

                "rewards":

                    self.agent.buffer.rewards,

                "values":

                    self.agent.buffer.values,

                "log_probs":

                    self.agent.buffer.log_probs,

                "dones":

                    self.agent.buffer.dones

            },

            filename

        )

        ################################################

        self.get_logger().info(

            f"Saved "

            f"{filename}"

        )

        self.get_logger().info(

            f"Transitions : "

            f"{len(self.agent.buffer.states)}"

        )

        ################################################

        self.agent.buffer.clear()

        self.dataset_id += 1

####################################################


def main(args=None):

    rclpy.init(

        args=args

    )

    node = PPOTrainer()

    try:

        rclpy.spin(

            node

        )

    except KeyboardInterrupt:

        pass

    finally:

        node.save_dataset()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
