import math


class RewardFunction:

    def __init__(self):

        self.goal_threshold = 0.5

        self.social_distance = 1.0

        self.goal_bonus = 50.0

        self.collision_penalty = -50.0

        self.social_penalty = -5.0

        self.progress_scale = 5.0

        self.smoothness_scale = 0.1

        self.angular_penalty = 0.05

        self.alive_bonus = 0.01

    #####################################

    def compute_reward(

            self,

            prev_goal_dist,

            current_goal_dist,

            humans,

            linear_vel,

            angular_vel,

            previous_action,

            collision=False

    ):

        reward = 0.0

        ##################################
        # Progress reward
        ##################################

        progress = (

            prev_goal_dist

            -

            current_goal_dist

        )

        reward += (

            self.progress_scale

            *

            progress

        )

        ##################################
        # Goal reached
        ##################################

        if current_goal_dist < self.goal_threshold:

            reward += self.goal_bonus

        ##################################
        # Collision
        ##################################

        if collision:

            reward += self.collision_penalty

        ##################################
        # Human avoidance
        ##################################

        for human in humans:

            hx = human[0]

            hy = human[1]

            dist = math.sqrt(

                hx**2

                +

                hy**2

            )

            if dist < self.social_distance:

                reward += (

                    self.social_penalty

                    *

                    (

                        self.social_distance

                        -

                        dist

                    )

                )

        ##################################
        # Smooth motion
        ##################################

        if previous_action is not None:

            dv = abs(

                linear_vel

                -

                previous_action[0]

            )

            dw = abs(

                angular_vel

                -

                previous_action[1]

            )

            reward -= (

                self.smoothness_scale

                *

                (

                    dv

                    +

                    dw

                )

            )

        ##################################
        # Angular velocity penalty
        ##################################

        reward -= (

            self.angular_penalty

            *

            abs(

                angular_vel

            )

        )

        ##################################
        # Alive reward
        ##################################

        reward += self.alive_bonus

        return reward
