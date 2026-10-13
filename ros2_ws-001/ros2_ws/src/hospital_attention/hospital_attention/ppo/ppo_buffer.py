import numpy as np


class PPOBuffer:

    def __init__(self):

        self.clear()

    ##################################################

    def clear(self):

        self.states = []

        self.actions = []

        self.rewards = []

        self.values = []

        self.log_probs = []

        self.dones = []

    ##################################################

    def store(

            self,

            state,

            action,

            reward,

            value,

            log_prob,

            done

    ):

        self.states.append(

            np.asarray(

                state,

                dtype=np.float32

            )

        )

        self.actions.append(

            np.asarray(

                action,

                dtype=np.float32

            )

        )

        self.rewards.append(

            float(reward)

        )

        self.values.append(

            float(value)

        )

        self.log_probs.append(

            float(log_prob)

        )

        self.dones.append(

            float(done)

        )

    ##################################################

    def get(self):

        return {

            "states":

                np.asarray(

                    self.states,

                    dtype=np.float32

                ),

            "actions":

                np.asarray(

                    self.actions,

                    dtype=np.float32

                ),

            "rewards":

                np.asarray(

                    self.rewards,

                    dtype=np.float32

                ),

            "values":

                np.asarray(

                    self.values,

                    dtype=np.float32

                ),

            "log_probs":

                np.asarray(

                    self.log_probs,

                    dtype=np.float32

                ),

            "dones":

                np.asarray(

                    self.dones,

                    dtype=np.float32

                )

        }

    ##################################################

    def size(self):

        return len(

            self.states

        )

    ##################################################

    def __len__(self):

        return len(

            self.states

        )
