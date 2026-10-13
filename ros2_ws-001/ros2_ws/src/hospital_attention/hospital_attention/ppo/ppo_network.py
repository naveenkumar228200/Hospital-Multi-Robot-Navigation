
import torch
import torch.nn as nn


###############################################
# Actor Network
###############################################

class PPOActor(nn.Module):

    def __init__(

            self,

            state_dim=160,

            action_dim=2

    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(

                state_dim,

                256

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                256

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                128

            ),

            nn.ReLU()

        )

        self.mean = nn.Linear(

            128,

            action_dim

        )

        self.tanh = nn.Tanh()

        self._init_weights()

    ################################################

    def _init_weights(self):

        for m in self.modules():

            if isinstance(

                    m,

                    nn.Linear

            ):

                nn.init.orthogonal_(

                    m.weight,

                    gain=1.0

                )

                nn.init.zeros_(

                    m.bias

                )

    ################################################

    def forward(

            self,

            x

    ):

        features = self.net(

            x

        )

        mean = self.mean(

            features

        )

        mean = self.tanh(

            mean

        )

        return mean


###############################################
# Critic Network
###############################################

class PPOCritic(nn.Module):

    def __init__(

            self,

            state_dim=160

    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(

                state_dim,

                256

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                256

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                128

            ),

            nn.ReLU(),

            nn.Linear(

                128,

                1

            )

        )

        self._init_weights()

    ################################################

    def _init_weights(self):

        for m in self.modules():

            if isinstance(

                    m,

                    nn.Linear

            ):

                nn.init.orthogonal_(

                    m.weight,

                    gain=1.0

                )

                nn.init.zeros_(

                    m.bias

                )

    ################################################

    def forward(

            self,

            x

    ):

        value = self.net(

            x

        )

        return value


###############################################
# PPO Network
###############################################

class PPO(nn.Module):

    def __init__(

            self,

            state_dim=160,

            action_dim=2

    ):

        super().__init__()

        self.actor = PPOActor(

            state_dim,

            action_dim

        )

        self.critic = PPOCritic(

            state_dim

        )

    ################################################

    def forward(

            self,

            x

    ):

        mean = self.actor(

            x

        )

        value = self.critic(

            x

        )

        return (

            mean,

            value

        )


################################################

if __name__ == "__main__":

    model = PPO()

    state = torch.randn(

        1,

        160

    )

    mean, value = model(

        state

    )

    print()

    print(

        "Mean shape:",

        mean.shape

    )

    print(

        "Value shape:",

        value.shape

    )

    print()

    print(

        mean

    )

    print()

    print(

        value

    )

