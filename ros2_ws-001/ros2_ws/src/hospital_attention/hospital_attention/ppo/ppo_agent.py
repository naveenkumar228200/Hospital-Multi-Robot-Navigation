import torch
import torch.nn as nn
import torch.optim as optim
from .ppo_network import PPO
from .ppo_buffer import PPOBuffer

#from hospital_attention.ppo.ppo_network import PPO
#from hospital_attention.ppo.ppo_buffer import PPOBuffer


class PPOAgent:

    def __init__(self):

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        self.model = PPO().to(

            self.device

        )

        self.optimizer = optim.Adam(

            self.model.parameters(),

            lr=3e-4

        )

        self.buffer = PPOBuffer()

        self.gamma = 0.99

        self.clip = 0.2

        self.value_coef = 0.5

        self.entropy_coef = 0.01

        self.epochs = 10

        self.batch_size = 64

    ################################################

    def select_action(

            self,

            state

    ):

        state = torch.tensor(

            state,

            dtype=torch.float32

        ).unsqueeze(

            0

        ).to(

            self.device

        )

        with torch.no_grad():

            mean, value = self.model(

                state

            )

        std = torch.ones_like(

            mean

        ) * 0.2

        dist = torch.distributions.Normal(

            mean,

            std

        )

        action = dist.sample()

        log_prob = dist.log_prob(

            action

        ).sum(

            dim=-1

        )

        return (

            action.squeeze(

                0

            ).cpu().numpy(),

            log_prob.item(),

            value.item()

        )

    ################################################

    def compute_advantages(

            self,

            rewards,

            values,

            dones

    ):

        advantages = []

        returns = []

        gae = 0

        next_value = 0

        lam = 0.95

        for t in reversed(

                range(

                    len(

                        rewards

                    )

                )

        ):

            delta = (

                    rewards[t]

                    +

                    self.gamma

                    *

                    next_value

                    *

                    (

                        1 -

                        dones[t]

                    )

                    -

                    values[t]

            )

            gae = (

                    delta

                    +

                    self.gamma

                    *

                    lam

                    *

                    (

                        1 -

                        dones[t]

                    )

                    *

                    gae

            )

            advantages.insert(

                0,

                gae

            )

            next_value = values[t]

        for a, v in zip(

                advantages,

                values

        ):

            returns.append(

                a + v

            )

        return (

            advantages,

            returns

        )

    ################################################

    def update(self):

        data = self.buffer.get()

        states = data["states"]

        actions = data["actions"]

        rewards = data["rewards"]

        dones = data["dones"]

        old_probs = data["log_probs"]

        values = data["values"]

        advantages, returns = (

            self.compute_advantages(

                rewards,

                values,

                dones

            )

        )

        states = torch.tensor(

            states,

            dtype=torch.float32

        ).to(

            self.device

        )

        actions = torch.tensor(

            actions,

            dtype=torch.float32

        ).to(

            self.device

        )

        old_probs = torch.tensor(

            old_probs,

            dtype=torch.float32

        ).to(

            self.device

        )

        advantages = torch.tensor(

            advantages,

            dtype=torch.float32

        ).to(

            self.device

        )

        returns = torch.tensor(

            returns,

            dtype=torch.float32

        ).to(

            self.device

        )

        advantages = (

                advantages

                -

                advantages.mean()

        ) / (

                advantages.std()

                +

                1e-8

        )

        N = states.size(

            0

        )

        for _ in range(

                self.epochs

        ):

            indices = torch.randperm(

                N

            )

            for start in range(

                    0,

                    N,

                    self.batch_size

            ):

                end = (

                    start

                    +

                    self.batch_size

                )

                batch = indices[

                    start:end

                ]

                s = states[batch]

                a = actions[batch]

                op = old_probs[batch]

                adv = advantages[batch]

                ret = returns[batch]

                mean, val = self.model(

                    s

                )

                std = torch.ones_like(

                    mean

                ) * 0.2

                dist = (

                    torch.distributions.Normal(

                        mean,

                        std

                    )

                )

                log_probs = (

                    dist.log_prob(

                        a

                    ).sum(

                        dim=-1

                    )

                )

                entropy = (

                    dist.entropy(

                    ).mean()

                )

                ratio = torch.exp(

                    log_probs

                    -

                    op

                )

                surr1 = (

                        ratio

                        *

                        adv

                )

                surr2 = (

                        torch.clamp(

                            ratio,

                            1 -

                            self.clip,

                            1 +

                            self.clip

                        )

                        *

                        adv

                )

                actor_loss = (

                    -torch.min(

                        surr1,

                        surr2

                    ).mean()

                )

                critic_loss = (

                        (

                                ret

                                -

                                val.squeeze()

                        )

                        **

                        2

                ).mean()

                loss = (

                        actor_loss

                        +

                        self.value_coef

                        *

                        critic_loss

                        -

                        self.entropy_coef

                        *

                        entropy

                )

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    0.5

                )

                self.optimizer.step()

        self.buffer.clear()

    ################################################

    def save(

            self,

            path

    ):

        torch.save(

            {

                "model":

                    self.model.state_dict(),

                "optimizer":

                    self.optimizer.state_dict()

            },

            path

        )

    ################################################

    def load(

            self,

            path

    ):

        checkpoint = torch.load(

            path,

            map_location=self.device

        )

        self.model.load_state_dict(

            checkpoint[

                "model"

            ]

        )

        self.optimizer.load_state_dict(

            checkpoint[

                "optimizer"

            ]

        )

