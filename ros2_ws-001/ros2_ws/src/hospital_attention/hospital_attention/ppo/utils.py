
import torch
import numpy as np


###################################################
# Normalize Advantages
###################################################

def normalize(x):

    x = np.asarray(

        x,

        dtype=np.float32

    )

    return (

        x - np.mean(x)

    ) / (

        np.std(x)

        +

        1e-8

    )


###################################################
# Discounted Returns
###################################################

def discount_rewards(

        rewards,

        dones=None,

        gamma=0.99

):

    returns = []

    R = 0

    if dones is None:

        dones = [

            False

        ] * len(

            rewards

        )

    for reward, done in zip(

            reversed(

                rewards

            ),

            reversed(

                dones

            )

    ):

        if done:

            R = 0

        R = (

                reward

                +

                gamma

                *

                R

        )

        returns.insert(

            0,

            R

        )

    return returns


###################################################
# Generalized Advantage Estimation
###################################################

def compute_gae(

        rewards,

        values,

        dones,

        gamma=0.99,

        lam=0.95

):

    advantages = []

    gae = 0

    values = list(

        values

    )

    values.append(

        0

    )

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

                gamma

                *

                values[t + 1]

                *

                (

                    1

                    -

                    dones[t]

                )

                -

                values[t]

        )

        gae = (

                delta

                +

                gamma

                *

                lam

                *

                (

                    1

                    -

                    dones[t]

                )

                *

                gae

        )

        advantages.insert(

            0,

            gae

        )

    returns = [

        adv + val

        for adv, val

        in zip(

            advantages,

            values[:-1]

        )

    ]

    advantages = normalize(

        advantages

    )

    return (

        advantages,

        returns

    )


###################################################
# Save Model
###################################################

def save_model(

        model,

        path

):

    torch.save(

        model.state_dict(),

        path

    )


###################################################
# Load Model
###################################################

def load_model(

        model,

        path,

        device="cpu"

):

    with torch.no_grad():

        model.load_state_dict(

            torch.load(

                path,

                map_location=device

            )

        )

    model.eval()

    return model


###################################################
# Tensor Conversion
###################################################

def to_tensor(

        array,

        device="cpu"

):

    return torch.as_tensor(

        array,

        dtype=torch.float32,

        device=device

    )


###################################################
# Clip Action
###################################################

def clip_action(

        action,

        max_linear=0.5,

        max_angular=1.5

):

    linear = np.clip(

        action[0],

        0.0,

        max_linear

    )

    angular = np.clip(

        action[1],

        -max_angular,

        max_angular

    )

    return np.array(

        [

            linear,

            angular

        ],

        dtype=np.float32

    )


###################################################
# Test
###################################################

if __name__ == "__main__":

    action = np.array(

        [

            0.9,

            2.0

        ]

    )

    clipped = clip_action(

        action

    )

    print(

        clipped

    )

