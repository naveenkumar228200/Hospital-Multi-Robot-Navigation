import glob
import torch

from hospital_attention.ppo.ppo_agent import PPOAgent


agent = PPOAgent()

files = sorted(

    glob.glob(

        "/home/lucifer/ros2_ws/dataset_*.pth"

    )

)

print(

    "Datasets found:",

    len(files)

)

if len(files) == 0:

    print(

        "No datasets found."

    )

    exit()

##################################################

for file in files:

    print(

        "Loading",

        file

    )

    data = torch.load(

        file,

        weights_only=False

    )

    for i in range(

            len(

                data["states"]

            )

    ):

        agent.buffer.store(

            data["states"][i],

            data["actions"][i],

            data["rewards"][i],

            data["values"][i],

            data["log_probs"][i],

            data["dones"][i]

        )

##################################################

print()

print(

    "Transitions collected:",

    len(

        agent.buffer.states

    )

)

print()

if len(agent.buffer.states) == 0:

    print(

        "Buffer Empty."

    )

    exit()

##################################################

agent.update()

##################################################

agent.save(

    "ppo_checkpoint.pth"

)

agent.save(

    "best_model.pth"

)

##################################################

print()

print(

    "Training Finished"

)

print(

    "Checkpoint Saved"

)
