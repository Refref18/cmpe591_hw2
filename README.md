# Deep Q Network (DQN) for Object Pushing Task

## Overview

In this homework, a Deep Q Network (DQN) is trained to push an object to a desired position. The reinforcement learning agent interacts with the environment and learns to maximize the total reward through trial and error.

## Network Structure

- **Input Size**: 6 (high level state representation)
- **Hidden Layers**:
  - Layer 1: Fully connected (Linear) with 64 units, ReLU activation
- **Output Size**: 8 (corresponding to the number of discrete actions)

## Parameters

- **Number of Episodes**: 10,000 (couldn't run after 6700th episode due to time and computational restrictions)
- **Batch Size**: 64
- **Gamma (Discount Factor)**: 0.99
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Target Update Rate (Tau)**: 0.005
- **Learning Rate**: 1e-4
- **Replay Buffer Size**: 100,000
- **Update Frequency**: 10

## Results

Below are the plots for total reward per episode and reward per step:

### Total Reward per Episode

![Total Reward per Episode](plots/total_reward_plot_2025-03-05_11-37-34.png)

After the 1000th episode, the model starts learning, and from the 1500th episode onward, the total reward steadily increases, indicating stable improvement.

### Reward per Step

![Reward per Step](plots/reward_plot_2025-03-05_11-37-34.png)

## Additional Files

- original script that created the .txt files and the model is **`hw2.py`**. In **`hw2.py`** the train() and test() methods are created for readability.
- **`reward_per_step_v2.txt`** and **`total_reward_per_episode_v2.txt`** contain the reward values recorded during training.
- **`plot.py`** and **`plot_2.py`** contain the scripts used to generate the reward plots.
- All plots that are created during training are in **`plots`** folder
- The training took more than 20 hours and unfortunately in the end it could't reach the 10000th episode, so the plateou was not observed.
- The model is saved in **`dqn_model.pth`**.
