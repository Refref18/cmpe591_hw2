import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from homework2 import Hw2Env
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

N_ACTIONS = 8
env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(device)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    """def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state = np.array([x[0] for x in batch])
        action = np.array([x[1] for x in batch])
        reward = np.array([x[2] for x in batch])
        next_state = np.array([x[3] for x in batch])
        done = np.array([x[4] for x in batch])
        return (torch.FloatTensor(state), 
                torch.LongTensor(action), 
                torch.FloatTensor(reward), 
                torch.FloatTensor(next_state), 
                torch.FloatTensor(done))"""
    
    def sample(self, batch_size):
        """ NumPy değil, doğrudan PyTorch Tensor olarak döndür """
        batch = random.sample(self.memory, batch_size)

        # State ve next_state tuple olarak kalacak!
        state = tuple(x[0] for x in batch)
        #action = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device).view(-1,1).unsqueeze(1)
        #reward = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device).unsqueeze(1)

        # ✅ Action tuple içinde tensor([[value]]) formatında olacak
        action = tuple(torch.tensor([[x[1]]], dtype=torch.long, device=device) for x in batch)

        # ✅ Reward tuple içinde tensor([[value]]) formatında olacak
        reward = tuple(torch.tensor([x[2]], dtype=torch.float32, device=device) for x in batch)
        next_state = tuple(x[3] for x in batch)
        done = tuple(torch.tensor([[x[4]]], dtype=torch.bool, device=device) for x in batch)

        # Non-final mask (None olmayanları ayır)
        #non_final_mask = torch.tensor([x[3] is not None for x in batch], dtype=torch.bool, device=device)
        #non_final_states = tuple(x[3] for x in batch if x[3] is not None)

        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)
    


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, n_actions)
       

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)
        
    

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TAU = 0.005 #target_update_frequency = 200
LR = 1e-4


"""
+ num_episodes = 10000
update_frequency = 10
+ target_update_frequency = 200
+ epsilon = 1.0
+ epsilon_decay = 0.995
+ epsilon_min = 0.05
+ batch_size=64
+ gamma=0.99
+ buffer_size=100000

"""
# Get number of actions from gym action space
n_actions = 8
# Get the number of state observations
n_observations = 6
print(n_observations, n_actions)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
buffer_size = 100000
memory = ReplayMemory(buffer_size)


steps_done = 0

epsilon= EPS_START
def select_action(state):
    global epsilon
    sample = random.random()
    #eps_threshold = max(EPS_END, EPS_START * (EPS_DECAY ** steps_done))

    """eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)"""
    #steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    state, action, reward, next_state, done = transitions
    """print("state", len(state))
    print("action", len(action))
    print("reward", len(reward))
    print("next_state", len(next_state))
    
    print('state', (state[0]))
    print('acion', (action[0]))
    print('next state', (next_state[0]))
    print('reward', (reward[0]))"""

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in next_state if s is not None])

    #print("state", type(state))
    #print("action", type(action))
    #print("reward", type(reward))
    #print("next_state", type(next_state))

    state_batch = torch.cat(state)
    action_batch = torch.cat(action)
    reward_batch = torch.cat(reward)
    
    #print("non_final_mask", non_final_mask.size())
    #print("non_final_next_states", non_final_next_states.size())
    #print("state_batch", state_batch.size())
    #print("action_batch", action_batch.size())
    #print("reward_batch", reward_batch.size())

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if non_final_next_states.size(0) > 0:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 10000
else:
    num_episodes = 10000


reward_per_step_file = "reward_per_step.txt"
total_reward_file = "total_reward_per_episode.txt"


open(reward_per_step_file, "w").close()
open(total_reward_file, "w").close()

#TODO kaç episodda bir epsilon güncellenecek
#yiğit hocaın parametreleri

update_frequency = 10

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    env.reset()
    state = torch.tensor(env.high_level_state(), dtype=torch.float32, device=device).unsqueeze(0)
    #print("state", state.size())
    done=False
    episode_durations = []
    total_reward = 0  # 🔹 Total reward'ı tutmak için
    step = 0  # 🔹 Adım sayacı
    for t in count():
        action = select_action(state)
        #print("action", action.size())
        observation, reward, terminated, truncated = env.step(action.item())

        #print("obs", observation)
        #print("reward", reward)
        #print("terminated", terminated)
        #print("truncated", truncated)

        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

    
        total_reward += reward.item()  # 🔹 Toplam ödülü güncelle
        #print("reward", reward)
       
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        #print("next_state", next_state.size())

        # Store the transition in memory
        memory.push(state, action, reward, next_state, done)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if step % update_frequency == 0:
            optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        with open(reward_per_step_file, "a") as f:
            f.write(f"{step},{reward.item()}\n")
        
        step += 1  # Adım sayısını artır

        if done:
            episode_durations.append(t + 1)
            
            # Ortalama ödül hesapla (adım başına ödül)
            average_reward = total_reward / step if step > 0 else 0
            
            # 🔹 **Epsilon güncelleme** (her episode sonunda düşüş)
            epsilon = max(EPS_END, epsilon * EPS_DECAY)
            # Şu anki epsilon değeri
            current_epsilon = epsilon

            # Sonuçları ekrana yazdır
            print(f"Episode: {i_episode} | Total Reward: {total_reward:.2f} | Average Reward: {average_reward:.4f} | Epsilon: {current_epsilon:.6f}")

            # Total reward'u dosyaya kaydet
            with open(total_reward_file, "a") as f:
                f.write(f"{i_episode},{total_reward}\n")

            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
