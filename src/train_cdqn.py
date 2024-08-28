import math
import random
import os
import numpy as np
import subprocess
import pandas as pd

from tqdm import tqdm
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cdqn_env import CentralizedElastisityEnv


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, num_agents, num_actions_per_agent):
        super(DQN, self).__init__()
        self.num_agents = num_agents
        self.num_actions_per_agent = num_actions_per_agent
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_agents * num_actions_per_agent)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x.view(-1, self.num_agents, self.num_actions_per_agent)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
    state_batch = torch.stack(batch.state)
    # action_batch = torch.stack([torch.tensor(a).unsqueeze(0) for a in batch.action]) # deprecated
    action_batch = torch.stack([a.clone().detach().unsqueeze(0) for a in batch.action])
    reward_batch = torch.stack(batch.reward)

    state_action_values = policy_net(state_batch).gather(2, action_batch.squeeze().view(BATCH_SIZE, n_agents, 1))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(2)[0].detach().max(1)[0]

    expected_state_action_values = (next_state_values.unsqueeze(1).unsqueeze(2) * GAMMA) + reward_batch.unsqueeze(1).unsqueeze(2)
    # expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch.unsqueeze(1)

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy_net(state).max(1).indices.tolist()
            return torch.tensor([action], device=device, dtype=torch.long)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.15
    EPS_DECAY = 3000
    TAU = 0.005
    LR = 1e-4

    EPISODES = 400
    MEMORY_SIZE = 500

    MODEL = f'cdqn{EPISODES}ep{MEMORY_SIZE}m'
    os.makedirs(f'code/model_metric_data/{MODEL}', exist_ok=True)

    LOAD_WEIGHTS = False
    SAVE_WEIGHTS = True

    n_agents = 3
    n_actions_per_agent = 3

    env = CentralizedElastisityEnv(n_agents)

    state = env.reset()
    n_observations = len(state) * len(state[0])

    policy_net = DQN(n_observations, n_agents, n_actions_per_agent).to(device)

    if os.path.isfile('code/model_metric_data/{MODEL}/model_weights.pth') and LOAD_WEIGHTS:
        policy_net.load_state_dict(torch.load('code/model_metric_data/{MODEL}/model_weights.pth'))
    else:
        print("No weight file found, starting training from scratch.")

    target_net = DQN(n_observations, n_agents, n_actions_per_agent).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    summed_rewards = []
    latencies = []

    spam_process = subprocess.Popen(['python', 'code/spam_cluster.py', '--users', '400'])


    for i_episode in tqdm(range(EPISODES)):
        # Initialize the environment and get its state
        state = env.reset()
        state = torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        latency = []
        for t in count():
            action = select_action(state)
            # print(action.flatten().tolist())
            observation, reward, done, info = env.step(action.flatten().tolist())
            reward = torch.tensor([reward], device=device)
            total_reward += reward.item()
            latency.append(env.calculate_latency(1))

            if done:
                next_state = None
            else:
                next_state = torch.tensor(np.array(observation).flatten(), dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break
        summed_rewards.append(total_reward)
        latencies.append(np.mean(latency))
        print(f"Episode {i_episode} reward: {np.mean(summed_rewards)}, latenices: {np.mean(latencies)}")

    print(f'Complete with {np.mean(summed_rewards)} summed_rewards and {np.mean(latencies)} latency')
    spam_process.terminate()

    if SAVE_WEIGHTS:
        torch.save(policy_net.state_dict(),  f'code/model_metric_data/{MODEL}/model_weights.pth')

        # save collected data for later analysis
        ep_summed_rewards_df = pd.DataFrame({'Episode': range(len(summed_rewards)), 'Reward': summed_rewards})
        ep_summed_rewards_df.to_csv(f'code/model_metric_data/{MODEL}/ep_summed_rewards.csv', index=False)

        ep_latencies_df = pd.DataFrame({'Episode': range(len(latencies)), 'Mean Latency': latencies})
        ep_latencies_df.to_csv(f'code/model_metric_data/{MODEL}/ep_latencies.csv', index=False)
