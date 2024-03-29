import math
import random
import time
import numpy as np
import subprocess
import os

from tqdm import tqdm
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from env import ElastisityEnv
import pandas as pd


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

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

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
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
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

def select_action(state, policy_net, env):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if steps_done % 100 == 0:
        print(f"eps_threshold: {eps_threshold} at step {steps_done}")
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
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
    # EPS_END = 0.25
    EPS_END = 0.15
    EPS_DECAY = 3000
    TAU = 0.005
    LR = 1e-4

    # MEMORY_SIZE = 1000
    MEMORY_SIZE = 500
    EPISODES = 400

    MODEL = f'mdqn{EPISODES}ep{MEMORY_SIZE}m'
    os.makedirs(f'code/model_metric_data/{MODEL}', exist_ok=True)

    LOAD_WEIGHTS = False
    SAVE_WEIGHTS = True

    # shared reward weight
    alpha = 0.7

    n_agents = 3
    envs = [ElastisityEnv(i) for i in range(1, n_agents + 1)]
    # Get number of actions from gym action space
    n_actions = envs[0].action_space.n
    # Get the number of state observations
    state = envs[0].reset()
    n_observations = len(state) * len(state[0])
    print(f"Number of observations: {n_observations} and number of actions: {n_actions}")

    agents = [DQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
    if LOAD_WEIGHTS:
        for i, agent in enumerate(agents):
            agent.load_state_dict(torch.load(f'code/model_metric_data/{MODEL}/model_weights_agent_{i}.pth'))
        print(f"Loaded weights for agents")

    target_nets = [DQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
    memories = [ReplayMemory(MEMORY_SIZE) for _ in range(n_agents)]
    optimizers = [optim.AdamW(agent.parameters(), lr=LR, amsgrad=True) for agent in agents]

    for target_net, agent in zip(target_nets, agents):
        target_net.load_state_dict(agent.state_dict())

    steps_done = 0
    ep_summed_rewards = []
    ep_latencies = []
    agent_ep_summed_rewards = [[] for _ in range(n_agents)]

    # load the cluster
    spam_process = subprocess.Popen(['python', 'code/spam_cluster.py', '--users', '400'])

    for i_episode in tqdm(range(EPISODES)):
        # save weights every 5 episodes
        if i_episode % 5 == 0 and SAVE_WEIGHTS:
            for i, agent in enumerate(agents):
                torch.save(agent.state_dict(), f'code/model_metric_data/{MODEL}/model_weights_agent_{i}.pth')
                print(f"Checkpoint: Saved weights for agent {i}")

        states = [env.reset() for env in envs]
        states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0) for state in states]

        step_rewards = 0
        step_latencies = []
        agents_step_reward = [[] for _ in range(n_agents)]
        for t in count():
            time.sleep(1)

            actions = [select_action(state, agent, env) for state, agent, env in zip(states, agents, envs)]

            next_states, rewards, dones = [], [], []
            for i, action in enumerate(actions):
                # reward for efficiency
                observation, reward, done, _ = envs[i].step(action.item())
                next_states.append(np.array(observation).flatten())
                rewards.append(reward)
                dones.append(done)
                if done:
                    next_states[i] = None

            # latencies = [env.calculate_latency() for env in envs]
            # latenices_for_agent.append(latencies[0])
            # print(latencies)
            # shared_rewards = [alpha * reward + beta * (1 - np.mean(latencies) * 10) for reward in rewards]
            
            # calculate mean/geomean latency for the system, it doesnt matter which agent we use
            latency = envs[0].calculate_latency(30)
            step_latencies.append(envs[0].calculate_latency(1)) # without geo. mean
            # shared_rewards = [alpha * reward + beta * (1 - latency * 100) for reward in rewards]
            shared_rewards = [alpha * reward + (1 - alpha) * (1 - latency * 100) for reward in rewards]

            if t % 25 == 0:
                print(f"SharedR A*r+B*L: {shared_rewards}, reward_part: {rewards}, latency_part: {latency}. Step: {t}")

            [agents_step_reward[i].append(shared_rewards[i]) for i in range(n_agents)]

            # each agent has its own latency, so it isnt a shared reward
            # shared_rewards = [alpha * reward + beta * (1 - latency * 10) for reward, latency in zip(rewards, latencies)]

            # print(f"Rewards: {shared_rewards}, step: {t}")
            step_rewards += np.mean(shared_rewards)

            next_states = [torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if observation is not None else None for observation in next_states]
            rewards = [torch.tensor([reward], device=device) for reward in shared_rewards]

            for i in range(n_agents):
                memories[i].push(states[i], actions[i], next_states[i], rewards[i])

            states = next_states
            for i in range(n_agents):
                optimize_model(agents[i], target_nets[i], memories[i], optimizers[i])
            
            for agent, target_net in zip(agents, target_nets):
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = agent.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

            if any(dones):
                break
        
        ep_latencies.append(np.mean(step_latencies))
        ep_summed_rewards.append(step_rewards)
        
        [agent_ep_summed_rewards[i].append(np.sum(reward)) for i, reward in enumerate(agents_step_reward)]
        print(f"Episode {i_episode} reward: {step_rewards} mean latency: {np.mean(step_latencies)}")

    spam_process.terminate()
    print(f'Complete with {np.mean(ep_summed_rewards)} rewards')

    if SAVE_WEIGHTS:
        for i, agent in enumerate(agents):
            torch.save(agent.state_dict(), f'code/model_metric_data/{MODEL}/model_weights_agent_{i}.pth')
    
        # save collected data for later analysis
        ep_summed_rewards_df = pd.DataFrame({'Episode': range(len(ep_summed_rewards)), 'Reward': ep_summed_rewards})
        ep_summed_rewards_df.to_csv(f'code/model_metric_data/{MODEL}/ep_summed_rewards.csv', index=False)

        ep_latencies_df = pd.DataFrame({'Episode': range(len(ep_latencies)), 'Mean Latency': ep_latencies})
        ep_latencies_df.to_csv(f'code/model_metric_data/{MODEL}/ep_latencies.csv', index=False)

        for agent_idx, rewards in enumerate(agent_ep_summed_rewards):
            filename = f'code/model_metric_data/{MODEL}/agent_{agent_idx}_ep_summed_rewards.csv'
            agent_rewards_df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
            agent_rewards_df.to_csv(filename, index=False)
