import math
import random
import time
import numpy as np
import subprocess
import os
import argparse

from tqdm import tqdm
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import ElastisityEnv
import pandas as pd

from spam_cluster import spam_requests_single
from pod_controller import get_loadbalancer_external_port, set_container_cpu_values


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
        # self.layer1 = nn.Linear(n_observations, 128)
        # self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, n_actions)
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # return self.layer3(x)
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class DuelingDQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


def optimize_model(policy_net, target_net, memory, optimizer): #, scheduler):
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
        if double:
            next_state_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)
        else:
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
    # scheduler.step()


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


def set_available_resource(envs, initial_resources):
    max_group = initial_resources
    for env in envs:
        max_group -= env.ALLOCATED
    for env in envs:
        env.AVAILABLE = max_group


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
    EPS_DECAY = 8_000
    TAU = 0.005
    LR = 1e-4

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--init_resources', type=int, default=500)
    parser.add_argument('--increment_action', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=0.4, help="Weight for the shared reward, higher the more weight to latency, lower the more weight to efficiency")
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--rps', type=int, default=50, help="Requests per second for loading cluster")
    parser.add_argument('--random_rps', type=bool, default=False, help="Train on random requests every episode")
    parser.add_argument('--interval', type=int, default=1000, help="Milliseconds interval for requests")
    parser.add_argument('--dueling', type=bool, default=False, help="Dueling rl")
    parser.add_argument('--double', type=bool, default=False, help="Double rl")
    parser.add_argument('--load_weights', type=bool, default=False, help="Load weights from previous training")
    parser.add_argument('--variable_resources', type=bool, default=False, help="Random resources every 10 episodes")
    args = parser.parse_args()

    double = args.double
    dueling = args.dueling

    reqs_per_second = args.rps
    interval = args.interval
    randomize_reqs = args.random_rps
    variable_resources = args.variable_resources

    # MEMORY_SIZE = 1000
    MEMORY_SIZE = 500
    EPISODES = args.episodes

    LOAD_WEIGHTS = args.load_weights
    SAVE_WEIGHTS = True

    # env values
    RESOURCES = args.init_resources
    INCREMENT_ACTION = args.increment_action
    USERS = 10
    reqs_per_second -= USERS # interval is set to 1s

    # shared reward weight
    alpha = args.alpha

    set_container_cpu_values(cpus=100)

    MODEL = f'mdqn{EPISODES}ep{MEMORY_SIZE}m{INCREMENT_ACTION}inc{RESOURCES}mcmax{reqs_per_second}rps{interval}interval{alpha}alpha'
    if double:
        MODEL += '_double'
    if dueling:
        MODEL += '_dueling'
    os.makedirs(f'code/model_metric_data/{MODEL}', exist_ok=True)

    n_agents = args.n_agents
    envs = [ElastisityEnv(i, n_agents) for i in range(1, n_agents + 1)]
    for env in envs:
        env.MAX_CPU_LIMIT = RESOURCES
        env.INCREMENT = INCREMENT_ACTION

    # Get number of actions from gym action space
    n_actions = envs[0].action_space.n
    # Get the number of state observations
    state = envs[0].reset()
    n_observations = len(state) * len(state[0])
    print(f"Number of observations: {n_observations} and number of actions: {n_actions}")

    # init envs
    set_available_resource(envs, RESOURCES)

    # create networks
    if dueling:
        agents = [DuelingDQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
        target_nets = [DuelingDQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
    else:
        agents = [DQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
        target_nets = [DQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
    
    if LOAD_WEIGHTS:
        for i, agent in enumerate(agents):
            agent.load_state_dict(torch.load(f'code/model_metric_data/{MODEL}/model_weights_agent_{i}.pth'))
        print(f"Loaded weights for agents")

    memories = [ReplayMemory(MEMORY_SIZE) for _ in range(n_agents)]
    optimizers = [optim.AdamW(agent.parameters(), lr=LR, amsgrad=True) for agent in agents]
    # schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) for optimizer in optimizers]

    for target_net, agent in zip(target_nets, agents):
        target_net.load_state_dict(agent.state_dict())

    steps_done = 0
    summed_rewards = []
    mean_latencies = []
    agent_ep_summed_rewards = [[] for _ in range(n_agents)]
    resource_dev = []

    # url_spam = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}/predict"
    url_spam = f"http://localhost:30888/predict"

    for i_episode in tqdm(range(EPISODES)):
        if i_episode % 5 == 0 and i_episode != 0 and SAVE_WEIGHTS:
            for i, agent in enumerate(agents):
                torch.save(agent.state_dict(), f'code/model_metric_data/{MODEL}/model_weights_agent_{i}.pth')
                print(f"Checkpoint: Saved weights for agent {i}")
        if variable_resources and i_episode % 10 == 0 and i_episode != 0:
            RESOURCES = random.choice([500, 750, 1000, 1250, 1500, 1750, 2000])
            for env in envs:
                env.MAX_CPU_LIMIT = RESOURCES
            print(f"Resources changed to {RESOURCES} for episode {i_episode}")

        # randomize the requests per second to get rid of bias
        random_rps = np.random.randint(5, reqs_per_second) if randomize_reqs else reqs_per_second
        
        # can overfill, so we reset the loading process on every episode
        spam_process = subprocess.Popen(['python', 'code/spam_cluster.py', '--users', str(random_rps), '--interval', str(interval)])
        print(f"Loading the cluster with {random_rps} requests/second")
        time.sleep(1) # for the limits to be set
        states = [env.reset() for env in envs]
        set_available_resource(envs, RESOURCES)
        states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0) for state in states]

        ep_rewards = 0
        ep_latencies = []
        agents_ep_reward = [[] for _ in range(n_agents)]
        ep_std = []
        for t in count():
            time.sleep(1)
            latencies = spam_requests_single(USERS, url_spam)

            actions = [select_action(state, agent, env) for state, agent, env in zip(states, agents, envs)]

            next_states, rewards, dones = [], [], []
            resources = []
            for i, action in enumerate(actions):
                # reward for efficiency
                observation, reward, done, _ = envs[i].step(action.item())
                set_available_resource(envs, RESOURCES) # heavy
                next_states.append(np.array(observation).flatten())
                rewards.append(reward)
                dones.append(done)
                resources.append(envs[i].ALLOCATED)
                if done:
                    next_states[i] = None

            latency = np.mean([latency for latency in latencies if latency is not None])
            ep_latencies.append(latency)
            resource_std_dev = np.std(resources) / 500
            ep_std.append(resource_std_dev)
            # shared_rewards = [alpha * reward + (1 - alpha) * (1 - latency * 10) for reward in rewards]
            shared_rewards = [alpha * reward + (1 - alpha) * (1 - latency * 10) - resource_std_dev for reward in rewards]

            if t % 25 == 0 and t != 0:
                print(f"SharedR A*r+B*L: {shared_rewards}, reward_part: {rewards}, latency_part: {latency}, resource deviation: {resource_std_dev}. Step: {t}")
                for env in envs:
                    print(f"Agent {env.id}: {env.last_cpu_percentage} % CPU, {env.AVAILABLE} available CPU", end=" ")
                print()

            [agents_ep_reward[i].append(shared_rewards[i]) for i in range(n_agents)]

            ep_rewards += np.mean(shared_rewards)

            next_states = [torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if observation is not None else None for observation in next_states]
            shared_reward_tensors = [torch.tensor([shared_reward], device=device) for shared_reward in shared_rewards]

            for i in range(n_agents):
                memories[i].push(states[i], actions[i], next_states[i], shared_reward_tensors[i])

            states = next_states
            for i in range(n_agents):
                optimize_model(agents[i], target_nets[i], memories[i], optimizers[i]) #, schedulers[i])
            
            for agent, target_net in zip(agents, target_nets):
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = agent.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

            if any(dones):
                for env in envs:
                    env.save_last_limit()
                break
        
        mean_latencies.append(np.mean(ep_latencies))
        summed_rewards.append(ep_rewards)
        resource_dev.append(np.mean(ep_std))
        
        [agent_ep_summed_rewards[i].append(np.sum(reward)) for i, reward in enumerate(agents_ep_reward)]
        print(f"Episode {i_episode} reward: {ep_rewards} mean latency: {np.mean(ep_latencies)}")

        print("Cleaning up remaining requests...")
        # HACK INCOMING
        spam_process.terminate()
        set_container_cpu_values(1000)
        for i in range(n_agents):
            while True:
                (_, _, cpu_percentage), (_, _, _), (_, _) = envs[i].node.get_container_usage(envs[i].container_id)
                if cpu_percentage > 20:
                    time.sleep(5)
                else:
                    break

        for env in envs:
            env.set_last_limit()

    print(f'Completed {EPISODES} episodes with {np.mean(summed_rewards)} rewards and {np.mean(mean_latencies)} mean latencies.')

    if SAVE_WEIGHTS:
        for i, agent in enumerate(agents):
            torch.save(agent.state_dict(), f'code/model_metric_data/{MODEL}/model_weights_agent_{i}.pth')
    
        # save collected data for later analysis
        ep_summed_rewards_df = pd.DataFrame({'Episode': range(len(summed_rewards)), 'Reward': summed_rewards})
        ep_summed_rewards_df.to_csv(f'code/model_metric_data/{MODEL}/ep_summed_rewards.csv', index=False)

        ep_latencies_df = pd.DataFrame({'Episode': range(len(mean_latencies)), 'Mean Latency': mean_latencies})
        ep_latencies_df.to_csv(f'code/model_metric_data/{MODEL}/ep_latencies.csv', index=False)

        ep_dev = pd.DataFrame({'Episode': range(len(resource_dev)), 'Deviation': resource_dev})
        ep_dev.to_csv(f'code/model_metric_data/{MODEL}/resource_dev.csv', index=False)

        for agent_idx, rewards in enumerate(agent_ep_summed_rewards):
            filename = f'code/model_metric_data/{MODEL}/agent_{agent_idx}_ep_summed_rewards.csv'
            agent_rewards_df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
            agent_rewards_df.to_csv(filename, index=False)
