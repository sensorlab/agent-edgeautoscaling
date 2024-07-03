from continous_env import ContinousElasticityEnv
from spam_cluster import spam_requests_single
from pod_controller import set_container_cpu_values

import os
import subprocess
import time
import gymnasium as gym
import random
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.Wrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.linear4 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear5 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.linear4 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear5 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = torch.tanh(self.linear5(x))
        return x


class DDPGagent():
    def __init__(self, env, hidden_size=256, actor_learning_rate=3e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2,
                 max_memory_size=50000):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_learning_rate, weight_decay=1e-5)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_learning_rate, weight_decay=1e-5)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0, 0]
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))

        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
    
    def save_model(self, path):
        torch.save(self.actor.state_dict(), path + "_actor.pth")
        torch.save(self.critic.state_dict(), path + "_critic.pth")


def describe_env(env):
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {env.MAX_STEPS}")


def set_available_resource(envs, initial_resources):
    max_group = initial_resources
    for env in envs:
        max_group -= env.ALLOCATED
    for env in envs:
        env.AVAILABLE = max_group


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--init_resources', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.75, help="Weight for the shared reward, higher the more weight to latency, lower the more weight to efficiency")
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--rps', type=int, default=50, help="Requests per second for loading cluster")
    parser.add_argument('--min_rps', type=int, default=10, help="Minimum Requests per second for loading cluster, if the random requests are enabled")
    parser.add_argument('--interval', type=int, default=1000, help="Milliseconds interval for requests")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--gamma_latency', type=float, default=0.5, help="Latency normalization")
    parser.add_argument('--scale_action', type=int, default=50, help="How much does the agent scale with an action")
    parser.add_argument('--load_weights', type=str, default=False, help="Load weights from previous training with a string of the model parent directory")

    parser.add_argument('--make_checkpoints', action='store_true', default=False, help="Save weights every 5 episodes")
    parser.add_argument('--random_rps', action='store_true', default=False, help="Train on random requests every episode")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug mode")
    parser.add_argument('--variable_resources', action='store_true', default=False, help="Random resources every 10 episodes")
    args = parser.parse_args()

    SAVE_WEIGHTS = True # Always save weightsB)
    weights_dir = args.load_weights
    RESOURCES = args.init_resources
    alpha = args.alpha
    episodes = args.episodes
    variable_resources = args.variable_resources
    interval = args.interval
    randomize_reqs = args.random_rps
    reqs_per_second = args.rps
    n_agents = args.n_agents
    bs = args.batch_size
    debug = args.debug
    gamma_latency = args.gamma_latency
    scale_action = args.scale_action
    min_rps = args.min_rps
    make_checkpoints = args.make_checkpoints

    url = f"http://localhost:30888/predict"
    USERS = 10

    envs = [ContinousElasticityEnv(i, n_agents) for i in range(1, n_agents + 1)]
    for env in envs:
        env.MAX_CPU_LIMIT = RESOURCES
        env.DEBUG = False
        env.scale_action = scale_action
    agents = [DDPGagent(env, hidden_size=64, max_memory_size=60000) for env in envs]
    decay_period = envs[0].MAX_STEPS * episodes / 1.1 # Makes sense for now
    # noises = [OUNoise(env.action_space, max_sigma=0.2, min_sigma=0.005, decay_period=decay_period) for env in envs]
    noises = [OUNoise(env.action_space, max_sigma=0.2, min_sigma=0, decay_period=decay_period) for env in envs]
    # noises = [OUNoise(env.action_space, max_sigma=0.2, min_sigma=0.005, decay_period=1250) for env in envs]

    parent_dir = 'code/model_metric_data/ddpg'
    MODEL = f'{episodes}ep{RESOURCES}resources{reqs_per_second}rps{interval}interval{alpha}alpha{scale_action}scale_a{gamma_latency}gl'
    if weights_dir:
        [agent.load_model(f"{parent_dir}/{weights_dir}/agent_{i}_actor.pth", f"{parent_dir}/{weights_dir}/agent_{i}_critic.pth") for i, agent in enumerate(agents)]
        print(f"Successfully loaded weights from {parent_dir}/{weights_dir}")
        MODEL += "_pretrained"
    os.makedirs(f'{parent_dir}/{MODEL}', exist_ok=True)

    print(f"Training {n_agents} agents for {episodes} episodes with {RESOURCES} resources, {reqs_per_second} requests per second, {interval} ms interval, {alpha} alpha, {bs} batch size\nModel name {MODEL}, OUNoise decay period {decay_period}\n")

    rewards = []
    avg_rewards = []
    mean_latencies = []
    agents_summed_rewards = [[] for _ in range(n_agents)]

    set_container_cpu_values(100)
    set_available_resource(envs, RESOURCES)

    for episode in tqdm(range(episodes)):
        # Checkpoint
        if episode % 5 == 0 and episode != 0 and make_checkpoints:
            for i, agent in enumerate(agents):
                agent.save_model(f"{parent_dir}/{MODEL}/agent_{i}")

        random_rps = np.random.randint(min_rps, reqs_per_second) if randomize_reqs else reqs_per_second
        spam_process = subprocess.Popen(['python', 'code/spam_cluster.py', '--users', str(random_rps), '--interval', str(interval)])
        print(f"Loading cluster with {random_rps} requests per second")
        
        states = [np.array(env.reset()).flatten() for env in envs]
        set_available_resource(envs, RESOURCES)

        [noise.reset() for noise in noises]

        ep_latencies = []
        ep_rewards = []
        agents_ep_reward = [[] for _ in range(n_agents)]

        for step in range(envs[0].MAX_STEPS):
            time.sleep(1)
            agents_step_rewards = []

            latencies = spam_requests_single(USERS, url)
            latency = np.mean([latency for latency in latencies if latency is not None])
            ep_latencies.append(latency)
            latency = min(latency, gamma_latency)

            actions, new_states, dones = [], [], []
            for i, agent in enumerate(agents):
                state = states[i]
                action = agent.get_action(state)
                action = noises[i].get_action(action, step)
                actions.append(action)

            for i, env in enumerate(envs):
                new_state, reward, done, _ = env.step(actions[i])
                new_state = np.array(new_state).flatten()
                set_available_resource(envs, RESOURCES)

                # reward = alpha * reward + (1 - alpha) * (1 - latency * 10)
                shared_reward = (gamma_latency - latency) / gamma_latency
                reward = alpha * reward + (1 - alpha) * shared_reward

                agents[i].memory.push(states[i], actions[i], reward, new_state, done)
                new_states.append(new_state)
                agents_ep_reward[i].append(reward)
                dones.append(done)
                if len(agents[i].memory) > bs:
                    agents[i].update(bs)
                agents_step_rewards.append(reward)
                if debug:
                    print(f"Agent {env.id}, ACTION: {actions[i]}, LIMIT: {env.ALLOCATED}, AVAILABLE: {env.AVAILABLE}, reward: {reward} state(limit, usage, others): {env.state[-1]}, shared_reward: {shared_reward}, agent_reward: {reward}")
            if debug:
                print()
            
            if step % 30 == 0 and step != 0:
                print(f"Shared: {agents_step_rewards}, latency: {latency}")
                for env in envs:
                    print(f"Agent {env.id}: {env.last_cpu_percentage} % CPU, {env.AVAILABLE} available CPU", end=" ")
                print()

            ep_rewards.append(np.mean(agents_step_rewards))

            if any(dones):
                # for env in envs:
                #     env.save_last_limit()
                break

            states = new_states
        
        mean_latencies.append(np.mean(ep_latencies))
        rewards.append(sum(ep_rewards))
        [agents_summed_rewards[i].append(np.sum(reward)) for i, reward in enumerate(agents_ep_reward)]

        spam_process.terminate()
        set_container_cpu_values(1000)
        for i in range(n_agents):
            while True:
                (_, _, cpu_percentage), (_, _, _), (_, _) = envs[i].node.get_container_usage(envs[i].container_id)
                if cpu_percentage > 20:
                    time.sleep(5)
                else:
                    break
        
        # for env in envs:
        #     env.set_last_limit()
        
        print(f"Episode {episode} reward: {rewards[-1]} mean latency: {np.mean(ep_latencies)}")

    if SAVE_WEIGHTS:
        for i, agent in enumerate(agents):
            agent.save_model(f"{parent_dir}/{MODEL}/agent_{i}")
        
        ep_summed_rewards_df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
        ep_summed_rewards_df.to_csv(f'{parent_dir}/{MODEL}/ep_summed_rewards.csv', index=False)

        ep_latencies_df = pd.DataFrame({'Episode': range(len(mean_latencies)), 'Mean Latency': mean_latencies})
        ep_latencies_df.to_csv(f'{parent_dir}/{MODEL}/ep_latencies.csv', index=False)

        for agent_idx, rewards in enumerate(agents_summed_rewards):
            filename = f'{parent_dir}/{MODEL}/agent_{agent_idx}_ep_summed_rewards.csv'
            agent_rewards_df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
            agent_rewards_df.to_csv(filename, index=False)

