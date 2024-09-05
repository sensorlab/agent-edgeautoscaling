from envs import ContinuousElasticityEnv, InstantContinuousElasticityEnv, set_other_utilization, set_other_priorities, set_available_resource
from spam_cluster import get_response_times
from pod_controller import set_container_cpu_values, get_loadbalancer_external_port
from utils import save_training_data

import os
import subprocess
import time
import gymnasium as gym
import random
import numpy as np
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

    def delete_last(self, batch_size):
        for _ in range(batch_size):
            if self.buffer:
                self.buffer.pop()

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
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4, sigmoid_output=False):
        super(Actor, self).__init__()
        self.sigmoid_output = sigmoid_output
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
        x = torch.sigmoid(self.linear5(x)) if self.sigmoid_output else torch.tanh(self.linear5(x))
        return x


class DDPGagent():
    def __init__(self, env, hidden_size=256, actor_learning_rate=3e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000, sigmoid_output=False):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions, sigmoid_output)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions, sigmoid_output)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

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
    
    def load(self, folder_path, agent_id):
        self.actor.load_state_dict(torch.load(f"{folder_path}/agent_{agent_id}_actor.pth", map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(f"{folder_path}/agent_{agent_id}_critic.pth", map_location=lambda storage, loc: storage))
    
    def save(self, folder_path, agent_id=None):
        torch.save(self.actor.state_dict(), f"{folder_path}/agent_{agent_id}_actor.pth")
        torch.save(self.critic.state_dict(), f"{folder_path}/agent_{agent_id}_critic.pth")
    
    def save_checkpoint(self, path):
        torch.save(self.actor.state_dict(), path + "_actor.pth")
        torch.save(self.critic.state_dict(), path + "_critic.pth")

    def load_checkpoint(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pth"))
        self.critic.load_state_dict(torch.load(path + "_critic.pth"))

def describe_env(env):
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {env.MAX_STEPS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--init_resources', type=int, default=1000, help="Cpu resoruces given to the cluster")
    parser.add_argument('--alpha', type=float, default=0.75, help="Weight for the shared reward, higher the more weight to response time, lower the more weight to efficiency")
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--rps', type=int, default=50, help="Baseline bound of requests per second for loading cluster, if random, it is the upper bound")
    parser.add_argument('--min_rps', type=int, default=10, help="Minimum Requests per second for loading cluster, if the random requests are enabled")
    parser.add_argument('--interval', type=int, default=1000, help="Milliseconds interval for requests")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--scale_action', type=int, default=50, help="How much does the agent scale with an action")
    parser.add_argument('--load_weights', type=str, default=False, help="Load weights from previous training with a string of the model parent directory")
    parser.add_argument('--max_sigma', type=float, default=0.25, help="Max sigma for noise")
    parser.add_argument('--min_sigma', type=float, default=0.01, help="Min sigma for noise")


    parser.add_argument('--priority', type=int, default=0, help="Options: 0, 1, 2")

    parser.add_argument('--independent_state', action='store_true', default=False, help="Dont use metrics from other pods (except for available resources)")
    parser.add_argument('--make_checkpoints', action='store_true', default=False, help="Save weights every 5 episodes")
    parser.add_argument('--random_rps', action='store_true', default=False, help="Train on random requests every episode")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug mode")
    parser.add_argument('--variable_resources', action='store_true', default=False, help="Random resources every 10 episodes")
    parser.add_argument('--old_reward', action='store_true', default=False, help="Use the old reward function")
    parser.add_argument('--instant', action='store_true', default=False, help="Use instant scaling elasticity environemnt")
    parser.add_argument('--reset_env', action='store_true', default=False, help="Resetting the env every 10th episode")
    args = parser.parse_args()

    SAVE_WEIGHTS = True # Always save weightsB)
    weights_dir = args.load_weights
    RESOURCES = args.init_resources
    ALPHA_CONSTANT = args.alpha
    episodes = args.episodes
    variable_resources = args.variable_resources
    interval = args.interval
    randomize_reqs = args.random_rps
    reqs_per_second = args.rps
    n_agents = args.n_agents
    bs = args.batch_size
    debug = args.debug
    scale_action = args.scale_action
    min_rps = args.min_rps
    make_checkpoints = args.make_checkpoints
    old_reward = args.old_reward
    instant = args.instant
    independent_state = args.independent_state
    priority = args.priority
    reset_env = args.reset_env
    max_sigma = args.max_sigma
    min_sigma = args.min_sigma

    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}"
    # url = f"http://localhost:30888/predict"
    USERS = 1

    if instant:
        envs = [InstantContinuousElasticityEnv(i, independent_state=independent_state) for i in range(1, n_agents + 1)]
    else:
        envs = [ContinuousElasticityEnv(i, independent_state=independent_state) for i in range(1, n_agents + 1)]

    other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))] # For every env its other envs (pre-computing), used for priority and utilization
    for i, env in enumerate(envs):
        env.MAX_CPU_LIMIT = RESOURCES
        env.scale_action = scale_action
        if not independent_state:
            set_other_utilization(env, other_envs[i])
            set_other_priorities(env, other_envs[i])

    train_priority = False
    match priority:
        case 1:
            envs[0].priority = 0.1
            envs[1].priority = 1.0
            envs[2].priority = 0.1
        case 2:
            envs[0].priority = 1.0
            envs[1].priority = 0.1
            envs[2].priority = 0.1
        case _:
            train_priority = True
            print("Using default priority setting... which is training priority...")

    agents = [DDPGagent(env, hidden_size=64, max_memory_size=5000, sigmoid_output=instant) for env in envs]
    # decay_period = envs[0].MAX_STEPS * episodes / 1.1 # Makes sense for now
    decay_period = envs[0].MAX_STEPS * episodes / 2.5
    # noises = [OUNoise(env.action_space, max_sigma=0.2, min_sigma=0.005, decay_period=decay_period) for env in envs]
    # noises = [OUNoise(env.action_space, max_sigma=0.25, min_sigma=0.025, decay_period=decay_period) for env in envs]
    noises = [OUNoise(env.action_space, max_sigma=max_sigma, min_sigma=min_sigma, decay_period=decay_period) for env in envs]
    # noises = [OUNoise(env.action_space, max_sigma=0.2, min_sigma=0, decay_period=decay_period) for env in envs]
    # noises = [OUNoise(env.action_space, max_sigma=0.07, min_sigma=0, decay_period=decay_period) for env in envs]
    # noises = [OUNoise(env.action_space, max_sigma=0.2, min_sigma=0.005, decay_period=1250) for env in envs]
    print(f"Noise max sigma: {max_sigma}, decay period: {decay_period}, min sigma: {min_sigma}")

    parent_dir = 'src/model_metric_data/ddpg'
    MODEL = f'{episodes}ep_2rf_{reqs_per_second}rps{ALPHA_CONSTANT}alpha'
    if instant:
        MODEL += '_instant'
    else:
        MODEL += f'_{scale_action}scale'
    if not variable_resources:
        MODEL += f"{RESOURCES}resources"
    if independent_state:
        MODEL += "_independent_state"
    if weights_dir:
        [agent.load(weights_dir, agent_id=i) for i, agent in enumerate(agents)]
        print(f"Successfully loaded weights from {weights_dir}")
        MODEL += "_pretrained"
    os.makedirs(f'{parent_dir}/{MODEL}', exist_ok=True)

    print(f"Training {n_agents} agents for {episodes} episodes with {RESOURCES} resources, {reqs_per_second} requests per second, {interval} ms interval, {ALPHA_CONSTANT} alpha, {bs} batch size\nModel name {MODEL}\n")

    rewards = []
    avg_rewards = []
    mean_rts = []
    agents_summed_rewards = [[] for _ in range(n_agents)]
    agents_mean_rts = [[] for _ in range(n_agents)]

    set_container_cpu_values(100)
    set_available_resource(envs, RESOURCES)

    for episode in tqdm(range(episodes)):
        # Checkpoint
        if episode % 50 == 0 and episode != 0 and make_checkpoints:
            for i, agent in enumerate(agents):
                agent.save_checkpoint(f"{parent_dir}/{MODEL}/ep_{episode}_agent_{i}")
            print(f"Checkpoint saved at episode {episode} for {n_agents} agents")

        if variable_resources and episode % 5 == 0:
            RESOURCES = random.choice([500, 750, 1000, 1250, 1500, 1750, 2000])
            for env in envs:
                env.MAX_CPU_LIMIT = RESOURCES
                env.patch(100)
            print(f"Resources changed to {RESOURCES} for episode {episode}")
        
        if episode % 10 == 0 and reset_env:
            for env in envs:
                env.patch(100)
                env.reset()

        if episode % 4 == 0 and train_priority:
            for env in envs:
                env.priority = random.randint(1, 10) / 10.0

        command = ['python', 'src/spam_cluster.py', '--users', str(reqs_per_second), '--interval', str(interval), '--variable', '--all']
        if randomize_reqs:
            command.append('--random_rps')
        spam_process = subprocess.Popen(command)
        
        states = [np.array(env.reset()).flatten() for env in envs]
        set_available_resource(envs, RESOURCES)

        [noise.reset() for noise in noises]

        ep_rts = []
        ep_rewards = []
        agents_ep_reward = [[] for _ in range(n_agents)]
        agents_ep_mean_rt = [[] for _ in range(n_agents)]

        for step in range(envs[0].MAX_STEPS):
            time.sleep(1)
            agents_step_rewards = []

            rts = [np.mean([rt if rt is not None else 2 for rt in get_response_times(USERS, f'{url}/api{env.id}/predict')]) for env in envs]
            for i, rt in enumerate(rts):
                agents_ep_mean_rt[i].append(rt)
            rt = np.mean(rts) # Avg response time of all pods
            ep_rts.append(rt)

            priority_weighted_rt = sum((1 + env.priority) * rt for env, rt in zip(envs, rts))
            shared_reward = 1 - ALPHA_CONSTANT * (priority_weighted_rt - 0.01)

            actions, new_states, dones = [], [], []
            for i, agent in enumerate(agents):
                state = states[i]
                action = agent.get_action(state)
                action = noises[i].get_action(action, step)
                actions.append(action)

            for i, env in enumerate(envs):
                if not independent_state:
                    set_other_utilization(envs[i], other_envs[i])
                    set_other_priorities(envs[i], other_envs[i])

                new_state, agent_reward, done, _ = env.step(actions[i], 2)
                new_state = np.array(new_state).flatten()
                set_available_resource(envs, RESOURCES)

                # reward = alpha * agent_reward + (1 - alpha) * shared_reward
                reward = 0.5 * agent_reward + shared_reward

                agents[i].memory.push(states[i], actions[i], reward, new_state, done)
                new_states.append(new_state)
                agents_ep_reward[i].append(reward)
                dones.append(done)
                if len(agents[i].memory) > bs:
                    agents[i].update(bs)
                agents_step_rewards.append(reward)
                if debug:
                    print(f"{envs[i].id}: ACTION: {actions[i]}, LIMIT: {envs[i].ALLOCATED}, {envs[i].last_cpu_percentage:.2f}%, AVAILABLE: {envs[i].AVAILABLE}, reward: {reward:.2f} state: {envs[i].state[-1]}, shared_reward: {shared_reward:.2f}, agent_reward: {agent_reward:.2f}")
            if debug:
                print()

            states = new_states
            
            ep_rewards.append(np.mean(agents_step_rewards))

            if any(dones):
                break
        
        mean_rts.append(np.mean(ep_rts))
        rewards.append(sum(ep_rewards))
        [agents_summed_rewards[i].append(np.sum(reward)) for i, reward in enumerate(agents_ep_reward)]
        [agents_mean_rts[i].append(np.mean(rt)) for i, rt in enumerate(agents_ep_mean_rt)]

        spam_process.terminate()
        set_container_cpu_values(1000)
        for i in range(n_agents):
            while True:
                (_, _, cpu_percentage), (_, _, _), (_, _), _ = envs[i].node.get_container_usage(envs[i].container_id)
                if cpu_percentage > 20:
                    time.sleep(1.5)
                else:
                    break
        
        for env in envs:
            env.set_last_limit()
        
        print(f"Episode {episode} reward: {rewards[-1]} mean response time: {np.mean(ep_rts)}")

    if SAVE_WEIGHTS:
        for i, agent in enumerate(agents):
            agent.save(f"{parent_dir}/{MODEL}", agent_id=i)
        
        save_training_data(f'{parent_dir}/{MODEL}', rewards, mean_rts, agents_summed_rewards, agent_mean_rts=agents_mean_rts)
