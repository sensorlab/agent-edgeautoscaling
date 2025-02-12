import argparse
import math
import os
import random
import subprocess
import time
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from envs import (DiscreteElasticityEnv, FiveDiscreteElasticityEnv, ElevenDiscrElasticityEnv, 
                  set_available_resource, set_other_priorities, set_other_utilization)
from pod_controller import set_container_cpu_values, get_loadbalancer_external_port
from spam_cluster import get_response_times
from utils import save_training_data


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size=64):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer4 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQNAgent:
    def __init__(self, env, double=False, dueling=False, hidden_size=64, learning_rate=1e-4, gamma=0.99, epsilon=0.9,
                 epsilon_min=0.15, epsilon_decay=1500, memory_size=1000, device="cpu"):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(memory_size)
        self.criterion = nn.MSELoss()
        self.device = device
        self.double = double
        if dueling:
            self.policy_net = DuelingDQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
            self.target_net = DuelingDQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        else:
            self.policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
            self.target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.env = env
        self.steps_done = 0

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        act_values = self.policy_net(state)
        return torch.argmax(act_values[0]).item()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        if self.steps_done % 100 == 0:
            print(f"eps_threshold: {eps_threshold} at step {self.steps_done}")
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # Get the action with the best expected reward
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t) and Q(s_{t+1})
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            if self.double:
                next_state_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states).gather(1, next_state_actions).squeeze(1)
            else:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def load(self, folder_path, agent_id=None):
        self.policy_net.load_state_dict(torch.load(f"{folder_path}/model_weights_agent_{agent_id}.pth", weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Loaded weights for agent {agent_id}, located: {folder_path}")

    def save_checkpoint(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_checkpoint(self, path):
        self.policy_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Loaded checkpoint located: {path}")

    def save(self, folder_path, agent_id=None):
        torch.save(self.policy_net.state_dict(), f"{folder_path}/model_weights_agent_{agent_id}.pth")
        print(f"Saved weights for agent {agent_id}, located: {folder_path}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    TAU = 0.005

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--init_resources', type=int, default=500)
    parser.add_argument('--increment_action', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=5.0,
                        help="Weight for the shared reward, higher the more weight to response time, lower the more weight to efficiency")
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--rps', type=int, default=50, help="Requests per second for loading cluster")
    parser.add_argument('--interval', type=int, default=1000, help="Milliseconds interval for requests")
    parser.add_argument('--priority', type=int, default=0, help="Priority for the environment")
    parser.add_argument('--reward_function', type=int, default=2, help="Setting for the reward function")

    parser.add_argument('--variable_resources', type=bool, default=False, help="Random resources every 10 episodes")
    parser.add_argument('--load_weights', type=str, default=False,
                        help="Load weights from previous training with a string of the model parent directory")

    parser.add_argument('--random_rps', action='store_true', default=False,
                        help="Train on random requests every episode")
    parser.add_argument('--dueling', action='store_true', default=False, help="Dueling rl")
    parser.add_argument('--double', action='store_true', default=False, help="Double rl")
    parser.add_argument('--independent_state', action='store_true', default=False,
                        help="Dont use metrics from other pods (except for available resources)")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug mode")
    parser.add_argument('--five', action='store_true', default=False, help="Five actions for dqn")
    parser.add_argument('--eleven', action='store_true', default=False, help="Eleven actions for dqn")
    parser.add_argument('--reset_env', action='store_true', default=False, help="Resetting the env every 10th episode")
    args = parser.parse_args()

    double = args.double
    dueling = args.dueling

    reqs_per_second = args.rps
    interval = args.interval
    randomize_reqs = args.random_rps
    variable_resources = args.variable_resources

    debug = args.debug
    rf = args.reward_function
    priority = args.priority
    alpha = args.alpha
    independent_state = args.independent_state
    reset_env = args.reset_env

    five_actions_env = args.five
    eleven_actions_env = args.eleven

    MEMORY_SIZE = 1000
    EPISODES = args.episodes

    weights_dir = args.load_weights
    SAVE_WEIGHTS = True

    # env values
    RESOURCES = args.init_resources
    INCREMENT_ACTION = args.increment_action
    USERS = 1

    set_container_cpu_values(cpus=100)

    parent_dir = 'src/model_metric_data/dqn'
    # parent_dir = 'src/model_metric_data/dqn_j_experiments'
    MODEL = f'mdqn{EPISODES}ep{MEMORY_SIZE}m{INCREMENT_ACTION}inc{rf}_rf_{reqs_per_second}rps{alpha}alpha'
    if not variable_resources:
        MODEL += f'{RESOURCES}res'
    suffixes = ['_double' if double else '', '_dueling' if dueling else '', '_varres' if variable_resources else '',
                '_pretrained' if weights_dir else '']
    MODEL += ''.join(suffixes)
    if independent_state:
        MODEL += "_independent_state"
    if five_actions_env:
        MODEL += "_five_actions"
    if eleven_actions_env:
        MODEL += "_eleven_actions"
    os.makedirs(f'{parent_dir}/{MODEL}', exist_ok=True)

    print(
        f"Initialized model {MODEL}, random_rps {randomize_reqs}, variable_resoruces {variable_resources}, "
        f"interval {interval} ms, rps {reqs_per_second}")

    n_agents = args.n_agents

    if five_actions_env:
        envs = [FiveDiscreteElasticityEnv(i, independent_state=independent_state) for i in range(1, n_agents + 1)]
    elif eleven_actions_env:
        envs = [ElevenDiscrElasticityEnv(i, independent_state=independent_state) for i in range(1, n_agents + 1)]
    else:
        envs = [DiscreteElasticityEnv(i, independent_state=independent_state) for i in range(1, n_agents + 1)]

    other_envs = [[env for env in envs if env != envs[i]] for i in
                  range(len(envs))]  # For every env its other envs (pre-computing), used for priority and utilization

    for i, env in enumerate(envs):
        env.MAX_CPU_LIMIT = RESOURCES
        env.INCREMENT = INCREMENT_ACTION
        if not independent_state:
            set_other_utilization(env, other_envs[i])
            set_other_priorities(env, other_envs[i])

    train_priority = False
    match priority:
        case 1:
            envs[0].priority = 1.0
            envs[1].priority = 1.0
            envs[2].priority = 1.0
        case 2:
            envs[0].priority = 0.1
            envs[1].priority = 1.0
            envs[2].priority = 0.1
        case 3:
            envs[0].priority = 1.0
            envs[1].priority = 0.1
            envs[2].priority = 0.1
        case _:
            train_priority = True
            print("Using default priority setting...")

    set_available_resource(envs, RESOURCES)

    eps_start, eps_end, eps_decay = 0.9, 0.15, 1000
    agents = [DQNAgent(env, double=double, dueling=dueling, device=str(device), memory_size=MEMORY_SIZE,
                       epsilon=eps_start, epsilon_decay=eps_decay, epsilon_min=eps_end) for env in envs]

    if weights_dir:
        for i, agent in enumerate(agents):
            agent.load(weights_dir, agent_id=i)
        print(f"Loaded weights for agents")

    steps_done = 0
    summed_rewards = []
    mean_rts = []
    agents_summed_rewards = [[] for _ in range(n_agents)]
    agents_mean_rts = [[] for _ in range(n_agents)]

    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}"
    # url = f"http://localhost:30888/predict"

    for i_episode in tqdm(range(EPISODES)):
        if i_episode % 50 == 0 and i_episode != 0 and SAVE_WEIGHTS:
            for i, agent in enumerate(agents):
                agent.save_checkpoint(f'{parent_dir}/{MODEL}/ep_{i_episode}_agent_{i}.pth')
                print(f"Checkpoint: Saved weights for agent {i}")
        if variable_resources and i_episode % 5 == 0:
            RESOURCES = random.choice([500, 750, 1000, 1250, 1500, 1750, 2000])
            for env in envs:
                env.MAX_CPU_LIMIT = RESOURCES
                env.patch(100)
            print(f"Resources changed to {RESOURCES} for episode {i_episode}")

        if i_episode % 10 == 0 and reset_env:
            for env in envs:
                env.patch(100)
                env.reset()

        if i_episode % 4 == 0 and train_priority:
            for env in envs:
                env.priority = random.randint(1, 10) / 10.0

        command = ['python', 'src/spam_cluster.py', '--users', str(reqs_per_second), '--interval', str(interval),
                   '--variable', '--all']
        if randomize_reqs:
            command.append('--random_rps')
        spam_process = subprocess.Popen(command)

        states = [env.reset() for env in envs]
        set_available_resource(envs, RESOURCES)
        states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0) for state in
                  states]

        ep_rewards = 0
        ep_rts = []
        agents_ep_reward = [[] for _ in range(n_agents)]
        ep_std = []
        agent_ep_mean_rt = [[] for _ in range(n_agents)]

        for t in count():
            time.sleep(1)

            rts = [
                np.mean([rt if rt is not None else 2 for rt in get_response_times(USERS, f'{url}/api{env.id}/predict')])
                for env in envs]
            for i, rt in enumerate(rts):
                agent_ep_mean_rt[i].append(rt)

            rt = np.mean(rts)  # Avg renspose time of all pods
            ep_rts.append(rt)

            priority_weighted_rt = sum((1 + env.priority) * rt for env, rt in zip(envs, rts))
            shared_reward = 1 - alpha * (priority_weighted_rt - 0.01)

            actions = [agent.select_action(state) for state, agent in zip(states, agents)]

            next_states, rewards, dones = [], [], []
            for i, action in enumerate(actions):
                if not independent_state:
                    set_other_utilization(envs[i], other_envs[i])
                    set_other_priorities(envs[i], other_envs[i])

                observation, agent_reward, done, _ = envs[i].step(action.item(), rf)
                set_available_resource(envs, RESOURCES)  # heavy

                reward = 0.5 * agent_reward + shared_reward

                next_states.append(np.array(observation).flatten())
                rewards.append(reward)
                dones.append(done)
                if done:
                    next_states[i] = None

                if debug:
                    print(
                        f"{envs[i].id}: ACTION: {action}, LIMIT: {envs[i].ALLOCATED}, "
                        f"{envs[i].last_cpu_percentage:.2f}%, AVAILABLE: {envs[i].AVAILABLE}, "
                        f"reward: {reward:.2f} state: {envs[i].state[-1]}, shared_reward: {shared_reward:.2f}, "
                        f"agent_reward: {agent_reward:.2f}")
            if debug:
                print()

            [agents_ep_reward[i].append(rewards[i]) for i in range(n_agents)]

            ep_rewards += np.mean(rewards)

            next_states = [torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(
                0) if observation is not None else None for observation in next_states]
            reward_tensors = [torch.tensor([reward], device=device) for reward in rewards]

            for agent in agents:
                agent.memory.push(states[i], actions[i], next_states[i], reward_tensors[i])

            states = next_states
            for agent in agents:
                agent.update(BATCH_SIZE)

            for agent in agents:
                target_net_state_dict = agent.target_net.state_dict()
                policy_net_state_dict = agent.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                            1 - TAU)
                agent.target_net.load_state_dict(target_net_state_dict)

            if any(dones):
                break

        mean_rts.append(np.mean(ep_rts))
        summed_rewards.append(ep_rewards)

        [agents_summed_rewards[i].append(np.sum(reward)) for i, reward in enumerate(agents_ep_reward)]
        [agents_mean_rts[i].append(np.mean(rt)) for i, rt in enumerate(agent_ep_mean_rt)]
        print(f"Episode {i_episode} reward: {ep_rewards} mean response time: {np.mean(ep_rts)}")

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

    print(f'Completed {EPISODES} episodes with {np.mean(summed_rewards)} rewards '
          f'and {np.mean(mean_rts)} mean response times.')

    if SAVE_WEIGHTS:
        for i, agent in enumerate(agents):
            agent.save(f'{parent_dir}/{MODEL}', agent_id=i)

        save_training_data(f'{parent_dir}/{MODEL}', summed_rewards, mean_rts, agents_summed_rewards,
                           agent_mean_rts=agents_mean_rts)
