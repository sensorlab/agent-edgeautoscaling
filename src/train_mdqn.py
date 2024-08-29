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

from envs import DiscreteElasticityEnv, set_available_resource, set_other_priorities, set_other_utilization
from spam_cluster import get_response_times
from pod_controller import set_container_cpu_values, get_loadbalancer_external_port
from utils import save_training_data


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


# TODO: Change variables named latency with response time
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
    EPS_DECAY = 1500
    TAU = 0.005
    LR = 1e-4

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--init_resources', type=int, default=500)
    parser.add_argument('--increment_action', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=5.0, help="Weight for the shared reward, higher the more weight to latency, lower the more weight to efficiency")
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--rps', type=int, default=50, help="Requests per second for loading cluster")
    parser.add_argument('--random_rps', type=bool, default=False, help="Train on random requests every episode")
    parser.add_argument('--interval', type=int, default=1000, help="Milliseconds interval for requests")
    
    parser.add_argument('--priority', type=int, default=0, help="Priority for the environment")
    parser.add_argument('--reward_function', type=int, default=2, help="Setting for the reward function")

    parser.add_argument('--dueling', type=bool, default=False, help="Dueling rl")
    parser.add_argument('--double', type=bool, default=False, help="Double rl")
    parser.add_argument('--load_weights', type=bool, default=False, help="Load weights from previous training")
    parser.add_argument('--variable_resources', type=bool, default=False, help="Random resources every 10 episodes")
    parser.add_argument('--gamma_latency', type=float, default=0.5, help="Latency normalization")

    parser.add_argument('--independent_state', action='store_true', default=False, help="Dont use metrics from other pods (except for available resources)")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug mode")
    parser.add_argument('--reset_env', action='store_true', default=False, help="Resetting the env every 10th episode")
    args = parser.parse_args()

    double = args.double
    dueling = args.dueling

    reqs_per_second = args.rps
    interval = args.interval
    randomize_reqs = args.random_rps
    variable_resources = args.variable_resources

    gamma_latency = args.gamma_latency
    debug = args.debug
    rf = args.reward_function
    priority = args.priority
    alpha = args.alpha
    independent_state = args.independent_state
    reset_env = args.reset_env

    MEMORY_SIZE = 1000
    EPISODES = args.episodes

    LOAD_WEIGHTS = args.load_weights
    SAVE_WEIGHTS = True

    # env values
    RESOURCES = args.init_resources
    INCREMENT_ACTION = args.increment_action
    USERS = 1
    # reqs_per_second -= USERS # interval is set to 1s


    set_container_cpu_values(cpus=100)

    parent_dir = 'src/model_metric_data/dqn'
    MODEL = f'mdqn{EPISODES}ep{MEMORY_SIZE}m{INCREMENT_ACTION}inc{rf}_rf_{reqs_per_second}rps{alpha}alpha'
    if not variable_resources:
        MODEL += f'{RESOURCES}res'
    suffixes = ['_double' if double else '', '_dueling' if dueling else '', '_varres' if variable_resources else '', '_pretrained' if LOAD_WEIGHTS else '']
    MODEL += ''.join(suffixes)
    if independent_state:
        MODEL += "_independent_state"
    os.makedirs(f'{parent_dir}/{MODEL}', exist_ok=True)

    print(f"Initialized model {MODEL}, random_rps {randomize_reqs}, variable_resoruces {variable_resources}, interval {interval} ms, rps {reqs_per_second}")

    n_agents = args.n_agents
    envs = [DiscreteElasticityEnv(i, independent_state=independent_state) for i in range(1, n_agents + 1)]
    other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))] # For every env its other envs (pre-computing), used for priority and utilization

    for i, env in enumerate(envs):
        env.MAX_CPU_LIMIT = RESOURCES
        env.INCREMENT = INCREMENT_ACTION
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
            print("Using default priority setting...")

    n_actions = envs[0].action_space.n
    state = envs[0].reset()
    n_observations = len(state) * len(state[0])
    print(f"Number of observations: {n_observations} and number of actions: {n_actions}")

    set_available_resource(envs, RESOURCES)

    if dueling:
        agents = [DuelingDQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
        target_nets = [DuelingDQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
    else:
        agents = [DQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
        target_nets = [DQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
    
    if LOAD_WEIGHTS:
        for i, agent in enumerate(agents):
            agent.load_state_dict(torch.load(f'src/model_metric_data/dqn/mdqn310ep1000m25inc2_rf_20rps5.0alpha1000res_double_dueling_pretrained?/model_weights_agent_{i}.pth'))
        print(f"Loaded weights for agents")

    memories = [ReplayMemory(MEMORY_SIZE) for _ in range(n_agents)]
    optimizers = [optim.AdamW(agent.parameters(), lr=LR, amsgrad=True) for agent in agents]

    for target_net, agent in zip(target_nets, agents):
        target_net.load_state_dict(agent.state_dict())

    steps_done = 0
    summed_rewards = []
    mean_latencies = []
    agents_summed_rewards = [[] for _ in range(n_agents)]
    agents_mean_latenices = [[] for _ in range(n_agents)]
    
    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}"
    # url = f"http://localhost:30888/predict"

    for i_episode in tqdm(range(EPISODES)):
        if i_episode % 50 == 0 and i_episode != 0 and SAVE_WEIGHTS:
            for i, agent in enumerate(agents):
                torch.save(agent.state_dict(), f'{parent_dir}/{MODEL}/ep_{i_episode}_agent_{i}.pth')
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

        command = ['python', 'src/spam_cluster.py', '--users', str(reqs_per_second), '--interval', str(interval), '--variable', '--all']
        if randomize_reqs:
            command.append('--random_rps')
        spam_process = subprocess.Popen(command)

        states = [env.reset() for env in envs]
        set_available_resource(envs, RESOURCES)
        states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0) for state in states]

        ep_rewards = 0
        ep_latencies = []
        agents_ep_reward = [[] for _ in range(n_agents)]
        ep_std = []
        agents_ep_mean_latency = [[] for _ in range(n_agents)]

        for t in count():
            time.sleep(1)
            
            latencies = [np.mean([latency if latency is not None else 2 for latency in get_response_times(USERS, f'{url}/api{env.id}/predict')]) for env in envs]
            for i, latency in enumerate(latencies):
                agents_ep_mean_latency[i].append(latency)

            latency = np.mean(latencies) # Avg latency of all pods
            ep_latencies.append(latency)

            priority_weighted_latency = sum((1 + env.priority) * latency for env, latency in zip(envs, latencies))
            shared_reward = 1 - alpha * (priority_weighted_latency - 0.01)

            actions = [select_action(state, agent, env) for state, agent, env in zip(states, agents, envs)]

            next_states, rewards, dones = [], [], []
            resources = []
            for i, action in enumerate(actions):
                if not independent_state:
                    set_other_utilization(envs[i], other_envs[i])
                    set_other_priorities(envs[i], other_envs[i])

                observation, agent_reward, done, _ = envs[i].step(action.item(), rf)
                set_available_resource(envs, RESOURCES) # heavy

                reward = 0.5 * agent_reward + shared_reward
                
                next_states.append(np.array(observation).flatten())
                rewards.append(reward)
                dones.append(done)
                resources.append(envs[i].ALLOCATED)
                if done:
                    next_states[i] = None

            #     if debug or t % (envs[i].MAX_STEPS // 2) == 0:
            #         print(f"{envs[i].id}: ACTION: {action}, LIMIT: {envs[i].ALLOCATED}, {envs[i].last_cpu_percentage:.2f}%, AVAILABLE: {envs[i].AVAILABLE}, reward: {reward:.2f} state: {envs[i].state[-1]}, shared_reward: {shared_reward:.2f}, agent_reward: {agent_reward:.2f}")
            # if debug or t % envs[i].MAX_STEPS / 2 == 0:
            #     print()
        
            [agents_ep_reward[i].append(rewards[i]) for i in range(n_agents)]

            ep_rewards += np.mean(rewards)

            next_states = [torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if observation is not None else None for observation in next_states]
            reward_tensors = [torch.tensor([reward], device=device) for reward in rewards]

            for i in range(n_agents):
                memories[i].push(states[i], actions[i], next_states[i], reward_tensors[i])

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
                break
        
        mean_latencies.append(np.mean(ep_latencies))
        summed_rewards.append(ep_rewards)
        
        [agents_summed_rewards[i].append(np.sum(reward)) for i, reward in enumerate(agents_ep_reward)]
        [agents_mean_latenices[i].append(np.mean(latency)) for i, latency in enumerate(agents_ep_mean_latency)]
        print(f"Episode {i_episode} reward: {ep_rewards} mean latency: {np.mean(ep_latencies)}")

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

    print(f'Completed {EPISODES} episodes with {np.mean(summed_rewards)} rewards and {np.mean(mean_latencies)} mean latencies.')

    if SAVE_WEIGHTS:
        for i, agent in enumerate(agents):
            torch.save(agent.state_dict(), f'{parent_dir}/{MODEL}/model_weights_agent_{i}.pth')
    
        save_training_data(f'{parent_dir}/{MODEL}', summed_rewards, mean_latencies, agents_summed_rewards, agents_mean_latenices=agents_mean_latenices)
