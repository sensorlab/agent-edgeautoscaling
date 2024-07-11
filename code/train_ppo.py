from continous_env import ContinousElasticityEnv
from spam_cluster import spam_requests_single
from pod_controller import set_container_cpu_values
from utils import calculate_dynamic_rps

import os
import subprocess
import time
import gymnasium as gym
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.autograd
from torch.distributions import Categorical, MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)


    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, has_continuous_action_space, eps_clip=0.2, action_std_init=0.6, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=40):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
            self.set_action_std(self.action_std)


    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()


    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def set_available_resource(envs, initial_resources):
    max_group = initial_resources
    for env in envs:
        max_group -= env.ALLOCATED
    for env in envs:
        env.AVAILABLE = max_group


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--init_resources', type=int, default=1000, help="Cpu resoruces given to the cluster")
    parser.add_argument('--alpha', type=float, default=0.75, help="Weight for the shared reward, higher the more weight to latency, lower the more weight to efficiency")
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--rps', type=int, default=50, help="Baseline bound of requests per second for loading cluster, if random, it is the upper bound")
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
    
    # PPO custom parameters
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    

    # update_timestep = 150
    # initial_action_std = 0.6
    # action_std_decay_rate = 0.075
    # min_action_std = 1e-7
    # action_std_decay_freq = 125  # Frequency of decay

    # update_timestep = 2000
    # initial_action_std = 0.6
    # action_std_decay_rate = 0.075
    # min_action_std = 1e-7
    # action_std_decay_freq = 3000  # Frequency of decay

    # 300 episodes
    # update_timestep = 250
    # initial_action_std = 0.6
    # action_std_decay_rate = 0.065
    # min_action_std = 1e-7
    # action_std_decay_freq = 1500 # Frequency of decay

    # 50 episodes
    update_timestep = 150
    initial_action_std = 0.6
    action_std_decay_rate = 0.065
    min_action_std = 1e-7
    action_std_decay_freq = 250 # Frequency of decay
    
    time_step = 0

    agents = [PPO(state_dim, action_dim, has_continuous_action_space=True, action_std_init=initial_action_std) for _ in envs]


    parent_dir = 'code/model_metric_data/ppo'
    MODEL = f'{episodes}ep{RESOURCES}resources{reqs_per_second}rps{interval}interval{alpha}alpha{scale_action}scale_a{gamma_latency}gl'
    if weights_dir:
        [agent.load_model(f"{parent_dir}/pretrained/{weights_dir}/agent_{i}_actor.pth", f"{parent_dir}/pretrained/{weights_dir}/agent_{i}_critic.pth") for i, agent in enumerate(agents)]
        print(f"Successfully loaded weights from {parent_dir}/{weights_dir}")
        MODEL += "_pretrained"
    os.makedirs(f'{parent_dir}/{MODEL}', exist_ok=True)

    print(f"Training {n_agents} agents for {episodes} episodes with {RESOURCES} resources, {reqs_per_second} requests per second, {interval} ms interval, {alpha} alpha, {bs} batch size\nModel name {MODEL}\n")

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
                agent.save(f"{parent_dir}/{MODEL}/agent_{i}.pth")
            print(f"Checkpoint saved at episode {episode} for {n_agents} agents")

        if episode % 2 == 0 and episode != 0:
            for env in envs:
                env.patch(100)

        random_rps = np.random.randint(min_rps, reqs_per_second) if randomize_reqs else reqs_per_second
        
        spam_process = subprocess.Popen(['python', 'code/spam_cluster.py', '--users', str(random_rps), '--interval', str(interval)])
        print(f"Loading cluster with {random_rps} requests per second")
        
        states = [np.array(env.reset()).flatten() for env in envs]
        set_available_resource(envs, RESOURCES)

        ep_latencies = []
        ep_rewards = []
        agents_ep_reward = [[] for _ in range(n_agents)]

        for step in range(envs[0].MAX_STEPS):
            time.sleep(1)
            agents_step_rewards = []

            latencies = spam_requests_single(USERS, url)
            latency = np.mean([latency for latency in latencies if latency is not None])
            ep_latencies.append(latency)

            # shared_reward = 1 - latency * 10

            latency = min(latency, gamma_latency)
            shared_reward = (gamma_latency - latency) / gamma_latency

            actions, new_states, dones = [], [], []
            for i, agent in enumerate(agents):
                action = agent.select_action(states[i])

                new_state, agent_reward, done, _ = envs[i].step(action)
                new_state = np.array(new_state).flatten()
                new_states.append(new_state)
                dones.append(done)

                time_step += 1

                set_available_resource(envs, RESOURCES)

                reward = alpha * agent_reward + (1 - alpha) * shared_reward
                agents_ep_reward[i].append(reward)
                agents_step_rewards.append(reward)

                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)

                if time_step % update_timestep == 0:
                    agent.update()
                
                if time_step % action_std_decay_freq == 0:
                    agent.decay_action_std(action_std_decay_rate, min_action_std)
                
                if debug:
                    print(f"Agent {envs[i].id}, ACTION: {action}, LIMIT: {envs[i].ALLOCATED}, AVAILABLE: {envs[i].AVAILABLE}, reward: {reward} state(limit, usage, others): {envs[i].state[-1]}, shared_reward: {shared_reward}, agent_reward: {agent_reward}")
            if debug:
                print()
            
            states = new_states
            ep_rewards.append(np.mean(agents_step_rewards))
            
            if any(dones):
                break
        
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
        
        for env in envs:
            env.set_last_limit()
        
        print(f"Episode {episode} reward: {rewards[-1]} mean latency: {np.mean(ep_latencies)}")

    if SAVE_WEIGHTS:
        for i, agent in enumerate(agents):
            agent.save(f"{parent_dir}/{MODEL}/agent_{i}.pth")
        
        ep_summed_rewards_df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
        ep_summed_rewards_df.to_csv(f'{parent_dir}/{MODEL}/ep_summed_rewards.csv', index=False)

        ep_latencies_df = pd.DataFrame({'Episode': range(len(mean_latencies)), 'Mean Latency': mean_latencies})
        ep_latencies_df.to_csv(f'{parent_dir}/{MODEL}/ep_latencies.csv', index=False)

        for agent_idx, rewards in enumerate(agents_summed_rewards):
            filename = f'{parent_dir}/{MODEL}/agent_{agent_idx}_ep_summed_rewards.csv'
            agent_rewards_df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
            agent_rewards_df.to_csv(filename, index=False)

