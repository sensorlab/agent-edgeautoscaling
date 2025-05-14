import argparse
import os
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.autograd
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
from tqdm import tqdm

from envs import (ContinuousElasticityEnv, DiscreteElasticityEnv, InstantContinuousElasticityEnv,
                  set_available_resource, set_other_priorities, set_other_utilization)
from pod_controller import set_container_cpu_values, get_loadbalancer_external_port
from spam_cluster import get_response_times
from utils import save_training_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def __len__(self):
        return len(self.states)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, sigmoid_output=False,
                 hidden_size=64):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # dropout = 0.1
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(hidden_size, action_dim),
            )
            if sigmoid_output:
                self.actor.add_module('7', nn.Sigmoid())
            else:
                self.actor.add_module('7', nn.Tanh())

        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(hidden_size, action_dim),
                nn.Softmax(dim=-1)
            )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):
        if self.has_continuous_action_space:
            # Mean of the action distribution
            action_mean = self.actor(state)
            # Variance of each action dimension
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        # Detach creates a new view of the variable that does not require gradients
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
    def __init__(self, env, has_continuous_action_space, eps_clip=0.2, action_std_init=0.6, lr_actor=0.0003,
                 lr_critic=0.001, gamma=0.99, K_epochs=40, sigmoid_output=False):
        state_dim = env.observation_space.shape[0]
        if has_continuous_action_space:
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init,
                                  sigmoid_output=sigmoid_output).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init,
                                      sigmoid_output=sigmoid_output).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.lambda_ = 0.92

    def set_action_std(self, action_std):
        if self.has_continuous_action_space:
            self.action_std = action_std
            self.policy.set_action_std(action_std)
            self.policy_old.set_action_std(action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
            self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        # clip action to env?
        # action = torch.clamp(action, 0, 1)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if self.has_continuous_action_space:
                action = self.policy.actor(state)
                action = action.detach().cpu().numpy().flatten()
            else:
                action_probs = self.policy.actor(state)
                action = torch.argmax(action_probs, dim=-1)
            return action

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

        # Convert lists to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        # TODO: calcualte the advantages using GAE, needs to be tested, if better results
        '''
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * (old_state_values[i + 1] if i + 1 < len(rewards) else 0) - old_state_values[i]
            gae = delta + self.gamma * self.lambda_ * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        '''

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, folder_path, agent_id=None):
        torch.save(self.policy_old.state_dict(), f"{folder_path}/agent_{agent_id}.pth")

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True))

    def load(self, folder_path, agent_id=None):
        self.policy_old.load_state_dict(
            torch.load(f"{folder_path}/agent_{agent_id}.pth", map_location=lambda storage, loc: storage, weights_only=True))
        self.policy.load_state_dict(
            torch.load(f"{folder_path}/agent_{agent_id}.pth", map_location=lambda storage, loc: storage, weights_only=True))
        print(f"Loaded model from {folder_path}/agent_{agent_id}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--init_resources', type=int, default=1000, help="Cpu resoruces given to the cluster")
    parser.add_argument('--alpha', type=float, default=5, help="Weight for the shared reward")
    parser.add_argument('--initial_action_std', type=float, default=0.5,
                        help="Initial action standard deviation for PPO exploaration")
    parser.add_argument('--update_every', type=int, default=3, help="On how many episodes update the models")
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--rps', type=int, default=50,
                        help="Baseline bound of requests per second for loading cluster, if random, it is the upper bound")
    parser.add_argument('--min_rps', type=int, default=10,
                        help="Minimum Requests per second for loading cluster, if the random requests are enabled")
    parser.add_argument('--interval', type=int, default=1000, help="Milliseconds interval for requests")
    parser.add_argument('--scale_action', type=int, default=50, help="How much does the agent scale with an action")
    parser.add_argument('--load_weights', type=str, default=False,
                        help="Load weights from previous training with a string of the model parent directory")
    parser.add_argument('--k_epochs', type=int, default=10, help="K-epochs for ppo training")
    parser.add_argument('--reward_function', type=int, default=2, help="Options: 1, 2, 3, 4, 5...")
    parser.add_argument('--priority', type=int, default=0, help="Options: 0, 1, 2... 0 means to train priority")

    parser.add_argument('--independent_state', action='store_true', default=False,
                        help="Dont use metrics from other pods (except for available resources)")
    parser.add_argument('--make_checkpoints', action='store_true', default=False, help="Save weights every 5 episodes")
    parser.add_argument('--random_rps', action='store_true', default=False,
                        help="Train on random requests every episode")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug mode")
    parser.add_argument('--variable_resources', action='store_true', default=False,
                        help="Random choice of [500, 750, 1000, 1250, 1500, 1750, 2000] resource every 5th episode")
    parser.add_argument('--discrete', action='store_true', default=False, help="Use discrete actions")
    parser.add_argument('--instant', action='store_true', default=False, help="Scale directly to value")

    parser.add_argument('--reset_env', action='store_true', default=False, help="Resetting the env every 10th episode")
    args = parser.parse_args()

    SAVE_WEIGHTS = True  # Always save weightsB)
    weights_dir = args.load_weights
    RESOURCES = args.init_resources
    ALPHA_CONSTANT = args.alpha
    episodes = args.episodes
    variable_resources = args.variable_resources
    interval = args.interval
    randomize_reqs = args.random_rps
    reqs_per_second = args.rps
    n_agents = args.n_agents
    debug = args.debug
    scale_action = args.scale_action
    min_rps = args.min_rps
    make_checkpoints = args.make_checkpoints
    discrete = args.discrete
    reward_function = args.reward_function
    k_epochs = args.k_epochs
    instant = args.instant
    reset_env = args.reset_env
    priority = args.priority
    independent_state = args.independent_state
    initial_action_std = args.initial_action_std
    update_every = args.update_every

    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}"
    # Maybe change it later on to get "truer" response times, but 1 is set for faster training
    USERS = 1

    if discrete:
        envs = [DiscreteElasticityEnv(i, independent_state) for i in range(1, n_agents + 1)]
        increment_action = 25
        for env in envs:
            env.INCREMENT = increment_action
    else:
        if instant:
            envs = [InstantContinuousElasticityEnv(i, independent_state) for i in range(1, n_agents + 1)]
        else:
            envs = [ContinuousElasticityEnv(i, independent_state) for i in range(1, n_agents + 1)]

        for i, env in enumerate(envs):
            env.scale_action = scale_action

    other_envs = [[env for env in envs if env != envs[i]] for i in
                  range(len(envs))]  # For every env its other envs (pre-computing), used for priority and utilization

    train_priority = False
    priorities = [1.0, 1.0, 1.0]
    match priority:
        case 1:
            priorities = [1.0, 0.1, 0.1]
        case 2:
            priorities = [0.1, 1.0, 0.1]
        case 3:
            priorities = [0.1, 0.1, 1.0]
        case 0:
            train_priority = True
        case _:
            print("Using default priority setting... [1, 1, ..., 1]")

    for i, env in enumerate(envs):
        env.MAX_CPU_LIMIT = RESOURCES
        env.priority = priorities[i]
        if not independent_state:
            set_other_utilization(env, other_envs[i])
            set_other_priorities(env, other_envs[i])

    total_steps = envs[0].MAX_STEPS * episodes
    update_timestep = envs[0].MAX_STEPS * update_every
    # initial_action_std = initial_action_std
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = total_steps // 30  # Frequency of decay

    print(
        f"Settings for PPO: {episodes} episodes, {n_agents} agents, {initial_action_std} initial action std, "
        f"{action_std_decay_rate} action std decay rate, {min_action_std} min action std, {update_every} update every, {k_epochs} k epochs")

    time_step = 0

    agents = [PPO(env, has_continuous_action_space=not discrete, action_std_init=initial_action_std, K_epochs=k_epochs,
                  sigmoid_output=instant) for env in envs]
    # Set dropout to training mode
    for agent in agents:
        agent.policy.train()
        agent.policy_old.train()

    parent_dir = 'src/model_metric_data/ppo'
    # parent_dir = 'src/model_metric_data/ppo_j_experiments'
    # MODEL = f'{episodes}ep{RESOURCES}resources_rf_{reward_function}_{reqs_per_second}rps{interval}interval{k_epochs}kepochs{ALPHA_CONSTANT}alpha{scale_action}scale_a{priority}priority'
    MODEL = f'{episodes}ep_rf_{reward_function}_{reqs_per_second}rps{k_epochs}kepochs{int(ALPHA_CONSTANT)}alpha{update_every}epupdate'
    if priority != 0:
        MODEL += f'_{priority}priority'
    if independent_state:
        MODEL += "_independent_state"
    if discrete:
        MODEL += '_discrete'
    if instant:
        MODEL += '_instantscale'
    else:
        MODEL += f"{scale_action}scale_a"
    if not reset_env:
        MODEL += '_NOreseting'
    # No need to specify resources if they are variable
    if variable_resources:
        MODEL += '_vari_res'
    else:
        MODEL += f'_{RESOURCES}resources'
    if weights_dir:
        [agent.load(weights_dir, agent_id=i) for i, agent in enumerate(agents)]
        print(f"Successfully loaded weights from {weights_dir}")
        MODEL += "_pretrained"
    os.makedirs(f'{parent_dir}/{MODEL}', exist_ok=True)

    print(
        f"Training {n_agents} agents for {episodes} episodes with {RESOURCES} resources, {reqs_per_second} requests per "
        f"second, {interval} ms interval, {ALPHA_CONSTANT} alpha\nModel name {MODEL}\n")

    rewards = []
    avg_rewards = []
    mean_rts = []
    agents_summed_rewards = [[] for _ in range(n_agents)]
    agents_mean_rts = [[] for _ in range(n_agents)]

    set_container_cpu_values(100)
    set_available_resource(envs, RESOURCES)

    utilization_all_step_rewards = [[] for _ in range(n_agents)]
    all_steps_rts_rewards = []
    all_steps_rts = []

    for episode in tqdm(range(episodes)):
        # Checkpoint
        if episode % 50 == 0 and episode != 0 and make_checkpoints:
            for i, agent in enumerate(agents):
                agent.save_checkpoint(f"{parent_dir}/{MODEL}/ep_{episode}_agent_{i}.pth")
            print(f"Checkpoint saved at episode {episode} for {n_agents} agents")

        if variable_resources and episode % 5 == 0:
            RESOURCES = random.choice([500, 750, 1000, 1250, 1500, 1750, 2000])
            for env in envs:
                env.MAX_CPU_LIMIT = RESOURCES
                # It can happen that they have allocated more than available and be stuck, so patch them in case
                env.patch(100)
            print(f"Resources changed to {RESOURCES} for episode {episode}")

        if episode % 10 == 0 and reset_env:
            for env in envs:
                env.patch(100)
                env.reset()

        if episode % 4 == 0 and train_priority:
            priorities = [random.randint(1, 10) / 10.0 for _ in range(n_agents)]
            for i, env in enumerate(envs):
                env.priority = priorities[i]
                # if reset_env:
                #     env.patch(100)
            print(f"Priorities for envs changed to {priorities} for episode {episode}")

        # random_rps = np.random.randint(min_rps, reqs_per_second) if randomize_reqs else reqs_per_second
        # In this training the random rps is handled by the loading script

        command = ['python', 'src/spam_cluster.py', '--users', str(reqs_per_second), '--interval', str(interval),
                   '--variable', '--all']
        if randomize_reqs:
            command.append('--random_rps')
        spam_process = subprocess.Popen(command)

        states = [np.array(env.reset()).flatten() for env in envs]
        set_available_resource(envs, RESOURCES)

        ep_rts = []
        ep_rewards = []
        agents_ep_reward = [[] for _ in range(n_agents)]
        agents_ep_mean_rt = [[] for _ in range(n_agents)]

        for step in range(envs[0].MAX_STEPS):
            time.sleep(1)
            time_step += 1
            agents_step_rewards = []

            # Give it 2, to avoid mean of None type
            rts = [
                np.mean([rt if rt is not None else 2 for rt in get_response_times(USERS, f'{url}/api{env.id}/predict')])
                for env in envs]

            for i, rt in enumerate(rts):
                agents_ep_mean_rt[i].append(rt)

            rt = np.mean(rts)  # Avg reponse time of all pods

            all_steps_rts.append(rt)
            ep_rts.append(rt)

            match reward_function:
                case 1 | 2:
                    priority_weighted_rt = sum((1 + env.priority) * rt for env, rt in zip(envs, rts))
                    shared_reward = 1 - ALPHA_CONSTANT * (priority_weighted_rt - 0.01)
                case 3:
                    shared_reward = (1 - 10 * rt)
                case 4:
                    shared_reward = (- rt * 10)
                case 5:
                    shared_reward = (1 - 10 * rt)
                case _:
                    print("No implemented reward function")
                    break

            all_steps_rts_rewards.append(shared_reward)

            actions, new_states, dones = [], [], []
            for i, agent in enumerate(agents):
                if not independent_state:
                    set_other_utilization(envs[i], other_envs[i])
                    set_other_priorities(envs[i], other_envs[i])

                action = agent.select_action(states[i])

                new_state, agent_reward, done, _ = envs[i].step(action, reward_function)
                new_state = np.array(new_state).flatten()
                new_states.append(new_state)
                dones.append(done)

                set_available_resource(envs, RESOURCES)

                reward = 0.5 * agent_reward + shared_reward

                utilization_all_step_rewards[i].append(agent_reward)

                agents_ep_reward[i].append(reward)
                agents_step_rewards.append(reward)

                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)

                if time_step % update_timestep == 0:
                    agent.update()
                    if debug:
                        print(f"Agent {i} updated at time step {time_step}")

                if time_step % action_std_decay_freq == 0:
                    agent.decay_action_std(action_std_decay_rate, min_action_std)
                    if debug:
                        print(f"Agent {i} action std decayed at time step {time_step}")

                if debug:
                    print(
                        f"{envs[i].id}: ACTION: {action}, LIMIT: {envs[i].ALLOCATED}, "
                        f"{envs[i].last_cpu_percentage:.2f}%, AVAILABLE: {envs[i].AVAILABLE}, reward: {reward:.2f} "
                        f"state: {envs[i].state[-1]}, shared_reward: {shared_reward:.2f}, agent_reward: {agent_reward:.2f}")
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

        save_training_data(f'{parent_dir}/{MODEL}', rewards, mean_rts, agents_summed_rewards,
                           agent_mean_rts=agents_mean_rts)

        for agent_idx, rewards in enumerate(utilization_all_step_rewards):
            filename = f'{parent_dir}/{MODEL}/agent_{agent_idx}_step_util_rewards.csv'
            agent_rewards_df = pd.DataFrame({'Step': range(len(rewards)), 'Reward': rewards})
            agent_rewards_df.to_csv(filename, index=False)

        step_rt_reward_df = pd.DataFrame(
            {'Step': range(len(all_steps_rts_rewards)), 'Shared reward': all_steps_rts_rewards})
        step_rt_reward_df.to_csv(f'{parent_dir}/{MODEL}/step_latency_shared_reward.csv', index=False)

        step_rt_df = pd.DataFrame({'Step': range(len(all_steps_rts)), 'Latency': all_steps_rts})
        step_rt_df.to_csv(f'{parent_dir}/{MODEL}/step_latencies.csv', index=False)
