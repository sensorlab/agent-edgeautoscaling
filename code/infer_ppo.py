from train_ppo import PPO, set_available_resource
from continous_env import ContinousElasticityEnv

import numpy as np
import time

n_agents = 3
initial_action_std = 1e-7

envs = [ContinousElasticityEnv(i, n_agents) for i in range(1, n_agents + 1)]
agents = [PPO(env, has_continuous_action_space=True, action_std_init=initial_action_std) for env in envs]
print(f"Frist agent action std: {agents[0].action_std}")

RESOURCES = 1000
model_folder = 'code/model_metric_data/ppo/30ep1000resources50rps1000interval0.75alpha50scale_a0.3gl'

for id, agent in enumerate(agents):
    agent.load(f'{model_folder}/agent_{id}.pth')
    print(f'Loaded {model_folder}/agent_{id}.pth')

debug = True
for env in envs:
    env.DEBUG = False

set_available_resource(envs, RESOURCES)
states = [np.array(env.reset()).flatten() for env in envs]
while True:
    time.sleep(1)
    actions = [agent.select_action(state) for state, agent in zip(states, agents)]
    states, rewards, dones, _ = [], [], [], []
    for env, action in zip(envs, actions):
        state, reward, done, _ = env.step(action)
        set_available_resource(envs, RESOURCES)
        states.append(np.array(state).flatten())
        rewards.append(reward)
        dones.append(done)
        if debug:
            print(f"Agent {env.id}, ACTION: {action}, LIMIT: {env.ALLOCATED}, AVAILABLE: {env.AVAILABLE}, reward: {reward} state(limit, usage, others): {env.state[-1]}")
    if debug:
        print()
