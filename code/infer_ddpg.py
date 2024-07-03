from train_ddpg import DDPGagent, set_available_resource
from continous_env import ContinousElasticityEnv

import numpy as np
import time

n_agents = 3
envs = [ContinousElasticityEnv(i, n_agents) for i in range(1, n_agents + 1)]
agents = [DDPGagent(env, hidden_size=64) for env in envs]
RESOURCES = 1000
# model_folder = 'trained/continous/ddpg/1000ep1000resources50rps1000interval0.75alpha'
model_folder = 'code/model_metric_data/ddpg/500ep1000resources50rps1000interval0.5alpha50scale_a0.5gl'

for id, agent in enumerate(agents):
    print(f'{model_folder}/agent_{id}_actor.pth', f'{model_folder}/agent_{id}_critic.pth')
    agent.load_model(f'{model_folder}/agent_{id}_actor.pth', f'{model_folder}/agent_{id}_critic.pth')

debug = True
for env in envs:
    env.DEBUG = False

set_available_resource(envs, RESOURCES)
states = [np.array(env.reset()).flatten() for env in envs]
while True:
    time.sleep(1)
    actions = [[agent.get_action(state)] for state, agent in zip(states, agents)]
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
    # if any(dones):
    #     break
