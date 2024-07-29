from train_ddpg import DDPGagent, set_available_resource
from train_ppo import set_other_priorities, set_other_utilization

from pod_controller import set_container_cpu_values

from envs import ContinuousElasticityEnv, InstantContinuousElasticityEnv

import numpy as np
import time

n_agents = 3
envs = [ContinuousElasticityEnv(i) for i in range(1, n_agents + 1)]
# envs = [InstantContinuousElasticityEnv(i) for i in range(1, n_agents + 1)]

other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))] # For every env its other envs (pre-computing)

agents = [DDPGagent(env, hidden_size=64, sigmoid_output=False) for env in envs]
RESOURCES = 1000
# model_folder = 'trained/continous/ddpg/600ep1000resources50rps1000interval0.6alpha50scale_a0.5gl'
# model_folder = 'code/model_metric_data/ddpg/300ep1000resources50rps1000interval0.5alpha50scale_a0.5gl_pretrained'
# model_folder = 'code/model_metric_data/ddpg/pretrained/100ep1000resources50rps1000interval0.75alpha50scale_a0.5gl'
model_folder = 'code/model_metric_data/ddpg/100ep_2rf_30rps5.0alpha'

for id, agent in enumerate(agents):
    print(f'{model_folder}/agent_{id}_actor.pth', f'{model_folder}/agent_{id}_critic.pth')
    agent.load_model(f'{model_folder}/agent_{id}_actor.pth', f'{model_folder}/agent_{id}_critic.pth')


priorities = [0.1,
              0.1,
              0.3]

debug = True
for i, env in enumerate(envs):
    env.DEBUG = False
    env.MAX_CPU_LIMIT = RESOURCES
    env.priority = priorities[i]

set_container_cpu_values(100)
set_available_resource(envs, RESOURCES)
states = [np.array(env.reset()).flatten() for env in envs]
while True:
    time.sleep(1)
    actions = [[agent.get_action(state)] for state, agent in zip(states, agents)]
    states, rewards, dones, _ = [], [], [], []
    for i, action in enumerate(actions):
        set_other_utilization(envs[i], other_envs[i])
        set_other_priorities(envs[i], other_envs[i])

        state, reward, done, _ = envs[i].step(action, 2)
        set_available_resource(envs, RESOURCES)
        states.append(np.array(state).flatten())
        rewards.append(reward)
        dones.append(done)
        if debug:
            print(f"{envs[i].id}: ACTION: {action}, LIMIT: {envs[i].ALLOCATED}, {envs[i].last_cpu_percentage:.2f}%, AVAILABLE: {envs[i].AVAILABLE}, reward: {reward} state(limit, usage, others): {envs[i].state[-1]}")
    if debug:
        print()
    # if any(dones):
    #     break
