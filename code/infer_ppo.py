from train_ppo import PPO, set_available_resource, set_other_priorities, set_other_utilization
from envs import ContinuousElasticityEnv, DiscreteElasticityEnv, InstantContinuousElasticityEnv
from pod_controller import set_container_cpu_values
import numpy as np
import time

n_agents = 3
initial_action_std = 1e-7
# initial_action_std = 0.1

# envs = [ContinuousElasticityEnv(i) for i in range(1, n_agents + 1)]
envs = [InstantContinuousElasticityEnv(i) for i in range(1, n_agents + 1)]
# envs = [DiscreteElasticityEnv(i) for i in range(1, n_agents + 1)]

other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))] # For every env its other envs (pre-computing)
# agents = [PPO(env, has_continuous_action_space=False) for env in envs]
agents = [PPO(env, has_continuous_action_space=True, action_std_init=initial_action_std, sigmoid_output=True) for env in envs]
# agents = [PPO(env, has_continuous_action_space=False) for env in envs]
# print(f"Frist agent action std: {agents[0].action_std}")

RESOURCES = 1500
# model_folder = f'code/model_metric_data/ppo/52ep1000resources_rf_2_75rps1000interval9kepochs5.0alpha50scale_a0.5gl_NOreseting'
# model_folder = f'code/model_metric_data/ppo/52ep1000resources_rf_2_75rps1000interval9kepochs5.0alpha50scale_a0.5gl_NOreseting'
# model_folder = f'code/model_metric_data/ppo/150ep_rf_2_30rps10kepochs5alpha50scale_a0priority_newloading_discrete_NOreseting_vari_res'
# model_folder = 'trained/ppo/100ep1000resources_rf_2_75rps1000interval9kepochs5.0alpha50scale_a_instantscale_NOreseting_vari_res'
model_folder = 'trained/ppo/100ep_rf_2_30rps10kepochs5alpha50scale_a0priority_newloading_instantscale_NOreseting_vari_res'

for id, agent in enumerate(agents):
    agent.load(f'{model_folder}/agent_{id}.pth')
    print(f'Loaded {model_folder}/agent_{id}.pth')

priorities = [1.0,
              0.2,
              0.2]

debug = True
for i, env in enumerate(envs):
    env.DEBUG = False
    agents[i].policy.eval() # set to evaluation mode
    env.MAX_CPU_LIMIT = RESOURCES
    env.INCREMENT = 25
    env.priority = priorities[i]

set_container_cpu_values(100)
set_available_resource(envs, RESOURCES)
states = [np.array(env.reset()).flatten() for env in envs]
while True:
    time.sleep(1)
    # actions = [agent.select_inference_action(state) for state, agent in zip(states, agents)]
    actions = [agent.select_action(state) for state, agent in zip(states, agents)]
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
