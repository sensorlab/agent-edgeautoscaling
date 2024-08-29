import numpy as np
import time
import argparse
import torch

from train_ppo import PPO
from train_ddpg import DDPGagent
from train_mdqn import DQN, DuelingDQN
from envs import ContinuousElasticityEnv, DiscreteElasticityEnv, InstantContinuousElasticityEnv, set_available_resource, set_other_priorities, set_other_utilization


def infer(n_agents=None, resources=None, independent=False, tl_agent=False, model=None, debug=False, action_interval=None, priorities=None, algorithm=None):
    instant, discrete = False, False

    match algorithm:
        case 'ppo' | 'ddpg':
            instant = False
            discrete = False
        case 'ippo' | 'iddpg':
            instant = True
            discrete = False
        case 'mdqn' | 'dmdqn' | 'ddmdqn' | 'dppo':
            instant = False
            discrete = True
        case _:
            print("Invalid algorithm")
            return

    if discrete:
        envs = [DiscreteElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]
    else:
        if instant:
            envs = [InstantContinuousElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]
        else:
            envs = [ContinuousElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]

    state = envs[0].reset()

    match algorithm:
        case 'ppo' | 'dppo':
            agents = [PPO(env, has_continuous_action_space=not discrete, action_std_init=1e-10, sigmoid_output=instant) for env in envs]
        case 'ippo':
            agents = [PPO(env, has_continuous_action_space=not discrete, action_std_init=1e-10, sigmoid_output=instant) for env in envs]
        case 'mdqn' | 'dmdqn':
            agents = [DQN(len(state) * len(state[0]), envs[0].action_space.n) for env in envs]
        case 'ddmdqn':
            agents = [DuelingDQN(len(state) * len(state[0]), envs[0].action_space.n) for env in envs]
        case 'ddpg' | 'iddpg':
            agents = [DDPGagent(env, hidden_size=64, sigmoid_output=instant) for env in envs]

    other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))]  # For every env its other envs (pre-computing)

    if not model:
        print("Please provide a model to load")
        return

    for id, agent in enumerate(agents):
        if 'ppo' in algorithm:
            if tl_agent:
                agent.load(f'{model}/agent_{tl_agent}.pth')
            else:
                agent.load(f'{model}/agent_{id}.pth')
        elif 'dqn' in algorithm:
            if tl_agent:
                agent.load_state_dict(torch.load(f'{model}/model_weights_agent_{tl_agent}.pth'))
            else:
                agent.load_state_dict(torch.load(f'{model}/model_weights_agent_{id}.pth'))
        elif 'ddpg' in algorithm:
            if tl_agent:
                agent.load_model(f'{model}/agent_{tl_agent}_actor.pth', f'{model}/agent_{tl_agent}_critic.pth')
            else:
                agent.load_model(f'{model}/agent_{id}_actor.pth', f'{model}/agent_{id}_critic.pth')

    for i, env in enumerate(envs):
        env.DEBUG = False
        env.MAX_CPU_LIMIT = resources
        env.INCREMENT = 25
        env.priority = priorities[i]
        env.scale_action = 100

    set_available_resource(envs, resources)
    states = [np.array(env.reset()).flatten() for env in envs]
    while True:
        start_time = time.time()

        if 'ppo' in algorithm:
            actions = [agent.select_action(state) for state, agent in zip(states, agents)]
        elif 'ddpg' in algorithm:
            actions = [[agent.get_action(state)] for state, agent in zip(states, agents)]
        elif 'dqn' in algorithm:
            with torch.no_grad():
                actions = [dqn(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).max(1).indices.view(1, 1) for dqn, state in zip(agents, states)]
        else:
            print("Something went terribly wrong")
            return
        
        states, rewards, dones, _ = [], [], [], []
        for i, action in enumerate(actions):
            set_other_utilization(envs[i], other_envs[i])
            set_other_priorities(envs[i], other_envs[i])

            state, reward, done, _ = envs[i].step(action, 2)
            set_available_resource(envs, resources)
            states.append(np.array(state).flatten())
            rewards.append(reward)
            dones.append(done)
            if debug:
                print(f"{envs[i].id}: ACTION: {action}, LIMIT: {envs[i].ALLOCATED}, {envs[i].last_cpu_percentage:.2f}%, AVAILABLE: {envs[i].AVAILABLE}, reward: {reward} state(limit, usage, others): {envs[i].state[-1]}")
        if debug:
            print()
        elapsed_time = time.time() - start_time
        if elapsed_time < action_interval:
            time.sleep(action_interval - elapsed_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--resources', type=int, default=1000)
    parser.add_argument('--load_model', type=str, default='trained/ppo/1000ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources') # Default trained weights for ppo model
    parser.add_argument('--action_interval', type=int, default=5)
    parser.add_argument('--priorities', type=float, nargs='+', default=[1.0, 1.0, 1.0], help='List of priorities (0.0 < value <= 1.0), default is 1.0 for all agents. Example: 1.0 1.0 1.0')
    
    parser.add_argument('--algorithm', type=str, default='ppo', help='Algorithm to use: ppo, ippo (instant ppo), dppo (discrete ppo), ddpg, iddpg (instant ddpg), mdqn, dmdqn, ddmdqn')

    parser.add_argument('--hack', type=int, default=None, help='Transfer learning agent, so every agent will loaded from this agent saved weights')

    # parser.add_argument('--independent', action='store_true', help='Independent')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    infer(algorithm=args.algorithm, n_agents=args.n_agents, resources=args.resources, tl_agent=args.hack, model=args.load_model, debug=args.debug, action_interval=args.action_interval, priorities=args.priorities)
