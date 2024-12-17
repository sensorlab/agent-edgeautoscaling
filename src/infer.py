import argparse
import time

import numpy as np

from envs import (ContinuousElasticityEnv, DiscreteElasticityEnv, InstantContinuousElasticityEnv, 
                  set_available_resource, set_other_priorities, set_other_utilization,
                  FiveDiscreteElasticityEnv, ElevenDiscrElasticityEnv)
from train_ddpg import DDPGagent
from train_mdqn import DQNAgent
from train_ppo import PPO


def initialize_agent(id=None, resources=1000, tl_agent=None, model=None, algorithm='ppo', independent=False,
                      priority=1.0, scale_action=None, pod_name=None):
    if not model:
        raise ValueError("Please provide a model to load")
    
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
            raise ValueError("Invalid algorithm")

    if discrete:
        env = DiscreteElasticityEnv(id, independent_state=independent, pod_name=pod_name)
    else:
        if instant:
            env = InstantContinuousElasticityEnv(id, independent_state=independent, pod_name=pod_name)
        else:
            env = ContinuousElasticityEnv(id, independent_state=independent, pod_name=pod_name)

    match algorithm:
        case 'ppo' | 'dppo' | 'ippo':
            agent = PPO(env, has_continuous_action_space=not discrete, action_std_init=1e-10, sigmoid_output=instant)
        case 'mdqn' | 'dmdqn':
            agent = DQNAgent(env)
        case 'ddmdqn':
            agent = DQNAgent(env, dueling=True)
        case 'ddpg' | 'iddpg':
            agent = DDPGagent(env, hidden_size=64, sigmoid_output=instant)

    if isinstance(tl_agent, int):
        agent.load(model, agent_id=tl_agent)
    else:
        agent.load(model, agent_id=id)

    env.MAX_CPU_LIMIT = resources
    env.priority = priority
    if scale_action:
        env.scale_action = scale_action

    return env, agent

def initialize_agents(n_agents=3, resources=1000, tl_agent=None, model=None, algorithm='ppo', independent=False,
                      priorities=[1.0, 1.0, 1.0], scale_action=None, five=False, eleven=False):
    if not model:
        raise ValueError("Please provide a model to load")
    
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
            raise ValueError("Invalid algorithm")

    if discrete:
        if five:
            envs = [FiveDiscreteElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]
        elif eleven:
            envs = [ElevenDiscrElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]
        else:
            envs = [DiscreteElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]
    else:
        if instant:
            envs = [InstantContinuousElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]
        else:
            envs = [ContinuousElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]

    match algorithm:
        case 'ppo' | 'dppo' | 'ippo':
            agents = [PPO(env, has_continuous_action_space=not discrete, action_std_init=1e-10, sigmoid_output=instant)
                      for env in envs]
        case 'mdqn' | 'dmdqn':
            agents = [DQNAgent(env) for env in envs]
        case 'ddmdqn':
            agents = [DQNAgent(env, dueling=True) for env in envs]
        case 'ddpg' | 'iddpg':
            agents = [DDPGagent(env, hidden_size=64, sigmoid_output=instant) for env in envs]

    for agent_id, agent in enumerate(agents):
        if isinstance(tl_agent, int):
            agent.load(model, agent_id=tl_agent)
        else:
            agent.load(model, agent_id=agent_id)

    for i, env in enumerate(envs):
        env.MAX_CPU_LIMIT = resources
        env.priority = priorities[i]
        if scale_action:
            env.scale_action = scale_action

    set_available_resource(envs, resources)

    return envs, agents


def infer(agents=None, envs=None, resources=None, debug=False, action_interval=None, shared_envs=None):
    if agents is None or envs is None or resources is None or action_interval is None:
        raise ValueError("Please provide agents, environments, resources and action interval")

    other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))]

    states = [np.array(env.reset()).flatten() for env in envs]
    while True:
        start_time = time.time()
        actions = [agent.get_action(state) for state, agent in zip(states, agents)]
        states, rewards, dones, _ = [], [], [], []
        for i, action in enumerate(actions):
            set_other_utilization(envs[i], other_envs[i])
            set_other_priorities(envs[i], other_envs[i])

            state, reward, done, _ = envs[i].step(action, 2)
            if shared_envs:
                shared_envs[i]['cummulative_delta'] = envs[i].cummulative_delta
                envs[i].priority = shared_envs[i]['priority']

            set_available_resource(envs, resources)
            states.append(np.array(state).flatten())
            rewards.append(reward)
            dones.append(done)
            if debug:
                print(
                    f"{envs[i].id}: ACTION: {action}, LIMIT: {envs[i].ALLOCATED}, {envs[i].last_cpu_percentage:.2f}%, "
                    f"AVAILABLE: {envs[i].AVAILABLE}, reward: {reward} state(limit, usage, others): {envs[i].state[-1]}")
        if debug:
            print()
        elapsed_time = time.time() - start_time
        if elapsed_time < action_interval:
            time.sleep(action_interval - elapsed_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--resources', type=int, default=1000)
    parser.add_argument('--load_model', type=str,
                        default='trained/ppo/1000ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources')  # Default trained weights for ppo model
    parser.add_argument('--action_interval', type=float, default=5.0)
    parser.add_argument('--priorities', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0],
                        help='List of priorities (0.0 < value <= 1.0), default is 1.0 for all agents. Example: 1.0 1.0 1.0')

    parser.add_argument('--algorithm', type=str, default='ppo',
                        help='Algorithm to use: ppo, ippo (instant ppo), dppo (discrete ppo), ddpg, iddpg (instant ddpg), mdqn, dmdqn, ddmdqn')

    parser.add_argument('--hack', type=int, default=None,
                        help='Transfer learning agent, so every agent will loaded from this agent saved weights')
    parser.add_argument('--scale_action', type=int, default=50,
                        help='Overwrite the scale action value for the agents')
    # parser.add_argument('--independent', action='store_true', help='Independent')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    envs, agents = initialize_agents(n_agents=args.n_agents, algorithm=args.algorithm, tl_agent=args.hack,
                                     model=args.load_model, priorities=args.priorities, resources=args.resources,
                                     scale_action=args.scale_action)

    infer(agents=agents, envs=envs, resources=args.resources, debug=args.debug, action_interval=args.action_interval)
