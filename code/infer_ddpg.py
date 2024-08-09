import numpy as np
import time
import argparse

from train_ddpg import DDPGagent, set_available_resource
from train_ppo import set_other_priorities, set_other_utilization
from pod_controller import set_container_cpu_values
from envs import ContinuousElasticityEnv, InstantContinuousElasticityEnv

def infer_ddpg(n_agents=3, resources=1000, instant=False, independent=False, hack=False, model=None, debug=False, action_interval=1, priority=0):
    if instant:
        envs = [InstantContinuousElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]
    else:
        envs = [ContinuousElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]

    other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))] # For every env its other envs (pre-computing)

    agents = [DDPGagent(env, hidden_size=64, sigmoid_output=instant) for env in envs]
    # model_folder = 'trained/continous/ddpg/600ep1000resources50rps1000interval0.6alpha50scale_a0.5gl'
    # model_folder = 'code/model_metric_data/ddpg/300ep1000resources50rps1000interval0.5alpha50scale_a0.5gl_pretrained'
    # model_folder = 'code/model_metric_data/ddpg/pretrained/100ep1000resources50rps1000interval0.75alpha50scale_a0.5gl'
    # model_folder = 'code/model_metric_data/ddpg/221ep_2rf_20rps5.0alpha_50scale1000resources'

    for id, agent in enumerate(agents):
        if hack:
            print(f'{model}/agent_{1}_actor.pth', f'{model}/agent_{1}_critic.pth')
            agent.load_model(f'{model}/agent_{1}_actor.pth', f'{model}/agent_{1}_critic.pth')
        else:
            print(f'{model}/agent_{id}_actor.pth', f'{model}/agent_{id}_critic.pth')
            agent.load_model(f'{model}/agent_{id}_actor.pth', f'{model}/agent_{id}_critic.pth')


    priorities = [1.0, 1.0, 1.0]
    match priority:
        case 1:
            priorities = [1.0, 0.1, 0.1]
        case 2:
            priorities = [0.1, 1.0, 0.1]
        case 3:
            priorities = [0.1, 0.1, 1.0]
        case _:
            print("Using default priority setting... [1, 1, ..., 1]")

    for i, env in enumerate(envs):
        env.DEBUG = False
        env.MAX_CPU_LIMIT = resources
        env.priority = priorities[i]

    set_container_cpu_values(100)
    set_available_resource(envs, resources)
    states = [np.array(env.reset()).flatten() for env in envs]
    while True:
        start_time = time.time()
        actions = [[agent.get_action(state)] for state, agent in zip(states, agents)]
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
    parser.add_argument('--priority', type=int, default=0, help="Options: 0, 1, 2... 0 means default")
    parser.add_argument('--resources', type=int, default=1000)
    parser.add_argument('--load_model', type=str, default='code/model_metric_data/ddpg/221ep_2rf_20rps5.0alpha_50scale1000resources')
    parser.add_argument('--action_interval', type=int, default=5)

    parser.add_argument('--instant', action='store_true', help='Instant')
    parser.add_argument('--independent', action='store_true', help='Independent')
    parser.add_argument('--hack', action='store_true', help='Transfer learning of agents for this model')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    n_agents = 3

    # set_container_cpu_values(100)

    infer_ddpg(n_agents=n_agents, resources=args.resources, instant=args.instant, independent=args.independent, hack=args.hack, model=args.load_model, debug=args.debug, action_interval=args.action_interval, priority=args.priority)