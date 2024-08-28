from train_ppo import PPO, set_available_resource, set_other_priorities, set_other_utilization
from envs import ContinuousElasticityEnv, DiscreteElasticityEnv, InstantContinuousElasticityEnv
from pod_controller import set_container_cpu_values
import numpy as np
import time
import argparse

def infer_ppo(n_agents=3, resources=1000, initial_action_std=1e-10, instant=False, discrete=False, independent=False, instant_hack=False, model=None, debug=False, action_interval=1, priorities=[1.0, 1.0, 1.0]):
    if discrete:
        envs = [DiscreteElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]
    else:
        if instant:
            envs = [InstantContinuousElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]
        else:
            envs = [ContinuousElasticityEnv(i, independent_state=independent) for i in range(1, n_agents + 1)]

    other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))] # For every env its other envs (pre-computing)

    agents = [PPO(env, has_continuous_action_space=not discrete, action_std_init=initial_action_std, sigmoid_output=instant) for env in envs]

    # model_folder = f'src/model_metric_data/ppo/52ep1000resources_rf_2_75rps1000interval9kepochs5.0alpha50scale_a0.5gl_NOreseting'
    # model_folder = f'src/model_metric_data/ppo/200ep_rf_2_60rps8kepochs5alpha_independent_state_instantscale_NOreseting_vari_res'
    # model_folder = f'src/model_metric_data/ppo/66ep_rf_2_30rps8kepochs5alpha50scale_a0priority_newloading_instantscale_NOreseting_vari_res_pretrained' # yoink the first or second agent only
    # model_folder = f'src/model_metric_data/ppo/210ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources'
    # model_folder = 'trained/ppo/100ep1000resources_rf_2_75rps1000interval9kepochs5.0alpha50scale_a_instantscale_NOreseting_vari_res'
    # model_folder = 'trained/ppo/100ep_rf_2_30rps10kepochs5alpha50scale_a0priority_newloading_instantscale_NOreseting_vari_res'
    # model_folder = 'trained/ppo/400ep_rf_2_30rps8kepochs5alpha50scale_a0priority_newloading_instantscale_NOreseting_vari_res'

    if not model:
        print("Please provide a model to load")
        return

    for id, agent in enumerate(agents):
        if instant_hack:
            agent.load(f'{model}/agent_{1}.pth')
            print(f'Loaded {model}/agent_{1}.pth')
        else:
            agent.load(f'{model}/agent_{id}.pth')
            print(f'Loaded {model}/agent_{id}.pth')

    for i, env in enumerate(envs):
        env.DEBUG = False
        agents[i].policy.eval() # set to evaluation mode
        env.MAX_CPU_LIMIT = resources
        env.INCREMENT = 25
        env.priority = priorities[i]
        env.scale_action = 100

    set_available_resource(envs, resources)
    states = [np.array(env.reset()).flatten() for env in envs]
    while True:
        # time.sleep(1)
        start_time = time.time()
        # actions = [agent.select_inference_action(state) for state, agent in zip(states, agents)]
        actions = [agent.select_action(state) for state, agent in zip(states, agents)]
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
    parser.add_argument('--priority', type=int, default=0, help="Options: 0, 1, 2... 0 means to default priority")
    parser.add_argument('--resources', type=int, default=1000)
    parser.add_argument('--load_model', type=str, default='src/model_metric_data/ppo/610ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources_pretrained')
    parser.add_argument('--action_interval', type=int, default=5)
    parser.add_argument('--priorities', type=float, nargs='+', default=[1.0, 1.0, 1.0], help='List of priorities')

    parser.add_argument('--instant', action='store_true', help='Instant')
    parser.add_argument('--discrete', action='store_true', help='Discrete')
    parser.add_argument('--independent', action='store_true', help='Independent')
    parser.add_argument('--instant_hack', action='store_true', help='Instant hack')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    n_agents = 3
    initial_action_std = 1e-10

    # set_container_cpu_values(100)

    infer_ppo(n_agents=n_agents, resources=args.resources, initial_action_std=initial_action_std, instant=args.instant, discrete=args.discrete, independent=args.independent, instant_hack=args.instant_hack, model=args.load_model, debug=args.debug, action_interval=args.action_interval, priorities=args.priorities)
