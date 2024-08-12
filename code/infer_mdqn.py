import torch
import numpy as np
import time
import argparse

from itertools import count

from train_mdqn import DQN, set_available_resource, DuelingDQN
from envs import DiscreteElasticityEnv
from pod_controller import set_container_cpu_values

from train_ppo import set_other_priorities, set_other_utilization


def infer_mdqn(n_agents=3, model='mdqn300ep500m', resources=1000, increment=25, debug=True, priorities=[1.0, 1.0, 1.0], action_interval=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = [DiscreteElasticityEnv(i, independent_state=False) for i in range(1, n_agents + 1)]
    
    other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))] # For every env its other envs (pre-computing)
    
    state = envs[0].reset()
    n_actions = envs[0].action_space.n
    n_observations = len(state) * len(state[0])

    # agents = [DQN(n_observations, n_actions).to(device) for _ in range(n_agents)]
    agents = [DuelingDQN(n_observations, n_actions).to(device) for _ in range(n_agents)]

    for i, agent in enumerate(agents):
        # agent.load_state_dict(torch.load(f'trained/{model}/model_weights_agent_{i}.pth'))
        # agent.load_state_dict(torch.load(f'code/model_metric_data/{model}/model_weights_agent_{i}.pth'))
        agent.load_state_dict(torch.load(f'{model}/model_weights_agent_{2}.pth'))
        print(f'Loaded weights for agent {i}')
        agent.eval()
    
    # get paremeters from model folder name
    for i, env in enumerate(envs):
        # env.DEBUG = debug
        env.MAX_CPU_LIMIT = resources
        env.INCREMENT = increment
        env.priority = priorities[i]

    set_available_resource(envs, resources)
    print(f"Loaded model with parameters: initial resources: {resources}, increment action: {increment}, n_agents: {n_agents}")

    states = [env.reset() for env in envs]
    states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0) for state in states]

    while True:
        # time.sleep(1)
        start_time = time.time()
        with torch.no_grad():
            actions = [dqn(state).max(1).indices.view(1, 1) for dqn, state in zip(agents, states)]

        next_states, rewards, dones = [], [], []
        for i, action in enumerate(actions):
            set_other_utilization(envs[i], other_envs[i])
            set_other_priorities(envs[i], other_envs[i])

            observation, reward, done, _ = envs[i].step(action.item(), 2)
            set_available_resource(envs, resources)
            next_states.append(np.array(observation).flatten())
            rewards.append(reward)
            dones.append(done)

            if debug:
                print(f"{envs[i].id}: ACTION: {action}, LIMIT: {envs[i].ALLOCATED}, {envs[i].last_cpu_percentage:.2f}%, AVAILABLE: {envs[i].AVAILABLE}, reward: {reward} state(limit, usage, others): {envs[i].state[-1]}")
        if debug:
            print()
        elapsed_time = time.time() - start_time
        if elapsed_time < action_interval:
            time.sleep(action_interval - elapsed_time)

        states = [torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if observation is not None else None for observation in next_states]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer MDQN')
    parser.add_argument('--resources', type=int, default=1000, help='Initial resources')
    parser.add_argument('--increment', type=int, default=25, help='Increment action')
    parser.add_argument('--action_interval', type=int, default=5)

    parser.add_argument('--debug', action='store_true', help='Debug')
    parser.add_argument('--priorities', type=float, nargs='+', default=[1.0, 1.0, 1.0], help='List of priorities')
    # parser.add_argument('--priority', type=int, default=0, help="Options: 0, 1, 2... 0 means default priority")
    # parser.add_argument('--load_model', type=str, default='code/model_metric_data/dqn/mdqn310ep1000m25inc2_rf_20rps5.0alpha1000res_double_dueling', help='Model to load') # Best model so far
    parser.add_argument('--load_model', type=str, default='code/model_metric_data/dqn/mdqn610ep1000m25inc2_rf_20rps5.0alpha1000res_double_dueling_pretrained', help='Model to load')
    
    args = parser.parse_args()

    # set_container_cpu_values(100)
    # model = 'variational_loading/variational_resources/mdqn1000ep500m25inc1000mcmax40rps500interval0.75alpha_double_dueling_varres'
    # model = 'variational_loading/variational_resources/mdqn600ep500m25inc1000mcmax50rps500interval0.75alpha_double_dueling' # best model so far
    # model = 'new_reward/dqn/mdqn300ep500m25inc1000mcmax50rps1000interval0.5alpha0.5gl_double_dueling_varres' # good, better changeable resources
    # model = 'dqn/mdqn310ep1000m25inc2_rf_20rps5.0alpha1000res_double_dueling'
    # model = 'code/model_metric_data/dqn/mdqn310ep1000m25inc2_rf_20rps5.0alpha1000res_double_dueling'

    # priorities = [1.0, 1.0, 1.0]
    # match args.priority:
    #     case 1:
    #         priorities = [1.0, 0.1, 0.1]
    #     case 2:
    #         priorities = [0.1, 1.0, 0.1]
    #     case 3:
    #         priorities = [0.1, 0.1, 1.0]
    #     case _:
    #         print("Using default priority setting... [1, 1, ..., 1]")

    # ?????? This doesnt make any sense why it works
    # priorities = [1.0, 1.0, 1.0]
    # match args.priority:
    #     case 1:
    #         priorities = [0.1, 1.0, 1.0]
    #     case 2:
    #         priorities = [1.0, 0.1, 1.0]
    #     case 3:
    #         priorities = [1.0, 1.0, 0.1]
    #     case _:
    #         print("Using default priority setting... [1, 1, ..., 1]")

    infer_mdqn(3, args.load_model, args.resources, args.increment, args.debug, priorities=args.priorities, action_interval=args.action_interval)
    # infer_mdqn(3, 'variational_loading/variational_resources/variational_intervals/mdqn600ep500m25inc1000mcmax50rps500interval0.75alpha_double_dueling_pretrained', args.resources, args.increment, args.debug)

    # infer_mdqn(3, 'variational_loading/mdqn600ep500m25inc500mcmax140rps0.5alpha_double_dueling')
    # infer_mdqn(3, 'variational_loading/mdqn300ep500m25inc500mcmax90rps0.75alpha_double_dueling_pretrained')
    # infer_mdqn(3, 'variational_loading/variational_resources/mdqn600ep500m25inc1000mcmax50rps500interval0.75alpha_double_dueling')
