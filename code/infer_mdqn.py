import torch
import numpy as np
import time

from itertools import count

from train_mdqn import DQN
from env import ElastisityEnv

from pod_controller import set_container_cpu_values
from train_mdqn import set_available_resource

def infer_mdqn(n_agents=3, model='mdqn300ep500m'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_agents = 3
    envs = [ElastisityEnv(i) for i in range(1, n_agents + 1)]
    state = envs[0].reset()
    n_actions = envs[0].action_space.n
    n_observations = len(state) * len(state[0])

    agents = [DQN(n_observations, n_actions).to(device) for _ in range(n_agents)]

    for i, agent in enumerate(agents):
        agent.load_state_dict(torch.load(f'trained/{model}/model_weights_agent_{i}.pth'))
        # agent.load_state_dict(torch.load(f'code/model_metric_data/{model}/model_weights_agent_{i}.pth'))
        print(f'Loaded weights for agent {i}')
        agent.eval()
    
    # get paremeters from model folder name
    INITIAL_RESOURCES = 1000
    INCREMENT_ACTION = 50
    for env in envs:
        env.MAX_CPU_LIMIT = INITIAL_RESOURCES
        env.INCREMENT = INCREMENT_ACTION

    set_available_resource(envs, INITIAL_RESOURCES)
    print(f"Loaded model with parameters: initial resources: {INITIAL_RESOURCES}, increment action: {INCREMENT_ACTION}, n_agents: {n_agents}")

    states = [env.reset() for env in envs]
    states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0) for state in states]

    while True:
        # states = [env.reset() for env in envs]
        # states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0) for state in states]

        # for t in count():
        time.sleep(1)

        with torch.no_grad():
            actions = [dqn(state).max(1).indices.view(1, 1) for dqn, state in zip(agents, states)]

        next_states, rewards, dones = [], [], []
        for i, action in enumerate(actions):
            observation, reward, done, _ = envs[i].step(action.item())
            set_available_resource(envs, INITIAL_RESOURCES)
            next_states.append(np.array(observation).flatten())
            rewards.append(reward)
            dones.append(done)
            # if done:
            #     next_states[i] = None

        # print(envs[0].state[-3])
        states = [torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if observation is not None else None for observation in next_states]

        # if any(dones):
        #     break

if __name__ == "__main__":
    set_container_cpu_values(50)
    infer_mdqn(3, 'testing_models/mdqn250ep500m50inc1000mcmax145rps0.6alpha')
