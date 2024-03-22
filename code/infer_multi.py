import torch
import numpy as np
import time

from itertools import count

from train_mdqn import DQN
from env import ElastisityEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_agents = 3
envs = [ElastisityEnv(i) for i in range(1, n_agents + 1)]
state = envs[0].reset()
n_actions = envs[0].action_space.n
n_observations = len(state) * len(state[0])

agents = [DQN(n_observations, n_actions).to(device) for _ in range(n_agents)]

for i, agent in enumerate(agents):
    agent.load_state_dict(torch.load(f'code/models/mdqn/model_weights_agent_{i}.pth'))
    agent.eval()

for i_episode in range(5):
    states = [env.reset() for env in envs]
    states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0) for state in states]

    for t in count():
        time.sleep(1)

        actions = [dqn(state).max(1).indices.view(1, 1) for dqn, state in zip(agents, states)]

        next_states, rewards, dones = [], [], []
        for i, action in enumerate(actions):
            observation, reward, done, _ = envs[i].step(action.item())
            next_states.append(np.array(observation).flatten())
            rewards.append(reward)
            dones.append(done)
            if done:
                next_states[i] = None

        states = [torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if observation is not None else None for observation in next_states]

        if any(dones):
            break
