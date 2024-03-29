import os
import torch
import time

from itertools import count

from train_dqn import ReplayMemory, DQN
from dqn_env import ElastisityEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = ElastisityEnv(1)
state = env.reset()

n_actions = env.action_space.n
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)

model = 'trained/dqn/weights_150ep.pth'
if os.path.isfile(model):
    policy_net.load_state_dict(torch.load(model))
    print('Loaded model weights')
else:
    print('No model weights found')

def infer(policy_net, state):
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)

state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

while True:
    for t in count():
        time.sleep(1)
        action = infer(policy_net, state)
        observation, reward, done, _ = env.step(action.item())

        if done:
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            break

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
