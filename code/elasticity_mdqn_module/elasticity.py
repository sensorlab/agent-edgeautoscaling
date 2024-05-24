import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import threading

from fastapi import FastAPI

from pod_controller import set_container_cpu_values
from env import ElastisityEnv

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DuelingDQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

def set_available_resource(envs, initial_resources):
    max_group = initial_resources
    for env in envs:
        max_group -= env.ALLOCATED
    for env in envs:
        env.AVAILABLE = max_group

def infer_mdqn(n_agents=3, stop_signal=None, resources=1000, increment=25, debug=True):
    
    states = [env.reset() for env in envs]
    states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=device).unsqueeze(0) for state in states]
    while not stop_signal.is_set():
        time.sleep(1)

        with torch.no_grad():
            actions = [dqn(state).max(1).indices.view(1, 1) for dqn, state in zip(agents, states)]

        next_states, rewards, dones = [], [], []
        for i, action in enumerate(actions):
            observation, reward, done, _ = envs[i].step(action.item())
            set_available_resource(envs, resources)
            next_states.append(np.array(observation).flatten())
            rewards.append(reward)
            dones.append(done)
            # if done:
            #     next_states[i] = None

        # print(envs[0].state[-3])
        states = [torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if observation is not None else None for observation in next_states]

        if stop_signal.is_set():
            break


# set pods initial values
set_container_cpu_values(cpus=100)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_agents = 3
envs = [ElastisityEnv(i, n_agents) for i in range(1, n_agents + 1)]
state = envs[0].reset()
n_actions = envs[0].action_space.n
n_observations = len(state) * len(state[0])

agents = [DuelingDQN(n_observations, n_actions).to(device) for _ in range(n_agents)]

resources = 1000
increment = 25
debug = True
for env in envs:
    env.DEBUG = debug
    env.MAX_CPU_LIMIT = resources
    env.INCREMENT = increment
set_available_resource(envs, resources)

for i, agent in enumerate(agents):
    agent.load_state_dict(torch.load(f'model_weights_agent_{i}.pth'))
    print(f'Loaded weights for agent {i}')
    agent.eval()

print(f"Loaded model with parameters: initial resources: {resources}, increment action: {increment}, n_agents: {n_agents}")

app = FastAPI()
infer_thread = None
stop_signal = threading.Event()

@app.post("/start")
def start_inference():
    global infer_thread, stop_signal
    if infer_thread is None or not infer_thread.is_alive():
        stop_signal.clear()
        infer_thread = threading.Thread(target=infer_mdqn, args=(3, stop_signal, resources, increment))
        infer_thread.start()
        return {"message": "Inference started"}
    else:
        return {"message": "Inference already running"}

@app.post("/stop")
def stop_inference():
    global infer_thread, stop_signal
    if infer_thread is not None and infer_thread.is_alive():
        stop_signal.set()
        return {"message": "Inference stopped"}
    else:
        return {"message": "Inference not running"}
