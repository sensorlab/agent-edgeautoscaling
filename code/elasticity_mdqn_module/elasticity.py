import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import threading
from pydantic import BaseModel

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

class Application:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_agents = 3
        self.envs = [ElastisityEnv(i, self.n_agents) for i in range(1, self.n_agents + 1)]
        state = self.envs[0].reset()
        n_actions = self.envs[0].action_space.n
        n_observations = len(state) * len(state[0])
        self.agents = [DuelingDQN(n_observations, n_actions).to(self.device) for _ in range(self.n_agents)]
        self.resources = 1000
        self.increment = 25
        self.debug = True
        for env in self.envs:
            env.DEBUG = self.debug
            env.MAX_CPU_LIMIT = self.resources
            env.INCREMENT = self.increment
        self.set_available_resource(self.resources)
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(torch.load(f'model_weights_agent_{i}.pth'))
            agent.eval()
        print(f"Loaded model with parameters: initial resources: {self.resources}, increment action: {self.increment}, n_agents: {self.n_agents}")
        self.infer_thread = None
        self.stop_signal = threading.Event()
    
    def infer_mdqn(self):
        states = [env.reset() for env in self.envs]
        states = [torch.tensor(np.array(state).flatten(), dtype=torch.float32, device=self.device).unsqueeze(0) for state in states]
        while not self.stop_signal.is_set():
            time.sleep(1)

            with torch.no_grad():
                actions = [dqn(state).max(1).indices.view(1, 1) for dqn, state in zip(self.agents, states)]

            next_states, rewards, dones = [], [], []
            for i, action in enumerate(actions):
                observation, reward, done, _ = self.envs[i].step(action.item())
                self.set_available_resource(self.resources)
                next_states.append(np.array(observation).flatten())
                rewards.append(reward)
                dones.append(done)

            states = [torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0) if observation is not None else None for observation in next_states]

            if self.stop_signal.is_set():
                break
    
    def set_available_resource(self, resources):
        max_group = resources
        for env in self.envs:
            max_group -= env.ALLOCATED
        for env in self.envs:
            env.AVAILABLE = max_group

    def start_inference(self):
        if self.infer_thread is None or not self.infer_thread.is_alive():
            self.stop_signal.clear()
            self.infer_thread = threading.Thread(target=self.infer_mdqn)
            self.infer_thread.start()
            return {"message": "Inference started"}
        else:
            return {"message": "Inference already running"}

    def stop_inference(self):
        if self.infer_thread is not None and self.infer_thread.is_alive():
            self.stop_signal.set()
            return {"message": "Inference stopped"}
        else:
            return {"message": "Inference not running"}

    def set_resources(self, new_resources):
        self.resources = new_resources
        for env in self.envs:
            env.MAX_CPU_LIMIT = new_resources
        self.set_available_resource(new_resources)
        print(f"Resources set to {new_resources}")
        return {"message": f"Resources set to {new_resources}"}

app = Application()
fastapi_app = FastAPI()

class Item(BaseModel):
    resources: int

@fastapi_app.post("/start")
def start_inference():
    return app.start_inference()

@fastapi_app.post("/stop")
def stop_inference():
    return app.stop_inference()

@fastapi_app.post("/set_resources")
def set_resources(item: Item):
    return app.set_resources(item.resources)
