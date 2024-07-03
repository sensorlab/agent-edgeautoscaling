import numpy as np

from gymnasium import spaces
from env import ElastisityEnv
from pod_controller import patch_pod

class ContinousElasticityEnv(ElastisityEnv):
    def __init__(self, id, n_agents):
        super().__init__(id, n_agents)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(1), shape=(self.STATE_LENTGH * len(self.state[0]),))
        self.scale_action = 50
        self.UPPER_CPU = 80
        self.LOWER_CPU = 40
        self.MIN_CPU_LIMIT = 25

    def step(self, action):
        self.state = self.get_current_usage()

        scale_action = action[0] * self.scale_action
        if scale_action < self.AVAILABLE:
            new_resource_limit = int(max(self.ALLOCATED + scale_action, self.MIN_CPU_LIMIT))
            self.ALLOCATED = new_resource_limit
            patch_pod(f'localization-api{self.id}', cpu_request=f"{new_resource_limit}m", cpu_limit=f"{new_resource_limit}m", container_name='localization-api', debug=True)

        reward = 0
        if self.LOWER_CPU <= self.last_cpu_percentage <= self.UPPER_CPU:
            reward = self.last_cpu_percentage / 100.0

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0

    def reset(self):
        cpu_limit = 50
        patch_pod(f'localization-api{self.id}', cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m", container_name='localization-api', debug=True)
        self.ALLOCATED = cpu_limit
        
        return super().reset()
