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
        self.UPPER_CPU = 60
        self.LOWER_CPU = 30
        self.MIN_CPU_LIMIT = 50

        self.dqn_reward = False

    def step(self, action):
        self.state = self.get_current_usage()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        scale_action = action[0] * self.scale_action
        if scale_action <= self.AVAILABLE:
            new_resource_limit = int(max(self.ALLOCATED + scale_action, self.MIN_CPU_LIMIT))
            self.ALLOCATED = new_resource_limit
            patch_pod(f'localization-api{self.id}', cpu_request=f"{new_resource_limit}m", cpu_limit=f"{new_resource_limit}m", container_name='localization-api', debug=True)

        if self.dqn_reward:
            # from dqn
            if self.last_cpu_percentage < self.LOWER_CPU:
                # usage_penalty = 1.3 - self.last_cpu_percentage / 100
                usage_penalty = 0.75 - self.last_cpu_percentage / 100 # lower penalty on this
            elif self.last_cpu_percentage > self.UPPER_CPU:
                usage_penalty = self.last_cpu_percentage / 100
            else:
                usage_penalty = 0
            reward = - usage_penalty
        else:
            reward = 0
            if self.LOWER_CPU <= self.last_cpu_percentage <= self.UPPER_CPU:
                # reward = (self.last_cpu_percentage - self.LOWER_CPU) / (self.UPPER_CPU - self.LOWER_CPU) # map 0 to 1
                reward = 1

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0

    def reset(self):
        # cpu_limit = 100
        # patch_pod(f'localization-api{self.id}', cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m", container_name='localization-api', debug=True)
        # self.ALLOCATED = cpu_limit
        
        return super().reset()

    def mimic_step(self):
        self.state = self.get_current_usage()

        if self.dqn_reward:
            # from dqn
            if self.last_cpu_percentage < self.LOWER_CPU:
                # usage_penalty = 1.3 - self.last_cpu_percentage / 100
                usage_penalty = 0.75 - self.last_cpu_percentage / 100 # lower penalty on this
            elif self.last_cpu_percentage > self.UPPER_CPU:
                usage_penalty = self.last_cpu_percentage / 100
            else:
                usage_penalty = 0
            reward = - usage_penalty
        else:
            reward = 0
            if self.LOWER_CPU <= self.last_cpu_percentage <= self.UPPER_CPU:
                reward = (self.last_cpu_percentage - self.LOWER_CPU) / (self.UPPER_CPU - self.LOWER_CPU) # map 0 to 1

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0
