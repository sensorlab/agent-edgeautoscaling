import numpy as np

from gymnasium import spaces
from env import ElastisityEnv
from pod_controller import patch_pod

class ContinousElasticityEnv(ElastisityEnv):
    def __init__(self, id, n_agents):
        super().__init__(id, n_agents)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(1), shape=(self.STATE_LENTGH * len(self.state[0]),))

    def step(self, action):
        self.state = self.get_current_usage()

        resource_penalty = 0
        new_resource_limit = int(max(action[0] * self.MAX_CPU_LIMIT, self.MIN_CPU_LIMIT))

        # If the new resource limit is greater than the available resources, set it to the maximum available
        # and give some penalty for overshooting
        if new_resource_limit > self.AVAILABLE:
            resource_penalty = 0.5
            new_resource_limit = self.ALLOCATED + self.AVAILABLE
            self.ALLOCATED = new_resource_limit
            patch_pod(f'localization-api{self.id}', cpu_request=f"{new_resource_limit}m", cpu_limit=f"{new_resource_limit}m", container_name='localization-api', debug=True)
        else:
            self.ALLOCATED = new_resource_limit
            patch_pod(f'localization-api{self.id}', cpu_request=f"{new_resource_limit}m", cpu_limit=f"{new_resource_limit}m", container_name='localization-api', debug=True)

        if self.last_cpu_percentage < self.LOWER_CPU:
            # usage_penalty = 1.3 - self.last_cpu_percentage / 100
            usage_penalty = 0.75 - self.last_cpu_percentage / 100 # lower penalty on this
        elif self.last_cpu_percentage > self.UPPER_CPU:
            usage_penalty = self.last_cpu_percentage / 100
        else:
            usage_penalty = 0


        reward = - usage_penalty - resource_penalty
        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0
