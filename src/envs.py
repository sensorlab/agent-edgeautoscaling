import numpy as np

from gymnasium import Env, spaces

from utils import init_nodes
from pod_controller import patch_pod


class BaseElasticityEnv(Env):
    def __init__(self, id, independent_state=False):
        super().__init__()
        # Targets pod localization-api1 with container localization-api
        self.container_name = 'localization-api'
        self.app_label = f'app={self.container_name}'
        self.pod_name = f'{self.container_name}{id}'

        self.DEBUG = False

        self.MAX_CPU_LIMIT = 1000
        self.MIN_CPU_LIMIT = 50
        self.INCREMENT = 50

        self.ALLOCATED = 50
        self.AVAILABLE = 1000

        self.independent_state = independent_state

        self.debug_deployment = True
        nodes = init_nodes(debug=self.debug_deployment, custom_label=self.app_label)

        self.id = id
        for node in nodes:
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                if pod_name == f'{self.pod_name}':
                    # grab node object and container_id for agent
                    self.container_id = container_id
                    self.node = node
                    break

        self.other_util = 0.0
        self.STATE_LENTGH = 6
        if self.independent_state:
            self.states_fifo = [[0, 0, 0, 0, 0] for _ in range(self.STATE_LENTGH)]
        else:
            self.states_fifo = [[0, 0, 0, 0, 0, 0, 0] for _ in range(self.STATE_LENTGH)]

        self.last_cpu_percentage = 0
        self.previous_cpu_percentage = 0
        self.priority = 1.0
        self.other_priorities = 0.0
        
        self.state = self.get_current_usage()
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(1), shape=(self.STATE_LENTGH * len(self.state[0]),))

        self.steps = 0
        self.MAX_STEPS = 60

        # 30% - 60%
        self.UPPER_CPU = 60
        self.LOWER_CPU = 30
        
        self.dqn_reward = False

    def norm_cpu(self, cpu_usage):
        return cpu_usage / self.MAX_CPU_LIMIT

    def get_current_usage(self):
        (cpu_limit, cpu, cpu_percentage), (memory_limit, memory, memory_percentage), (rx, tx), throttled = self.node.get_container_usage(self.container_id)
        self.previous_cpu_percentage = self.last_cpu_percentage
        self.last_cpu_percentage = cpu_percentage
        n_cpu_limit, n_cpu = self.norm_cpu(cpu_limit), self.norm_cpu(cpu)

        available_normed = self.norm_cpu(self.AVAILABLE)
        # state = [n_cpu_limit, n_cpu, available_normed]
        if self.independent_state:
            state = [n_cpu_limit, n_cpu, available_normed, cpu_percentage / 100, self.priority]
        else:
            state = [n_cpu_limit, n_cpu, available_normed, cpu_percentage / 100, self.other_util / 100, self.priority, self.other_priorities]
        # state = [n_cpu_limit, n_cpu, available_normed, self.last_cpu_percentage / 100]

        self.states_fifo.append(state)
        self.states_fifo.pop(0)
        if self.DEBUG:
            print(f'Agent {self.id}, LIMIT: {cpu_limit}, AVAILABLE: {self.AVAILABLE}, state(limit, usage, others): {state}') 
        return self.states_fifo
    
    def reset(self):
        self.state = self.get_current_usage()
        self.steps = 0
        
        # Fill state with the last value
        self.state = [self.state[-1]] * self.STATE_LENTGH

        return self.state

    def set_last_limit(self):
        patch_pod(self.pod_name, cpu_request=f"{self.ALLOCATED}m", cpu_limit=f"{self.ALLOCATED}m", container_name=self.container_name, debug=self.debug_deployment)

    def patch(self, limit):
        patch_pod(self.pod_name, cpu_request=f"{limit}m", cpu_limit=f"{limit}m", container_name=self.container_name, debug=self.debug_deployment)
        self.ALLOCATED = limit

    def get_container_usage(self):
        return self.node.get_container_usage(self.container_id)

    def calculate_agent_reward(self, rf):
        reward = 0
        match rf:
            case 1:
                if self.last_cpu_percentage <= self.LOWER_CPU:
                    reward = -1
                return reward
            case 2 | 3 | 42:
                if self.LOWER_CPU <= self.last_cpu_percentage <= self.UPPER_CPU:
                    reward = 1 + (self.last_cpu_percentage / (self.UPPER_CPU - self.LOWER_CPU))
                elif self.last_cpu_percentage < self.LOWER_CPU:
                    # Last = current
                    delta = self.last_cpu_percentage - self.previous_cpu_percentage
                    reward = min(delta / 10, 1) if delta > 0 else max(delta / 10, -1)
                    # So, if from 10 to 15, reward is 0.5. and from 40 to 10, reward is -1
                # elif self.last_cpu_percentage > self.UPPER_CPU:
                #     curr_percentage = min(self.last_cpu_percentage, 100) # Can be higher than 100, so we avoid that
                #     reward = 1 - 2 * ((curr_percentage - self.UPPER_CPU) / (100 - self.UPPER_CPU))
                return reward
            case 4 | 5:
                if self.last_cpu_percentage < self.LOWER_CPU:
                    # Last = current
                    delta = self.last_cpu_percentage - self.previous_cpu_percentage
                    reward = min(delta / 10, 1) if delta > 0 else max(delta / 10, -1)
                    # So, if from 10 to 15, reward is 0.5. and from 40 to 10, reward is -1
                return reward

        # if self.dqn_reward:
        #     if self.last_cpu_percentage < self.LOWER_CPU:
        #         # usage_penalty = 1.3 - self.last_cpu_percentage / 100
        #         usage_penalty = 0.75 - self.last_cpu_percentage / 100 # lower penalty on this
        #     elif self.last_cpu_percentage > self.UPPER_CPU:
        #         usage_penalty = self.last_cpu_percentage / 100
        #     else:
        #         usage_penalty = 0
        #     reward = - usage_penalty
        # else:
        #     reward = 0
        #     if self.LOWER_CPU <= self.last_cpu_percentage <= self.UPPER_CPU:
        #         # reward = (self.last_cpu_percentage - self.LOWER_CPU) / (self.UPPER_CPU - self.LOWER_CPU) # map 0 to 1
        #         # reward = 1
        #         reward = 1 + (self.last_cpu_percentage / (self.UPPER_CPU - self.LOWER_CPU))
        #     elif self.last_cpu_percentage < self.LOWER_CPU:
        #         # Last = current
        #         delta = self.last_cpu_percentage - self.previous_cpu_percentage
        #         reward = min(delta / 10, 1) if delta > 0 else max(delta / 10, -1)
        #         # So, if from 10 to 15, reward is 0.5. and from 40 to 10, reward is -1
        #         # if delta > 0:
        #         #     reward = min(delta / 10, 1)
        #         # else:
        #         #     reward = max(delta / 10, -1)
        #     elif self.last_cpu_percentage > self.UPPER_CPU:
        #         curr_percentage = min(self.last_cpu_percentage, 100) # Can be higher than 100, so we avoid that
        #         reward = 1 - 2 * ((curr_percentage - self.UPPER_CPU) / (100 - self.UPPER_CPU))
        # return reward


class DiscreteElasticityEnv(BaseElasticityEnv):
    def __init__(self, id, independent_state=False):
        super().__init__(id, independent_state=independent_state)
        self.action_space = spaces.Discrete(3)

    def step(self, action, rf):
        if action == 0:
            self.decrease_resources()
        elif action == 2:
            self.increase_resources()

        self.state = self.get_current_usage()

        reward = self.calculate_agent_reward(rf)

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0
    
    def increase_resources(self):
        # cpu_limit, memory_limit = self.node.get_container_limits(self.container_id)
        updated_cpu_limit = int(max(min(self.ALLOCATED + self.INCREMENT, self.ALLOCATED + self.AVAILABLE), self.MIN_CPU_LIMIT))
        self.ALLOCATED = updated_cpu_limit
        patch_pod(self.pod_name, cpu_request=f"{updated_cpu_limit}m", cpu_limit=f"{updated_cpu_limit}m", container_name=self.container_name, debug=self.debug_deployment)

    def decrease_resources(self):
        # cpu_limit, memory_limit = self.node.get_container_limits(self.container_id)
        updated_cpu_limit = int(max(self.ALLOCATED - self.INCREMENT, self.MIN_CPU_LIMIT))
        self.ALLOCATED = updated_cpu_limit
        patch_pod(self.pod_name, cpu_request=f"{updated_cpu_limit}m", cpu_limit=f"{updated_cpu_limit}m", container_name=self.container_name, debug=self.debug_deployment)


class ContinuousElasticityEnv(BaseElasticityEnv):
    def __init__(self, id, independent_state=False):
        super().__init__(id, independent_state=independent_state)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.scale_action = 50

    def step(self, action, rf):
        self.state = self.get_current_usage()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        scale_action = action[0] * self.scale_action
        if scale_action <= max(self.AVAILABLE, 0): # If available is negative
            new_resource_limit = int(max(self.ALLOCATED + scale_action, self.MIN_CPU_LIMIT))
            self.ALLOCATED = new_resource_limit
            patch_pod(self.pod_name, cpu_request=f"{new_resource_limit}m", cpu_limit=f"{new_resource_limit}m", container_name=self.container_name, debug=self.debug_deployment)

        reward = self.calculate_agent_reward(rf)

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0

    def mimic_step(self):
        self.state = self.get_current_usage()

        reward = self.calculate_agent_reward()

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0


class InstantContinuousElasticityEnv(BaseElasticityEnv):
    def __init__(self, id, independent_state=False):
        super().__init__(id, independent_state=independent_state)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.scale_action = 50 # Placeholder

    def step(self, action, rf):
        self.state = self.get_current_usage()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        scale_action = action[0] * self.MAX_CPU_LIMIT

        # if scale_action < self.ALLOCATED or (self.ALLOCATED - scale_action) <= max(self.AVAILABLE, 0):
        if (scale_action - self.ALLOCATED) <= max(self.AVAILABLE, 0):
            new_resource_limit = int(max(scale_action, self.MIN_CPU_LIMIT))
            self.ALLOCATED = new_resource_limit
            patch_pod(
                self.pod_name,
                cpu_request=f"{new_resource_limit}m",
                cpu_limit=f"{new_resource_limit}m",
                container_name=self.container_name,
                debug=self.debug_deployment
            )
        
        reward = self.calculate_agent_reward(rf)
        self.steps += 1
        done = self.steps >= self.MAX_STEPS
        return self.state, reward, done, 0
