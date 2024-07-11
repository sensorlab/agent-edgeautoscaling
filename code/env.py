import numpy as np

from gymnasium import Env, spaces

from utils import init_nodes
from pod_controller import patch_pod


# todo: add label for different applications
class ElastisityEnv(Env):
    def __init__(self, id, n_agents):
        super().__init__()
        self.DEBUG = False

        self.n_agents = n_agents
        self.MAX_CPU_LIMIT = 1000
        self.MIN_CPU_LIMIT = 50
        self.INCREMENT = 50

        self.ALLOCATED = 50
        self.AVAILABLE = 1000

        self.action_space = spaces.Discrete(3)
        nodes = init_nodes(debug=True, custom_label='app=localization-api')

        self.id = id
        for node in nodes:
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                if pod_name == f'localization-api{self.id}':
                    # grab node object and container_id for agent
                    self.container_id = container_id
                    self.node = node
                    break

        self.other_avg_util = 0.0

        self.STATE_LENTGH = 6
        self.states_fifo = [[0, 0, 0, False, 0] for _ in range(self.STATE_LENTGH)] # init state
        self.state = self.get_current_usage()

        self.steps = 0
        self.MAX_STEPS = 60

        # 30% - 60%
        self.UPPER_CPU = 60
        self.LOWER_CPU = 30

    def step(self, action):
        if action == 0:
            self.decrease_resources()
        elif action == 2:
            self.increase_resources()

        self.state = self.get_current_usage()

        if self.last_cpu_percentage < self.LOWER_CPU:
            # usage_penalty = 1.3 - self.last_cpu_percentage / 100
            usage_penalty = 0.75 - self.last_cpu_percentage / 100 # lower penalty on this
        elif self.last_cpu_percentage > self.UPPER_CPU:
            usage_penalty = self.last_cpu_percentage / 100
        else:
            usage_penalty = 0

        reward = - usage_penalty

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0
    
    def reset(self):
        # random resetting, not needed
        # cpu_limit = (randint(self.MIN_CPU_LIMIT, round(self.MAX_CPU_LIMIT / 3)) // self.INCREMENT) * self.INCREMENT # 3 - n_agents, random rounded numbers
        # patch_pod(f'localization-api{self.id}', cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m", container_name='localization-api', debug=True)

        self.state = self.get_current_usage()
        self.steps = 0
        
        # fill state with the last value
        self.state = [self.state[-1]] * self.STATE_LENTGH

        return self.state

    def normalize_cpu_usage(self, cpu_usage):
        return cpu_usage / self.MAX_CPU_LIMIT

    def get_current_usage(self):
        (cpu_limit, cpu, cpu_percentage), (memory_limit, memory, memory_percentage), (rx, tx), throttled = self.node.get_container_usage(self.container_id)
        self.last_cpu_percentage = cpu_percentage
        n_cpu_limit, n_cpu = self.normalize_cpu_usage(cpu_limit), self.normalize_cpu_usage(cpu)

        # self.ALLOCATED = int(cpu_limit) # used from outerscope for resource deviation
        available_normed = self.AVAILABLE / self.MAX_CPU_LIMIT
        # state = [n_cpu_limit, n_cpu, (cpu_percentage / 100), self.normalize_cpu_usage(self.ALLOCATED), available_normed]
        # state = [n_cpu_limit, n_cpu, (cpu_percentage / 100), self.normalize_cpu_usage(self.AVAILABLE)]
        state = [n_cpu_limit, n_cpu, available_normed, throttled, self.other_avg_util / 100]

        self.states_fifo.append(state)
        self.states_fifo.pop(0)
        if self.DEBUG:
            print(f'Agent {self.id}, LIMIT: {cpu_limit}, AVAILABLE: {self.AVAILABLE}, state(limit, usage, others): {state}') 
        return self.states_fifo

    def increase_resources(self):
        cpu_limit, memory_limit = self.node.get_container_limits(self.container_id)
        updated_cpu_limit = int(max(min(cpu_limit + self.INCREMENT, cpu_limit + self.AVAILABLE), self.MIN_CPU_LIMIT))
        self.ALLOCATED = updated_cpu_limit
        patch_pod(f'localization-api{self.id}', cpu_request=f"{updated_cpu_limit}m", cpu_limit=f"{updated_cpu_limit}m", container_name='localization-api', debug=True)

    def decrease_resources(self):
        cpu_limit, memory_limit = self.node.get_container_limits(self.container_id)
        updated_cpu_limit = int(max(cpu_limit - self.INCREMENT, self.MIN_CPU_LIMIT))
        self.ALLOCATED = updated_cpu_limit
        patch_pod(f'localization-api{self.id}', cpu_request=f"{updated_cpu_limit}m", cpu_limit=f"{updated_cpu_limit}m", container_name='localization-api', debug=True)

    def save_last_limit(self):
        (cpu_limit, cpu, cpu_percentage), (memory_limit, memory, memory_percentage), (rx, tx) = self.node.get_container_usage(self.container_id)
        self.last_cpu_limit = int(cpu_limit)
    
    def set_last_limit(self):
        # self.ALLOCATED = self.last_cpu_limit
        patch_pod(f'localization-api{self.id}', cpu_request=f"{self.ALLOCATED}m", cpu_limit=f"{self.ALLOCATED}m", container_name='localization-api', debug=True)
        # print(f"Set last limit to {self.last_cpu_limit} for agent {self.id} and pod localization-api{self.id}")

    def patch(self, limit):
        patch_pod(f'localization-api{self.id}', cpu_request=f"{limit}m", cpu_limit=f"{limit}m", container_name='localization-api', debug=True)
        self.ALLOCATED = limit

    def get_container_usage(self):
        return self.node.get_container_usage(self.container_id)
