import random
import time
import numpy as np
from gymnasium import Env
from gymnasium import spaces

from utils import make_request, init_nodes
from pod_controller import get_loadbalancer_external_port, patch_pod

class CentralizedElastisityEnv(Env):
    def __init__(self, num_agents):
        super().__init__()
        self.MAX_CPU_LIMIT = 1000
        self.MIN_CPU_LIMIT = 50
        self.INCREMENT = 50

        self.action_space = spaces.Tuple([spaces.Discrete(3) for _ in range(num_agents)])

        nodes = init_nodes(debug=True, custom_label='app=localization-api')

        self.ids = [i for i in range(num_agents)]
        self.container_ids = []
        self.nodes = []
        for id in self.ids:
            for node in nodes:
                for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                    if pod_name == f'localization-api{id + 1}':
                        # select node and container_id for agent
                        self.container_ids.append(container_id)
                        self.nodes.append(node)
                        break
        # print(self.nodes, self.container_ids)

        self.last_cpu_percentages = [0 for _ in range(num_agents)]
        # self.which_node = 1
        # self.node = next((node for node in nodes if node.name == 'raspberrypi' + str(self.which_node)), None
        self.SECTORS = 8
        self.STATE_LENTGH = self.SECTORS * num_agents
        # init state with random values
        self.states_fifo = [[random.random(), random.random(), random.random()] for _ in range(self.STATE_LENTGH)]
        self.state = self.get_current_usage()

        self.steps = 0
        self.MAX_STEPS = 50

        # 30% - 60%
        self.UPPER_CPU = 60
        self.LOWER_CPU = 30

        self.loadbalancer_port = get_loadbalancer_external_port(service_name='ingress-nginx-controller')

    def step(self, actions):
        assert len(actions) == len(self.ids), "Number of actions must be equal to number of agents"

        # rewards = []
        for id in self.ids:
            action = actions[id]
            if action == 0:
                self.decrease_resources(id)
            elif action == 2:
                self.increase_resources(id)

            # reward = self.calculate_reward(id)
            # rewards.append(reward)

        self.state = self.get_current_usage()

        latency = self.calculate_latency(20)
        # reward = 1 - latency
        reward = 1 - latency * 10

        # if percentages not in range
        mean_percentage = np.array(self.last_cpu_percentages).mean()
        if mean_percentage < self.LOWER_CPU:
            usage_penalty = 2 - mean_percentage / 100
        elif mean_percentage > self.UPPER_CPU:
            usage_penalty = mean_percentage / 100
        else:
            usage_penalty = 0

        reward = reward - usage_penalty

        # print(f"Steps {self.steps} Reward: {reward}, State {self.state}")

        self.steps += 1
        done = self.steps >= self.MAX_STEPS
        return self.state, reward, done, latency
    
    def reset(self):
        self.state = self.get_current_usage()
        self.steps = 0
        # patch_pod('localization-api1', cpu_request=f"{self.START_CPU}m", cpu_limit=f"{self.START_CPU}m", container_name='localization-api', debug=True)
        return self.state

    def normalize_cpu_usage(self, cpu_usage):
        # normalized_cpu_usage = (cpu_usage - 0) / (self.MAX_CPU_LIMIT - 0)
        normalized_cpu_usage = (cpu_usage - self.MIN_CPU_LIMIT) / (self.MAX_CPU_LIMIT - self.MIN_CPU_LIMIT)
        return normalized_cpu_usage

    def get_current_usage(self):
        # for node in self.nodes:
        # for container_id, (pod_name, container_name, pod_ip) in list(self.node.get_containers().items()):
        for id in self.ids:
            (cpu_limit, cpu, cpu_percentage), (memory_limit, memory, memory_percentage), (rx, tx) = self.nodes[id].get_container_usage(self.container_ids[id])
            self.last_cpu_percentages[id] = cpu_percentage
                # states = ([cpu_limit, cpu, memory_limit, memory, rx, tx])
            n_cpu_limit, n_cpu = self.normalize_cpu_usage(cpu_limit), self.normalize_cpu_usage(cpu)
                
            state = [n_cpu_limit, n_cpu, (cpu_percentage / 100)]
            
            # sectors of 8 for each agent
            start = id * self.SECTORS
            end = start + self.SECTORS
            self.states_fifo.pop(start)
            self.states_fifo.insert(end, state)
        return self.states_fifo

    def increase_resources(self, agent):
        cpu_limit, memory_limit = self.nodes[agent].get_container_limits(self.container_ids[agent])
        cpu_limit = min(cpu_limit + self.INCREMENT, self.MAX_CPU_LIMIT)  
        
        patch_pod(f'localization-api{agent + 1}', cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m", container_name='localization-api', debug=True)

    def decrease_resources(self, agent):
        cpu_limit, memory_limit = self.nodes[agent].get_container_limits(self.container_ids[agent])
        cpu_limit = max(cpu_limit - self.INCREMENT, self.MIN_CPU_LIMIT)

        patch_pod(f'localization-api{agent + 1}', cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m", container_name='localization-api', debug=True)

    def calculate_latency(self, num_requests):
        url = f"http://localhost:{self.loadbalancer_port}/predict"
        latencies = []
        for _ in range(num_requests):
            data = {
                "feature": random.randint(0, 130)
            }
            latency = make_request(url, data)
            if latency:
                latencies.append(latency)
        a = np.array(latencies)
        # return a.prod()**(1.0/len(a))
        return a.mean() if len(a) > 0 else 0
