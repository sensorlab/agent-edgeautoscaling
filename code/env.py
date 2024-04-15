from gymnasium import Env
from gymnasium import spaces

from utils import init_nodes
from pod_controller import get_loadbalancer_external_port, patch_pod

class ElastisityEnv(Env):
    def __init__(self, id):
        super().__init__()
        self.MAX_CPU_LIMIT = 1000
        self.MIN_CPU_LIMIT = 50
        self.INCREMENT = 50
        # self.START_CPU = 100

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

        self.STATE_LENTGH = 6
        # self.states_fifo = [[0, 0, 0, 0, 0] for _ in range(self.STATE_LENTGH)]
        self.states_fifo = [[0, 0, 0] for _ in range(self.STATE_LENTGH)]
        self.state = self.get_current_usage()

        self.steps = 0
        self.MAX_STEPS = 50

        # 30% - 60%
        self.UPPER_CPU = 60
        self.LOWER_CPU = 30

        self.loadbalancer_port = get_loadbalancer_external_port(service_name='ingress-nginx-controller')

    def step(self, action):
        if action == 0:
            self.decrease_resources()
        elif action == 2:
            self.increase_resources()

        self.state = self.get_current_usage()

        # latency = self.calculate_latency(1)
        # reward = 1 - latency
        # reward = 1 - latency * 10

        if self.last_cpu_percentage < self.LOWER_CPU:
            # usage_penalty = 1.3 - self.last_cpu_percentage / 100
            usage_penalty = 0.75 - self.last_cpu_percentage / 100 # lower penalty on this
        elif self.last_cpu_percentage > self.UPPER_CPU:
            usage_penalty = self.last_cpu_percentage / 100
        else:
            usage_penalty = 0

        reward = - usage_penalty

        # print(f"Steps {self.steps} Reward: {reward}, State {self.state}")
        # print(f"agent: {self.id}, action: {action}")

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            # print(f"Max steps reached")
            done = True
        else:
            done = False

        return self.state, reward, done, 0
    
    def reset(self):
        # cpu_limit = (randint(self.MIN_CPU_LIMIT, round(self.MAX_CPU_LIMIT / 3)) // self.INCREMENT) * self.INCREMENT # 3 - n_agents, random rounded numbers
        # patch_pod(f'localization-api{self.id}', cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m", container_name='localization-api', debug=True)

        self.state = self.get_current_usage()
        self.steps = 0
        
        # fill the state with the last value
        value = self.state[-1]
        self.state = [value for _ in range(self.STATE_LENTGH)]

        return self.state

    def normalize_cpu_usage(self, cpu_usage):
        normalized_cpu_usage = (cpu_usage - self.MIN_CPU_LIMIT) / (self.MAX_CPU_LIMIT - self.MIN_CPU_LIMIT)
        return normalized_cpu_usage

    def get_current_usage(self):
        (cpu_limit, cpu, cpu_percentage), (memory_limit, memory, memory_percentage), (rx, tx) = self.node.get_container_usage(self.container_id)
        self.last_cpu_percentage = cpu_percentage
        n_cpu_limit, n_cpu = self.normalize_cpu_usage(cpu_limit), self.normalize_cpu_usage(cpu)

        self.ALLOCATED = cpu_limit
        available_normed = self.AVAILABLE / self.MAX_CPU_LIMIT
        # state = [n_cpu_limit, n_cpu, (cpu_percentage / 100), self.normalize_cpu_usage(self.ALLOCATED), available_normed]
        # state = [n_cpu_limit, n_cpu, (cpu_percentage / 100), self.normalize_cpu_usage(self.AVAILABLE)]
        state = [n_cpu_limit, n_cpu, available_normed]

        self.states_fifo.append(state)
        self.states_fifo.pop(0)
        # print(f'Agent {self.id}: ALLOCATED: {self.ALLOCATED}, AVAILABLE: {self.AVAILABLE}, cpu_limit: {cpu_limit}')
        # print(f'Agent {self.id}: STATE: {state}')
        return self.states_fifo

    def increase_resources(self):
        cpu_limit, memory_limit = self.node.get_container_limits(self.container_id)
        updated_cpu_limit = int(max(min(cpu_limit + self.INCREMENT, cpu_limit + self.AVAILABLE), self.MIN_CPU_LIMIT))
        patch_pod(f'localization-api{self.id}', cpu_request=f"{updated_cpu_limit}m", cpu_limit=f"{updated_cpu_limit}m", container_name='localization-api', debug=True)

    def decrease_resources(self):
        cpu_limit, memory_limit = self.node.get_container_limits(self.container_id)
        updated_cpu_limit = int(max(cpu_limit - self.INCREMENT, self.MIN_CPU_LIMIT))

        patch_pod(f'localization-api{self.id}', cpu_request=f"{updated_cpu_limit}m", cpu_limit=f"{updated_cpu_limit}m", container_name='localization-api', debug=True)

    '''
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
    '''

    def save_last_limit(self):
        (cpu_limit, cpu, cpu_percentage), (memory_limit, memory, memory_percentage), (rx, tx) = self.node.get_container_usage(self.container_id)
        self.last_cpu_limit = int(cpu_limit)
    
    def set_last_limit(self):
        patch_pod(f'localization-api{self.id}', cpu_request=f"{self.last_cpu_limit}m", cpu_limit=f"{self.last_cpu_limit}m", container_name='localization-api', debug=True)
