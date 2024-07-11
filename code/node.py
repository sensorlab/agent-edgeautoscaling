import requests

from datetime import datetime
from kubernetes import client, config


class Node:
    def __init__(self, name, ca_ip, ip):
        self.name = name
        self.ip = ip
        self.ca_ip = ca_ip
        self.containers = dict()
        self.last_pod_limits = dict()

    def update_containers(self, debug=False, custom_label='type=ray'):
        self.containers = dict()
        if debug:
            config.load_kube_config()
        else:
            config.load_incluster_config()
        v1 = client.CoreV1Api()

        try:
            ret = v1.list_pod_for_all_namespaces(
                label_selector=custom_label, field_selector=f'spec.nodeName={self.name}'
            )
            for pod in ret.items:
                if pod.status.phase == "Running":
                    for container_status in pod.status.container_statuses:
                        # make sure its the proper container
                        if container_status.name == custom_label.split("=")[-1]:
                            self.containers[container_status.container_id.split("//")[1]] = (pod.metadata.name, container_status.name, pod.status.pod_ip)
        
        except Exception as e:
            print(f"Error: {e}")

    def get_containers(self):
        return self.containers

    def get_container_usage(self, container_id):
        containers_stats_url = f"http://{self.ca_ip}:8080/api/v1.3/subcontainers/kubepods/"

        response = requests.get(containers_stats_url)
        if response.status_code == 200:
            containers_stats = response.json()
            container = next((c for c in containers_stats if container_id in c["name"]), None)
            if container:
                # tweak the comparable metrics to -3 for more accurate metric
                current_cpu_usage_nanoseconds = container["stats"][-1]["cpu"]["usage"]["total"]
                previous_cpu_usage_nanoseconds = container["stats"][-3]["cpu"]["usage"]["total"]

                current_timestamp_str = container["stats"][-1]["timestamp"].split('.')[0] + 'Z'
                previous_timestamp_str = container["stats"][-3]["timestamp"].split('.')[0] + 'Z'

                current_timestamp = datetime.strptime(current_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                previous_timestamp = datetime.strptime(previous_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")

                time_interval = current_timestamp - previous_timestamp
                time_interval_seconds = time_interval.total_seconds()

                # print(f"curr: {current_timestamp}, prev: {previous_timestamp}, difference: {time_interval_seconds}")

                cpu_usage_delta_nanoseconds = current_cpu_usage_nanoseconds - previous_cpu_usage_nanoseconds
                cpu_usage_per_second = cpu_usage_delta_nanoseconds / time_interval_seconds

                cpu_usage_millicores = cpu_usage_per_second / 1000000
                # cpu_limit_mc = container["spec"]["cpu"]["limit"]
                cpu_limit_mc = container["spec"]["cpu"]["quota"] / 100
                cpu_usage_percentage = (cpu_usage_per_second / (cpu_limit_mc * 1_000_000)) * 100

                current_memory_usage_bytes = container["stats"][-1]["memory"]["usage"]

                memory_usage_megabytes = current_memory_usage_bytes / (1024 * 1024)
                memory_limit_bytes = container["spec"]["memory"]["limit"]
                memory_usage_percentage = (current_memory_usage_bytes / memory_limit_bytes) * 100

                network_rx_per_second_mb, network_tx_per_second_mb = self.get_throughput(time_interval_seconds)

                throttled = container['stats'][-1]['cpu']['cfs']['throttled_time'] > container['stats'][-3]['cpu']['cfs']['throttled_time']

                return (cpu_limit_mc, cpu_usage_millicores, cpu_usage_percentage), (memory_limit_bytes / (1024 * 1024), memory_usage_megabytes, memory_usage_percentage), (network_rx_per_second_mb, network_tx_per_second_mb), throttled 
            else:
                print(f"Container {container_id} not found")
                return (0, 0, 0), (0, 0, 0), (0, 0), (0, 0)
        else:
            print("Failed to fetch containers stats")

    def get_container_limits(self, container_id):
        containers_stats_url = f"http://{self.ca_ip}:8080/api/v1.3/subcontainers/kubepods/"

        response = requests.get(containers_stats_url)
        if response.status_code == 200:
            containers_stats = response.json()
            container = next((c for c in containers_stats if container_id in c["name"]), None)
            if container:
                # cpu_limit_mc = container["spec"]["cpu"]["limit"]
                cpu_limit_mc = container["spec"]["cpu"]["quota"] / 100
                memory_limit_bytes = container["spec"]["memory"]["limit"]
                return cpu_limit_mc, memory_limit_bytes
            else:
                print(f"Container {container_id} not found")
                return 0, 0
        else:
            print("Failed to fetch containers stats")

    def get_container_usage_saving_data(self, container_id):
        containers_stats_url = f"http://{self.ca_ip}:8080/api/v1.3/subcontainers/kubepods/"

        response = requests.get(containers_stats_url)
        if response.status_code == 200:
            containers_stats = response.json()
            container = next((c for c in containers_stats if container_id in c["name"]), None)
            if container:
                current_cpu_usage_nanoseconds = container["stats"][-1]["cpu"]["usage"]["total"]
                previous_cpu_usage_nanoseconds = container["stats"][-2]["cpu"]["usage"]["total"]

                current_timestamp_str = container["stats"][-1]["timestamp"].split('.')[0] + 'Z'
                previous_timestamp_str = container["stats"][-2]["timestamp"].split('.')[0] + 'Z'

                current_timestamp = datetime.strptime(current_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                previous_timestamp = datetime.strptime(previous_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")

                time_interval = current_timestamp - previous_timestamp
                time_interval_seconds = time_interval.total_seconds()

                cpu_usage_delta_nanoseconds = current_cpu_usage_nanoseconds - previous_cpu_usage_nanoseconds
                cpu_usage_per_second = cpu_usage_delta_nanoseconds / time_interval_seconds

                cpu_usage_millicores = cpu_usage_per_second / 1000000
                cpu_limit_mc = container["spec"]["cpu"]["limit"]
                cpu_usage_percentage = (cpu_usage_per_second / (cpu_limit_mc * 1_000_000)) * 100

                current_memory_usage_bytes = container["stats"][-1]["memory"]["usage"]

                memory_usage_megabytes = current_memory_usage_bytes / (1024 * 1024)
                memory_limit_bytes = container["spec"]["memory"]["limit"]
                memory_usage_percentage = (current_memory_usage_bytes / memory_limit_bytes) * 100

                memory_usage_cache = container["stats"][-1]["memory"]["cache"]
                memory_usage_rss = container["stats"][-1]["memory"]["rss"]
                memory_usage_swap = container["stats"][-1]["memory"]["swap"]
                memory_usage_mapped_file = container["stats"][-1]["memory"]["mapped_file"]
                memory_usage_working_set = container["stats"][-1]["memory"]["working_set"]
                previous_memory_usage_cache = container["stats"][-2]["memory"]["cache"]
                previous_memory_usage_rss = container["stats"][-2]["memory"]["rss"]
                previous_memory_usage_swap = container["stats"][-2]["memory"]["swap"]
                previous_memory_usage_mapped_file = container["stats"][-2]["memory"]["mapped_file"]
                previous_memory_usage_working_set = container["stats"][-2]["memory"]["working_set"]
                memory_usage_cache_delta = memory_usage_cache - previous_memory_usage_cache
                memory_usage_rss_delta = memory_usage_rss - previous_memory_usage_rss
                memory_usage_swap_delta = memory_usage_swap - previous_memory_usage_swap
                memory_usage_mapped_file_delta = memory_usage_mapped_file - previous_memory_usage_mapped_file
                memory_usage_working_set_delta = memory_usage_working_set - previous_memory_usage_working_set

                mem_usage_cache = (memory_usage_cache_delta / time_interval_seconds) / (1024 * 1024)
                mem_usage_rss = (memory_usage_rss_delta / time_interval_seconds) / (1024 * 1024)
                mem_usage_swap = (memory_usage_swap_delta / time_interval_seconds) / (1024 * 1024)
                mem_usage_mapped_file = (memory_usage_mapped_file_delta / time_interval_seconds) / (1024 * 1024)
                mem_usage_working_set = (memory_usage_working_set_delta / time_interval_seconds) / (1024 * 1024)

                io_read_curr = container["stats"][-1]["diskio"]["io_service_bytes"][0]['stats']['Read']
                io_read_prev = container["stats"][-2]["diskio"]["io_service_bytes"][0]['stats']['Read']
                io_delta = io_read_curr - io_read_prev
                io_read_per_second = io_delta / time_interval_seconds

                io_write_curr = container["stats"][-1]["diskio"]["io_service_bytes"][0]['stats']['Write']
                io_write_prev = container["stats"][-2]["diskio"]["io_service_bytes"][0]['stats']['Write']
                io_delta = io_write_curr - io_write_prev
                io_write_per_second = io_delta / time_interval_seconds
                
                network_rx_per_second_mb, network_tx_per_second_mb = self.get_throughput(time_interval_seconds)

                return (cpu_limit_mc, cpu_usage_millicores, cpu_usage_percentage), (memory_limit_bytes, memory_usage_megabytes, memory_usage_percentage), (io_read_per_second, io_write_per_second), (network_rx_per_second_mb, network_tx_per_second_mb), (mem_usage_cache, mem_usage_rss, mem_usage_swap, mem_usage_mapped_file, mem_usage_working_set)
            else:
                print(f"Container {container_id} not found")
                return (0, 0, 0), (0, 0, 0), (0, 0), (0, 0)
        else:
            print("Failed to fetch containers stats")

    def get_throughput(self, time_interval):
        containers_url = f"http://{self.ca_ip}:8080/api/v1.3/containers"
        response = requests.get(containers_url)
        if response.status_code == 200:
            containers_stats = response.json()
            # todo: fixme
            current_network_rx_bytes = containers_stats["stats"][-1]["network"]["interfaces"][-1]["rx_bytes"]
            previous_network_rx_bytes = containers_stats["stats"][-2]["network"]["interfaces"][-1]["rx_bytes"]
            network_rx_delta = current_network_rx_bytes - previous_network_rx_bytes
            network_rx_per_second = network_rx_delta / time_interval

            current_network_tx_bytes = containers_stats["stats"][-1]["network"]["interfaces"][-1]["tx_bytes"]
            previous_network_tx_bytes = containers_stats["stats"][-2]["network"]["interfaces"][-1]["tx_bytes"]
            network_tx_delta = current_network_tx_bytes - previous_network_tx_bytes
            network_tx_per_second = network_tx_delta / time_interval

            # current_network_rx_packets = containers_stats["stats"][-1]["network"]["interfaces"][-1]["rx_packets"]
            # previous_network_rx_packets = containers_stats["stats"][-2]["network"]["interfaces"][-1]["rx_packets"]
            # network_rx_packets_delta = current_network_rx_packets - previous_network_rx_packets
            # packets_per_second = network_rx_packets_delta / time_interval
            # print(f"Packets per second: {packets_per_second}")

            return (network_rx_per_second / (1024 * 1024)), (network_tx_per_second / (1024 * 1024))

    def get_usage(self):
        summary_url = f"http://{self.ca_ip}:8080/api/v2.0/summary"
        total_cpu_capacity_millicores, total_memory_capacity = self.get_node_capacity()

        response = requests.get(summary_url)
        if response.status_code == 200:
            summary_data = response.json()

            current_cpu_usage = summary_data.get("/", {}).get("latest_usage", {}).get("cpu")
            memory_usage = summary_data.get("/", {}).get("latest_usage", {}).get("memory")

            cpu_usage_percentage = (current_cpu_usage / total_cpu_capacity_millicores) * 100
            memory_usage_percent = (memory_usage / total_memory_capacity) * 100

            memory_usage = memory_usage / (1024 * 1024)

            free_cpu = total_cpu_capacity_millicores - current_cpu_usage
            free_mem = total_memory_capacity - memory_usage
            free_mem = free_mem / (1024 * 1024)

            return current_cpu_usage, cpu_usage_percentage, memory_usage, memory_usage_percent, free_cpu, free_mem
        else:
            print("Failed to fetch resource usage.")

    def get_means(self):
        summary_url = f"http://{self.ca_ip}:8080/api/v2.0/summary"

        response = requests.get(summary_url)
        if response.status_code == 200:
            summary_data = response.json()

            minute_cpu = summary_data.get("/", {}).get("minute_usage", {}).get("cpu")['mean']
            hour_cpu = summary_data.get("/", {}).get("hour_usage", {}).get("cpu")['mean']
            day_cpu = summary_data.get("/", {}).get("day_usage", {}).get("cpu")['mean']
            minute_memory = summary_data.get("/", {}).get("minute_usage", {}).get("memory")['mean']
            hour_memory = summary_data.get("/", {}).get("hour_usage", {}).get("memory")['mean']
            day_memory = summary_data.get("/", {}).get("day_usage", {}).get("memory")['mean']

            return (minute_cpu, hour_cpu, day_cpu), (minute_memory, hour_memory, day_memory)
        else:
            print("Failed to fetch resource usage.")

    def get_node_capacity(self):
        limits_url = f"http://{self.ca_ip}:8080/api/v2.0/machine"
        response = requests.get(limits_url)
        if response.status_code == 200:
            machine_data = response.json()
            total_cpu_capacity_millicores = machine_data.get("num_cores") * 1000
            total_memory_capacity = machine_data.get("memory_capacity")  # in bytes
            return total_cpu_capacity_millicores, total_memory_capacity
        else:
            print("Failed to fetch machine information.")

    def get_allocated_resources(self):
        allocated_cpu = 0
        allocated_memory = 0
        for container_id, _ in list(self.get_containers().items()):
            containers_url = f"http://{self.ca_ip}:8080/api/v1.3/subcontainers/kubepods/"
            response = requests.get(containers_url)
            if response.status_code == 200:
                containers_stats = response.json()
                container = next((c for c in containers_stats if container_id in c["name"]), None)
                if container:
                    allocated_cpu += container['spec']['cpu']['limit']
                    allocated_memory += container['spec']['memory']['limit']
        return allocated_cpu, allocated_memory

    # wip: calculates how much the pods from the application have allocated
    def get_unallocated_capacity(self):
        total_cpu_capacity, total_memory_capacity = self.get_node_capacity()
        allocated_cpu, allocated_memory = self.get_allocated_resources()

        unallocated_cpu = max(0, total_cpu_capacity - allocated_cpu)
        unallocated_memory = max(0, (total_memory_capacity - allocated_memory) / (1024 * 1024))

        return unallocated_cpu, unallocated_memory

    def get_root_storage(self):
        filesystem_url = f"http://{self.ca_ip}:8080/api/v2.0/storage"
        response = requests.get(filesystem_url)
        if response.status_code == 200:
            filesystem_stats = response.json()
            filesystem = next((fs for fs in filesystem_stats if fs['mountpoint'] == '/'), None)
            if filesystem:
                used = filesystem['usage']
                limit = filesystem['capacity']
                available = filesystem['available']
                percentage = (used / limit) * 100
                return ((used / (1024 * 1024 * 1024)), (limit / (1024 * 1024 * 1024)), (available / (1024 * 1024 * 1024)), percentage)
            else:
                print("Failed to fetch filesystem usage.")
        else:
            print("Failed to fetch filesystem usage.")
