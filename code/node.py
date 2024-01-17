from datetime import datetime

import requests
from kubernetes import client, config


class Node:
    def __init__(self, name, ca_ip):
        self.name = name
        self.ca_ip = ca_ip
        self.containers = dict()

    def update_containers(self, debug=False):
        self.containers = dict()
        if debug:
            config.load_kube_config()
        else:
            config.load_incluster_config()
        v1 = client.CoreV1Api()

        try:
            ret = v1.list_pod_for_all_namespaces(
                label_selector='type=ray', field_selector=f'spec.nodeName={self.name}'
            )
            for pod in ret.items:
                if pod.status.phase == "Running":
                    for container_status in pod.status.container_statuses:
                        self.containers[container_status.container_id.split("//")[1]] = (pod.metadata.name,
                                                                                         container_status.name,
                                                                                         pod.status.pod_ip)

        except Exception as e:
            print(f"Error: {e}")

    def get_containers(self):
        return self.containers

    def get_container_usage(self, container_id):
        containers_stats_url = f"http://{self.ca_ip}:8080/api/v1.3/subcontainers/kubepods/burstable/"

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
                total_cpu_capacity_nanoseconds = container["spec"]["cpu"]["limit"]
                cpu_usage_percentage = (cpu_usage_per_second / (total_cpu_capacity_nanoseconds * 1_000_000)) * 100

                current_memory_usage_bytes = container["stats"][-1]["memory"]["usage"]

                memory_usage_megabytes = current_memory_usage_bytes / (1024 * 1024)
                total_memory_capacity_bytes = container["spec"]["memory"]["limit"]
                memory_usage_percentage = (current_memory_usage_bytes / total_memory_capacity_bytes) * 100

                return cpu_usage_millicores, cpu_usage_percentage, memory_usage_megabytes, memory_usage_percentage
            else:
                print(f"Container {container_id} not found")
                return 0, 0, 0, 0
        else:
            print("Failed to fetch containers stats")

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
            containers_stats_url = f"http://{self.ca_ip}:8080/api/v1.3/subcontainers/kubepods/burstable/"
            response = requests.get(containers_stats_url)
            if response.status_code == 200:
                containers_stats = response.json()
                container = next((c for c in containers_stats if container_id in c["name"]), None)
                if container:
                    allocated_cpu += container['spec']['cpu']['limit']
                    allocated_memory += container['spec']['memory']['limit']
        return allocated_cpu, allocated_memory

    # wip: doesn't calculate accurately (just calculates how much the ray pods have allocated)
    def get_unallocated_capacity(self):
        total_cpu_capacity, total_memory_capacity = self.get_node_capacity()
        allocated_cpu, allocated_memory = self.get_allocated_resources()

        unallocated_cpu = max(0, total_cpu_capacity - allocated_cpu)
        unallocated_memory = max(0, (total_memory_capacity - allocated_memory) / (1024 * 1024))

        return unallocated_cpu, unallocated_memory
