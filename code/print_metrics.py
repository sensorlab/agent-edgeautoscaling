import time

from kubernetes import client, config

from node import Node

if __name__ == '__main__':
    nodes = []
    config.load_kube_config()
    v1 = client.CoreV1Api()

    ret = v1.list_pod_for_all_namespaces(label_selector='name=cadvisor')
    ret_arm64 = v1.list_pod_for_all_namespaces(label_selector='name=cadvisor-arm64')
    for pod in ret.items + ret_arm64.items:
        if pod.metadata.namespace == "kube-system":
            for status in pod.status.container_statuses:
                if status.name == "cadvisor" or status.name == "cadvisor-arm64":
                    nodes.append(Node(pod.spec.node_name, pod.status.pod_ip))

    for node in nodes:
        node.update_containers(debug=True)
        print(node.name, node.ca_ip)

    while True:
        for node in nodes:
            print(f"Current usage for node {node.name}, ca_advisor at {node.ca_ip}")

            node_cpu, node_cpu_percentage, node_memory, node_memory_percentage, free_cpu, free_mem = node.get_usage()
            print(f"CPU Usage : {node_cpu:.2f} mC, {node_cpu_percentage:.2f}%")
            print(f"Memory Usage: {node_memory:.2f} MB, {node_memory_percentage:.2f}%")
            print(f"Free: {free_cpu:.2f} mC, {free_mem:.2f} MB")

            unallocated_cpu, unallocated_memory = node.get_unallocated_capacity()
            print(f"Unallocated: {unallocated_cpu} mc, {unallocated_memory} ")

            print("Containers-------")
            for container_id, (container_status, container_name, pod_ip) in list(node.get_containers().items()):
                container_cpu, container_cpu_percentage, container_memory, container_memory_percentage = node.get_container_usage(
                    container_id, container_name)
                print(f"Usage for container {container_name} at pod {container_name}:{pod_ip}")
                print(f"CPU Usage : {container_cpu:.2f} mC, {container_cpu_percentage:.2f}%")
                print(f"Memory Usage: {container_memory:.2f} MB, {container_memory_percentage:.2f}%")
                print('')

        time.sleep(1)
        print("\033c")
