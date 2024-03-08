import time

from utils import init_nodes, load_config

if __name__ == '__main__':
    config = load_config()
    nodes = init_nodes(debug=config.get('DEBUG'), custom_label=config.get('custom_app_label'))

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
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                (cpu_limit, cpu, cpu_percentage), (memory_limit, memory, memory_percentage), (rx, tx) = node.get_container_usage(container_id)
                print(f"Usage for container {pod_name} at pod {container_name}:{pod_ip}")
                print(f"CPU Usage : {cpu:.2f} mC, {cpu_percentage:.2f}%, limit {cpu_limit} mC")
                print(f"Memory Usage: {memory:.2f} MB, {memory_percentage:.2f}%, limit {memory_limit} MB")
                print(f"RX: {rx} MB, TX: {tx} MB")
                print('')

        time.sleep(1)
        print("\033c")
