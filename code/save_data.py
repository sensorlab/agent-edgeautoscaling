import csv
import datetime
import time

from utils import init_nodes

DEBUG = True
timeseries_data = []


def write_to_csv(filename):
    with open(filename, mode='w', newline='') as file:
        fieldnames = ["pod_name", "timestamp", "cpu_millicores", "memory_mb", "max_cpu", "max_memory_mb"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for data_point in timeseries_data:
            writer.writerow(data_point)


if __name__ == '__main__':
    nodes = init_nodes(DEBUG)

    duration_seconds = 3
    interval_seconds = 1
    file_name = 'timeseries.csv'
    end_time = time.time() + duration_seconds

    while time.time() < end_time:
        for node in nodes:
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                cpu, cpu_p, memory_mb, memory_p = node.get_container_usage(container_id)
                max_cpu, max_memory = node.get_node_capacity()  # todo, no need to get it every time

                timeseries_data.append({
                    "pod_name": pod_name,
                    "timestamp": datetime.datetime.now(),
                    "cpu_millicores": cpu,
                    "memory_mb": memory_mb,
                    "max_cpu": max_cpu,
                    "max_memory_mb": max_memory / (1024 * 1024),
                })
        time.sleep(interval_seconds)

    write_to_csv(f"data/{file_name}")
    print(f"Saved data to file: {file_name}, interval: {interval_seconds}s for {duration_seconds}s")
