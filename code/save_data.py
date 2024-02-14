import csv
import datetime
import time

from utils import init_nodes, load_config

DEBUG = True
timeseries_data = []


def write_to_csv(filename):
    with open(filename, mode='w', newline='') as file:
        fieldnames = [
            "timestamp",
            "node_name",
            "node_cpu_usage_mc",
            "node_cpu_usage_percentage",
            "node_memory_usage_mb",
            "node_memory_usage_percentage",
            "node_free_cpu_mc",
            "node_free_memory_mb",
            "node_unallocated_cpu_mc",
            "node_unallocated_memory_mb",
            "node_max_cpu_mc",
            "node_max_memory_mb",
            "node_cpu_minute_mean_mc",
            "node_memory_minute_mean_mb",
            "node_cpu_hour_mean_mc",
            "node_memory_hour_mean_mb",
            "node_cpu_day_mean_mc",
            "node_memory_day_mean_mb",
            "node_root_storage_usage_gb",
            "node_root_storage_limit_gb",
            "node_root_storage_available_gb",
            "node_root_storage_percentage_usage_gb",
            "pod_name",
            "pod_cpu_usage_mc",
            "pod_cpu_limit_mc",
            "pod_cpu_usage_percentage",
            "pod_mem_usage_cache_b",
            "pod_mem_usage_rss_b",
            "pod_mem_usage_swap_b",
            "pod_mem_usage_mapped_file_b",
            "pod_memory_limit_mb",
            "pod_memory_usage_mb",
            "pod_memory_usage_percentage",
            "pod_io_read_usage",
            "pod_io_write_usage",
            "pod_network_rx_mb",
            "pod_network_tx_mb"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for data_point in timeseries_data:
            writer.writerow(data_point)


if __name__ == '__main__':
    config = load_config()
    nodes = init_nodes(debug=config.get('DEBUG'), custom_label=config.get('custom_app_label'))

    duration_seconds = 300
    sampling_interval = 5
    file_name = 'timeseries.csv'
    end_time = time.time() + duration_seconds

    while time.time() < end_time:
        start_time = time.time()
        for node in nodes:
            node_cpu, node_cpu_percentage, node_memory, node_memory_percentage, free_cpu, free_mem = node.get_usage()
            unallocated_cpu, unallocated_memory = node.get_unallocated_capacity()
            max_cpu, max_memory = node.get_node_capacity()
            (minute_cpu, hour_cpu, day_cpu), (minute_mem, hour_mem, day_mem) = node.get_means()
            (root_usage, root_limit, root_available, root_percentage) = node.get_root_storage()

            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                (cpu_limit, cpu, cpu_p), (mem_limit, mem, mem_p), (io_read, io_write), (rx, tx), (mem_usage_cache_sec, mem_usage_rss_sec, mem_usage_swap_sec, mem_usage_mapped_file_sec, mem_usage_working_set_sec) = node.get_container_usage_saving_data(container_id)

                timeseries_data.append({
                    "timestamp": datetime.datetime.now(),
                    "node_name": node.name,
                    "node_cpu_usage_mc": node_cpu,
                    "node_cpu_usage_percentage": node_cpu_percentage,
                    "node_memory_usage_mb": node_memory,
                    "node_memory_usage_percentage": node_memory_percentage,
                    "node_free_cpu_mc": free_cpu,
                    "node_free_memory_mb": free_mem,
                    "node_unallocated_cpu_mc": unallocated_cpu,
                    "node_unallocated_memory_mb": unallocated_memory,
                    "node_max_cpu_mc": max_cpu,
                    "node_max_memory_mb": max_memory / (1024 * 1024),
                    "node_cpu_minute_mean_mc": minute_cpu,
                    "node_memory_minute_mean_mb": minute_mem / (1024 * 1024),
                    "node_cpu_hour_mean_mc": hour_cpu,
                    "node_memory_hour_mean_mb": hour_mem / (1024 * 1024),
                    "node_cpu_day_mean_mc": day_cpu,
                    "node_memory_day_mean_mb": day_mem / (1024 * 1024),
                    "node_root_storage_usage_gb": root_usage,
                    "node_root_storage_limit_gb": root_limit,
                    "node_root_storage_available_gb": root_available,
                    "node_root_storage_percentage_usage_gb": root_percentage,
                    "pod_name": pod_name,
                    "pod_cpu_usage_mc": cpu,
                    "pod_cpu_limit_mc": cpu_limit,
                    "pod_cpu_usage_percentage": cpu_p,
                    "pod_memory_limit_mb": mem_limit / (1024 * 1024),
                    "pod_memory_usage_mb": mem,
                    "pod_memory_usage_percentage": mem_p,
                    "pod_mem_usage_cache_b": mem_usage_cache_sec,
                    "pod_mem_usage_rss_b": mem_usage_rss_sec,
                    "pod_mem_usage_swap_b": mem_usage_swap_sec,
                    "pod_mem_usage_mapped_file_b": mem_usage_mapped_file_sec,
                    "pod_io_read_usage": io_read,
                    "pod_io_write_usage": io_write,
                    "pod_network_rx_mb": rx,
                    "pod_network_tx_mb": tx,
                })
        elapsed_time = time.time() - start_time
        if elapsed_time < sampling_interval:
            time.sleep(sampling_interval - elapsed_time)

    write_to_csv(f"data/{file_name}")
    print(f"Saved data to file: {file_name}, interval: {sampling_interval}s for {duration_seconds}s")
