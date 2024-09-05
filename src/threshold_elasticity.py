import time

from pod_controller import patch_pod
from utils import init_nodes, load_config


if __name__ == '__main__':
    config = load_config()
    debug_deployment = config['debug_deployment']
    target_app_label = config['target_app_label']
    scale_cpu = config['discrete_increment']
    max_cpu = config['max_cpu']
    min_cpu = config['min_cpu']
    UPPER = config['upper_cpu']
    LOWER = config['lower_cpu']
    action_interval = config['action_interval']
    
    resources = 1000
    AVAILABLE = resources
    print_output = True

    print(f"DEBUG: {debug_deployment}, custom_app_label: {target_app_label}, scale_cpu: {scale_cpu}, max_cpu: {max_cpu}, CPU limits: upper: {UPPER}, lower: {LOWER}, action_interval: {action_interval}")

    nodes = init_nodes(debug_deployment, custom_label=target_app_label)

    for node in nodes:
        for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
            patch_pod(pod_name, cpu_request=f"{min_cpu}m", cpu_limit=f"{min_cpu}m",
                        container_name=container_name, debug=debug_deployment, print_output=print_output)

    updated_container_ids = []
    while True:
        start_time = time.time()
        AVAILABLE = resources

        for node in nodes:
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                (cpu_limit, cpu, cpu_p), (_, _, _), (_, _), _ = node.get_container_usage(container_id)
                AVAILABLE -= cpu_limit

        for node in nodes:
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                (cpu_limit, cpu, cpu_p), (_, _, _), (_, _), _ = node.get_container_usage(container_id)
                # AVAILABLE -= cpu_limit

                if cpu_p > UPPER and AVAILABLE >= scale_cpu:
                    cpu_limit = min(max_cpu, int(cpu_limit) + scale_cpu)
                    patch_pod(pod_name, cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m",
                              container_name=container_name, debug=debug_deployment, print_output=print_output)
                    AVAILABLE -= scale_cpu
                    if container_id not in updated_container_ids:
                        updated_container_ids.append(container_id)
                elif container_id in updated_container_ids and cpu_p < LOWER:
                    cpu_limit = max(min_cpu, int(cpu_limit) - scale_cpu)
                    patch_pod(pod_name, cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m",
                              container_name=container_name, debug=debug_deployment, print_output=print_output)
                    AVAILABLE += scale_cpu
                    if cpu_limit == min_cpu:
                        updated_container_ids.remove(container_id)
        elapsed_time = time.time() - start_time
        if elapsed_time < action_interval:
            time.sleep(action_interval - elapsed_time)
