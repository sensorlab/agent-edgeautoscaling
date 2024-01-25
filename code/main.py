import time

from pod_controller import create_pod_from, patch_pod
from utils import init_nodes

DEBUG = True

if __name__ == '__main__':
    nodes = init_nodes(DEBUG, custom_label='app=nvg-api')

    scale_cpu, max_cpu, min_cpu = 1000, 3750, 500 # 500m, 3750m, 500m (millicores)
    UPPER, LOWER = 95.0, 10.0 # thresholds for vpa scaling

    for node in nodes:
        for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
            patch_pod(pod_name, cpu_request=f"{min_cpu}m", cpu_limit=f"{min_cpu}m",
                        container_name=container_name, debug=DEBUG)

    updated_container_ids = []
    created_pod = False
    while True:
        for node in nodes:
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                (cpu_limit, cpu, cpu_p), (_, _, _) = node.get_container_usage(container_id)
                
                if cpu_p > UPPER:
                    cpu_limit = min(max_cpu, int(cpu_limit) + scale_cpu)
                    patch_pod(pod_name, cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m",
                              container_name=container_name, debug=DEBUG)
                    # primitive, todo
                    if cpu_limit == max_cpu and not created_pod:
                        create_pod_from(pod_name, 'raspberrypi1', debug=DEBUG)
                        created_pod = True
                    if container_id not in updated_container_ids:
                        updated_container_ids.append(container_id)
                elif container_id in updated_container_ids and cpu_p < LOWER:
                    cpu_limit = max(min_cpu, int(cpu_limit) - scale_cpu)
                    patch_pod(pod_name, cpu_request=f"{cpu_limit}m", cpu_limit=f"{cpu_limit}m",
                              container_name=container_name, debug=DEBUG)
                    if cpu_limit == min_cpu:
                        updated_container_ids.remove(container_id)

        time.sleep(5)
