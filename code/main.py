import time

from pod_controller import patch_pod
from utils import init_nodes

DEBUG = True

if __name__ == '__main__':
    nodes = init_nodes(DEBUG)

    # simple vpa scaling, meant to monitor and scale while running an operation
    updated_container_id = ''
    while True:
        for node in nodes:
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                cpu, cpu_p, _, _ = node.get_container_usage(container_id)

                if cpu_p > 95.0 and updated_container_id == '':
                    patch_pod(pod_name, cpu_request="2", cpu_limit="2",
                              container_name=container_name)
                    updated_container_id = container_id
                elif updated_container_id == container_id and cpu_p < 10:
                    patch_pod(pod_name, cpu_request="500m", cpu_limit="500m",
                              container_name=container_name, debug=DEBUG)
                    updated_container_id = ''

        time.sleep(5)
