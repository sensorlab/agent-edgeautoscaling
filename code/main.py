import time

from kubernetes import client, config

from node import Node
from pod_controller import patch_pod

DEBUG = True

if __name__ == '__main__':
    nodes = []
    if DEBUG:
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()

    ret = v1.list_pod_for_all_namespaces(label_selector='name=cadvisor')
    ret_arm64 = v1.list_pod_for_all_namespaces(label_selector='name=cadvisor-arm64')
    for pod in ret.items + ret_arm64.items:
        if pod.metadata.namespace == "kube-system":
            for status in pod.status.container_statuses:
                if status.name == "cadvisor" or status.name == "cadvisor-arm64":
                    nodes.append(Node(pod.spec.node_name, pod.status.pod_ip))

    for node in nodes:
        node.update_containers(debug=DEBUG)
        print(node.name, node.ca_ip, ", pods:", list(node.get_containers().values()))

    # simple vpa scaling, meant to monitor and scale while running an operation
    updated_container_id = ''
    while True:
        for node in nodes:
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                cpu, cpu_p, _, _ = node.get_container_usage(container_id, container_name)

                if cpu_p > 95.0 and updated_container_id == '':
                    patch_pod(pod_name, cpu_request="2", cpu_limit="2",
                              container_name=container_name)
                    updated_container_id = container_id
                elif updated_container_id == container_id and cpu_p < 10:
                    patch_pod(pod_name, cpu_request="500m", cpu_limit="500m",
                              container_name=container_name, debug=DEBUG)
                    updated_container_id = ''

        time.sleep(5)
