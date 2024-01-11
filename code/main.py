import time

from kubernetes import client, config

from node import Node
from pod_controller import patch_pod

if __name__ == '__main__':
    nodes = []
    config.load_kube_config()
    v1 = client.CoreV1Api()
    namespace = "kube-system"

    ret = v1.list_pod_for_all_namespaces(label_selector='name=cadvisor')
    ret_arm64 = v1.list_pod_for_all_namespaces(label_selector='name=cadvisor-arm64')
    for pod in ret.items + ret_arm64.items:
        if pod.metadata.namespace == namespace:
            for status in pod.status.container_statuses:
                if status.name == "cadvisor" or status.name == "cadvisor-arm64":
                    nodes.append(Node(pod.spec.node_name, pod.status.pod_ip))

    for node in nodes:
        node.update_containers()
        print(node.name, node.ca_ip)

    # simple vpa scaling, meant to monitor and scale while running an operation
    updated = False
    while True:
        for node in nodes:
            cpu, cpu_p, _, _ = node.get_containers_usage()

            if cpu_p > 95.0 and not updated:
                patch_pod(node.get_containers()[0][0], cpu_request="2", cpu_limit="2",
                          container_name=node.get_containers()[0][1])
                updated = True
            elif updated and cpu_p < 10:
                patch_pod(node.get_containers()[0][0], cpu_request="500m", cpu_limit="500m",
                          container_name=node.get_containers()[0][1])
                updated = False

        time.sleep(3)
