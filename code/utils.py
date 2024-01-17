from kubernetes import client, config

from node import Node


# returns list(nodes) from the connected cluster
def init_nodes(debug=False):
    nodes = []
    if debug:
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

    print("Observable pods/nodes:")
    for node in nodes:
        node.update_containers(debug=debug)
        print(f"{node.name}, ca: {node.ca_ip}, pods: {list(node.get_containers().values())}")
    print()

    return nodes
