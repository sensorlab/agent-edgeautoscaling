import re
import yaml
import time
import requests
import json
from kubernetes import client, config

from node import Node


# returns list(nodes) from the connected cluster
def init_nodes(debug=False, custom_label='type=ray'):
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
                    node_ip = None
                    if pod.status.host_ip:
                        node_ip = pod.status.host_ip
                    nodes.append(Node(pod.spec.node_name, pod.status.pod_ip, node_ip))

    print("Observable pods/nodes:")
    for node in nodes:
        node.update_containers(debug=debug, custom_label=custom_label)
        print(f"{node.name}:{node.ip}, ca: {node.ca_ip}, pods: {list(node.get_containers().values())}")
    print()

    return nodes

def make_request(url, data):
    headers = {'Content-Type': 'application/json'}
    start_time = time.time()
    response = requests.post(url, data=json.dumps(data), headers=headers, timeout=5) # 5 seconds timeout
    end_time = time.time()
    latency = end_time - start_time
    # print(f"Request latency: {latency} seconds")
    if response.status_code != 200:
        print(f"Error making prediction: {response.text}")
        return None
    # print(response.json())
    return latency

def increment_last_number(input_string):
    match = re.search(r'(\d+)$', input_string)

    if match:
        last_number = int(match.group(1))
        new_number = last_number + 1
        result_string = re.sub(r'\d+$', str(new_number), input_string)
        return result_string
    else:
        return input_string + '1'

def load_config():
    with open('application_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config
