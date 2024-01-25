import pickle
import numpy as np
import time 
import requests
import sys

from kubernetes import config, client
from concurrent.futures import ThreadPoolExecutor

from utils import init_nodes


def delete_pod(pod_name, debug=True):
    if debug:
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
    namespace = 'default'
    try:
        v1.delete_namespaced_pod(pod_name, namespace=namespace)
        print(f"Deleted pod {pod_name}")
    except Exception as e:
        print(f"Error deleting pod {pod_name}: {e}")

def get_loadbalancer_external_port(service_name, namespace='default'):
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        service = v1.read_namespaced_service(service_name, namespace)
        external_port = service.spec.ports[0].node_port
        return external_port
    except Exception as e:
        print(f"Error retrieving load balancer external port: {e}")
        return None

def make_request(url, data):
    response = requests.post(url, json=data)
    rtt = response.elapsed.total_seconds()

    if response.status_code != 200:
        print(f'Error: {response.status_code}')

    if not response.json()['pred']:
        print('Error: Response is empty')

    return rtt, response.json()["pred"]

def test_api_parallel(url, num_requests, data):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        latencies_responses = list(executor.map(lambda _: make_request(url, data), range(num_requests)))

    latencies = [latency for latency, _ in latencies_responses]
    responses = [response for _, response in latencies_responses]
    return latencies, responses, time.time() - start_time

def test_api_parallel_elastic(urls, threshold, num_requests, data, switch):
    start_time = time.time()

    if switch:
        def make_request_and_switch_url(request_num):
            nonlocal urls, threshold
            current_url = urls[request_num % len(urls)]
            response_time, prediction = make_request(current_url, data)
            if response_time > threshold:
                # print(f"Switching URL for request {request_num} from {current_url} to next URL due to high response time: {response_time}")
                next_url = urls[(request_num + 1) % len(urls)]
                response_time, prediction = make_request(next_url, data)
            return response_time, prediction

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            latencies_responses = list(executor.map(make_request_and_switch_url, range(num_requests)))
    else:
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            latencies_responses = list(executor.map(lambda _: make_request(urls[0], data), range(num_requests)))

    latencies = [latency for latency, _ in latencies_responses]
    responses = [response for _, response in latencies_responses]
    return latencies, responses, time.time() - start_time

if __name__ == '__main__':
    labels = ["Normal", "SlowD", "SuddenD", "SuddenR", "InstaD"]
    model_names = ['hivecote2', 'inception', 'rp', 'MTF', 'NVG']
    model_name = model_names[4]
    
    ingress = 'localhost'
    external_port = 31634
    external_port = get_loadbalancer_external_port(service_name='ingress-nginx-controller')
    url = f'http://{ingress}:{external_port}/{model_name}'

    with open('api_tester/data_new.pkl', 'rb') as f:
        data_all = pickle.load(f)

    nodes = init_nodes(debug=True, custom_label='app=nvg-api')

    # when using the load balancer
    # available_nodes = [node for node in nodes if len(node.get_containers()) > 0]
    # urls = [f'http://{node.ip}:{external_port}/{model_name}' for node in available_nodes]

    if len(sys.argv) > 1:
        parallel_requests = int(sys.argv[1])
    else:
        parallel_requests = 100
    threshold = 5.0

    data_t = np.expand_dims(data_all[0], axis=0)
    # latencies, responses, duration = test_api_parallel_elastic(urls, threshold, num_requests=parallel_requests, data=data_t.tolist(), switch=True)
    latencies, responses, duration = test_api_parallel(url, num_requests=parallel_requests, data=data_t.tolist())
    print(f"Mean of latencies {np.array(latencies).mean()}")
    print(f"{model_name} with {parallel_requests} requests - Done in {duration} s")
