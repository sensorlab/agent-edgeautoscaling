import ray

from utils import init_nodes


def execute_on_best_pod(function, *args, **kwargs):
    @ray.remote(num_cpus=4)
    def wrapper_function():
        return function(*args, **kwargs)

    nodes = init_nodes(True)

    # find head
    ray_head_ip = ''
    for node in nodes:
        for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
            if container_name == "ray-head":
                ray_head_ip = pod_ip

    ray.init(f"ray://{ray_head_ip}:10001")
    print(ray.available_resources())

    best_cpu_p, best_pod_ip, best_pod_name = 100.0, '', ''
    for node in nodes:
        for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
            (_, _, cpu_p), (_, _, _) = node.get_container_usage(container_id)
            if cpu_p < best_cpu_p:
                best_cpu_p, best_pod_ip, best_pod_name = cpu_p, pod_ip, pod_name

    print(f"Started ray job on pod {best_pod_name}:{best_pod_ip} with {best_cpu_p}%")
    result = ray.get(wrapper_function.options(resources={f"node:{best_pod_ip}": 1.0}).remote())

    ray.shutdown()
    return result
