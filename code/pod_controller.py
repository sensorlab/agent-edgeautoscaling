from kubernetes import client, config
from utils import increment_last_number


# wip
def remove_worker_pods_except_first(debug=False):
    if debug:
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
    namespace = 'default'

    try:
        worker_pods = v1.list_namespaced_pod(namespace=namespace, label_selector="component=ray-worker")
    except Exception as e:
        print(f"Error reading worker pods: {e}")
        return

    if len(worker_pods.items) <= 1:
        return

    for pod in worker_pods.items[1:]:
        try:
            v1.delete_namespaced_pod(name=pod.metadata.name, namespace=namespace)
            print(f"Deleted pod: {pod.metadata.name}")
        except Exception as e:
            print(f"Error deleting pod {pod.metadata.name}: {e}")


# wip
def create_worker_pod(node_name=None, debug=False):
    if debug:
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
    namespace = 'default'

    try:
        source_pod = v1.list_namespaced_pod(namespace=namespace, label_selector="component=ray-worker")
    except Exception as e:
        print(f"Error reading source pod: {e}")
        exit(1)

    new_pod = client.V1Pod()
    new_pod.metadata = client.V1ObjectMeta(
        name=f'{source_pod.items[-1].metadata.name}-new',
        namespace=namespace,
        labels={
            "type": "ray",
            "app": "ray-cluster-worker",
            "component": "ray-worker"
        }
    )

    new_pod.spec = source_pod.items[-1].spec
    if node_name:
        new_pod.spec.node_selector = None
        new_pod.spec.node_name = node_name

    try:
        created_pod = v1.create_namespaced_pod(namespace=namespace, body=new_pod)
        print(f"New pod '{created_pod.metadata.name}' created successfully")
    except Exception as e:
        print(f"Error creating pod: {e}")


def create_pod_from(source_pod_name, node_name=None, debug=False):
    if debug:
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
    namespace = 'default'

    try:
        source_pod = v1.read_namespaced_pod(name=source_pod_name, namespace=namespace)
    except Exception as e:
        print(f"Error reading source pod: {e}")
        exit(1)

    new_pod = client.V1Pod()
    new_pod.metadata = client.V1ObjectMeta(
        name=increment_last_number(source_pod_name),
        namespace=namespace,
        labels=source_pod.metadata.labels
    )

    new_pod.spec = source_pod.spec
    if node_name:
        new_pod.spec.node_selector = None
        new_pod.spec.node_name = node_name

    try:
        created_pod = v1.create_namespaced_pod(namespace=namespace, body=new_pod)
        print(f"New pod '{created_pod.metadata.name}' created successfully")
    except Exception as e:
        print(f"Error creating pod: {e}")


def patch_pod(pod_name, cpu_request="1", cpu_limit="1", memory_request=None, memory_limit=None,
              container_name="ray-worker", debug=False):
    if debug:
        config.load_kube_config()
    else:
        config.load_incluster_config()
    api_instance = client.CoreV1Api()

    if memory_request is not None and memory_limit is not None:
        patch = {
            "spec": {
                "containers": [
                    {
                        "name": container_name,
                        "resources": {
                            "requests": {"cpu": cpu_request, "memory": memory_request},
                            "limits": {"cpu": cpu_limit, "memory": memory_limit}
                        }
                    }
                ]
            }
        }
    else:
        patch = {
            "spec": {
                "containers": [
                    {
                        "name": container_name,
                        "resources": {
                            "requests": {"cpu": cpu_request},
                            "limits": {"cpu": cpu_limit}
                        }
                    }
                ]
            }
        }

    try:
        api_instance.patch_namespaced_pod(
            name=pod_name,
            namespace="default",
            body=patch,
        )
        print(f"Pod {pod_name} patched successfully to {cpu_request} request and {cpu_limit} limit")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    # patch_pod('ray-worker-pod', cpu_request="500m", cpu_limit="500m", memory_limit="2Gi", memory_request="1Gi")
    # patch_pod('ray-head-pod', cpu_request="1", cpu_limit="1500m")
    # patch_pod('ray-worker-pod', cpu_request="500m", cpu_limit="500m")
    # patch_pod('ray-worker-pod1', cpu_request="2", cpu_limit="2")
    # patch_pod('ray-worker-pod', cpu_request="1", cpu_limit="1500m")
    # create_pod_from('ray-worker-pod', node_name='jovyan-thinkpad-l14-gen-1')
    create_pod_from('ray-worker-pod', node_name='e6-orancloud')
    # create_worker_pod()
    # create_worker_pod()
    # create_worker_pod()
    # create_worker_pod('jovyan-thinkpad-l14-gen-1')
    # remove_worker_pods_except_first()
