import random
import time
import argparse
from utils import make_request
from concurrent.futures import ThreadPoolExecutor

from pod_controller import get_loadbalancer_external_port


def make_request_thread_single(url, data):
    try:
        data["feature"] = random.randint(0, 130)
        latency = make_request(url, data)
        if latency is not None:
            return latency
    except Exception as e:
        pass


def spam_requests_single(num_users, url):
    data = {
        "feature": 0
    }

    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = {executor.submit(make_request_thread_single, url, data) for _ in range(num_users)}
        return [future.result() for future in futures]


def make_request_thread(url, data, interval, variable=False):
    while True:
        try:
            local_data = data.copy()
            local_data["feature"] = random.randint(0, 130)
            make_request(url, local_data)
        except Exception as e:
            pass

        sleep_time = random.uniform(interval - interval / 2, interval + interval / 2) if variable else interval
        time.sleep(sleep_time)


def spam_requests(url, num_users, interval, variable=False):
    with ThreadPoolExecutor(max_workers=num_users) as executor:
        [
            executor.submit(make_request_thread, url, {"feature": 0}, interval, variable)
            for _ in range(num_users)
        ]
    # No need to wait for results or handle completed threads

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=2, help="Number of users")
    parser.add_argument("--interval", type=int, default=1000, help="Interval of spamming (in milliseconds)")
    parser.add_argument("--service", type=int, default=1, help="Service 1, 2, 3, 4, 5...")
    parser.add_argument('--variable', action='store_true', default=False, help="Variable number of users every interval")
    parser.add_argument('--all', action='store_true', default=False, help="Load the cluster on all services")
    args = parser.parse_args()

    num_users = args.users
    interval = args.interval / 1000
    variable = args.variable
    service = args.service
    all_services = args.all

    if not all_services:
        # url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}/api{service}/predict"
        url = f"http://localhost:32122/api{service}/predict" # Out ingress port of the service
        spam_requests(url, num_users, interval, variable=variable)
    else:
        import subprocess
        urls = [
            f"http://localhost:32122/api1/predict",
            f"http://localhost:32122/api2/predict",
            f"http://localhost:32122/api3/predict"
        ]
        processes = []
        for url in urls:
            process = subprocess.Popen(
                ["python3", __file__, "--users", str(num_users), "--interval", str(args.interval), "--service", url.split('/api')[1].split('/predict')[0], "--variable" if variable else ""],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            processes.append(process)

        for process in processes:
            process.wait()
