import random
import time
import argparse
from utils import make_request
from pod_controller import get_loadbalancer_external_port
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor


latencies = []
latencies_lock = Lock()


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


def make_request_thread(url, data, interval):
    while True:
        try:
            data["feature"] = random.randint(0, 130)
            make_request(url, data)
        except Exception as e:
            pass
        time.sleep(interval)


def spam_requests(url, num_users, interval):
    data = {
        "feature": 0
    }

    threads = []
    for _ in range(num_users + 1):
        thread = threading.Thread(target=make_request_thread, args=(url, data, interval))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=100, help="Number of users")
    parser.add_argument("--interval", type=int, default=500, help="Interval of spamming (in milliseconds)")
    args = parser.parse_args()

    num_users = args.users
    interval = args.interval / 1000

    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}/predict"
    spam_requests(url, num_users, interval)
