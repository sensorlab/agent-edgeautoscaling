import random
import time
import argparse

from utils import make_request
from pod_controller import get_loadbalancer_external_port

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=10, help="Number of users")
    parser.add_argument("--interval", type=int, default=250, help="Interval of spamming (in seconds)")
    args = parser.parse_args()

    num_users = args.users
    interval = args.interval / 1000

    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}/predict"
    data = {
        "feature": 0
    }

    while True:
        for _ in range(num_users + 1):
            data["feature"] = random.randint(0, 130)
            make_request(url, data)
        time.sleep(interval)