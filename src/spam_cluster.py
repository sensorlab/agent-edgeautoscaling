import argparse
import atexit
import os
import random
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from pod_controller import get_loadbalancer_external_port
from utils import make_request


# These 2 functions are not part of the loading process
def process_single_request(url, data):
    try:
        data["feature"] = random.randint(0, 130)
        rt = make_request(url, data)
        if rt is not None:
            return rt
    except Exception as e:
        pass


def get_response_times(num_users, url):
    data = {"feature": 0}

    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = {
            executor.submit(process_single_request, url, data) for _ in range(num_users)
        }
        return [future.result() for future in futures]


# Loading the cluster to simulate real world scenario
def make_request_thread(url, data, interval, variable=False):
    while True:
        try:
            local_data = data.copy()
            local_data["feature"] = random.randint(0, 130)
            make_request(url, local_data)
        except Exception as e:
            pass

        sleep_time = (
            random.uniform(interval - interval / 2, interval + interval / 2)
            if variable
            else interval
        )
        time.sleep(sleep_time)


def spam_requests(url, num_users, interval, variable=False):
    with ThreadPoolExecutor(max_workers=num_users) as executor:
        [
            executor.submit(
                make_request_thread, url, {"feature": 0}, interval, variable
            )
            for _ in range(num_users)
        ]
    # No need to wait for results or handle completed threads


def terminate_processes(processes):
    for process in processes:
        try:
            os.killpg(
                os.getpgid(process.pid), signal.SIGTERM
            )  # Terminate the process group
        except ProcessLookupError:
            pass  # Process already terminated


def signal_handler(signum, frame, processes):
    terminate_processes(processes)
    exit(0)


if __name__ == "__main__":
    """
    Load the cluster with synthetic requests to simulate real world scenario.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=2, help="Number of users")
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Interval of spamming (in milliseconds)",
    )
    parser.add_argument(
        "--service", type=int, default=1, help="Service 1, 2, 3, 4, 5..."
    )

    parser.add_argument(
        "--random_rps",
        action="store_true",
        default=False,
        help="Randomize num_users for each service",
    )
    parser.add_argument(
        "--variable",
        action="store_true",
        default=False,
        help="Variable number of users every interval",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Load the cluster on all services",
    )
    args = parser.parse_args()

    num_users = args.users
    interval = args.interval / 1000
    variable = args.variable
    service = args.service
    all_services = args.all
    random_rps = args.random_rps

    processes = []
    ingress_port = get_loadbalancer_external_port(
        service_name="ingress-nginx-controller"
    )

    if not all_services:
        # url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}/api{service}/predict"
        # url = f"http://localhost:{ingress_port}/api{service}/predict"  # Out ingress port of the service
        url = f"http://localhost:{ingress_port}/predict"  # Ingress port of the service

        # url = f"http://localhost:{ingress_port}/predict" # API for HPA scaling,
        # 1 igress linked to services to many pods (Round Robin load balancing)
        spam_requests(url, num_users, interval, variable=variable)
        print(f"Loaded the cluster with {num_users} users on {url}")
    else:
        urls = [
            f"http://localhost:{ingress_port}/api1/predict",
            f"http://localhost:{ingress_port}/api2/predict",
            f"http://localhost:{ingress_port}/api3/predict",
            # f"http://localhost:{ingress_port}/api4/predict",
        ]
        with ThreadPoolExecutor() as executor:
            futures = []
            for url in urls:
                users = random.randint(1, num_users) if random_rps else num_users
                print(f"Loading the cluster with {users} users on {url}")
                command = [
                    "python3",
                    __file__,
                    "--users",
                    str(users),
                    "--interval",
                    str(args.interval),
                    "--service",
                    url.split("/api")[1].split("/predict")[0],
                ]
                if variable:
                    command.append("--variable")
                future = executor.submit(
                    subprocess.Popen,
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                )
                futures.append(future)

            processes = [future.result() for future in futures]
            atexit.register(terminate_processes, processes)
            signal.signal(
                signal.SIGTERM,
                lambda signum, frame: signal_handler(signum, frame, processes),
            )

            for process in processes:
                process.wait()
