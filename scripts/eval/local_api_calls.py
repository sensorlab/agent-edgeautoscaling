import requests

BASE_URL = "http://localhost:8000"

"""
This script is used to make API calls to the local server running MARLISE agents.
"""


def get_resource():
    resources = requests.get(f"{BASE_URL}/resources").json()["resources"]
    print(f"Resources: {resources}")
    return resources


def start_inference():
    response = requests.post(f"{BASE_URL}/start")
    print(response.json())


def stop_inference():
    response = requests.post(f"{BASE_URL}/stop")
    print(response.json())


def set_resources(resources):
    response = requests.post(f"{BASE_URL}/set_resources", json={"resources": resources})
    print(response.json())


def set_interval(interval):
    response = requests.post(f"{BASE_URL}/set_interval", json={"interval": interval})
    print(response.json())


def set_dqn_algorithm():
    response = requests.post(f"{BASE_URL}/set_dqn_algorithm")
    print(response.json())


def set_ppo_algorithm():
    response = requests.post(f"{BASE_URL}/set_ppo_algorithm")
    print(response.json())


def set_ddpg_algorithm():
    response = requests.post(f"{BASE_URL}/set_ddpg_algorithm")
    print(response.json())


def add_agent():
    response = requests.post(f"{BASE_URL}/add_agent")
    print(response.json())


def remove_agent():
    response = requests.post(f"{BASE_URL}/remove_agent")
    print(response.json())


def remove_first_agent():
    response = requests.post(f"{BASE_URL}/remove_first_agent")
    print(response.json())


def get_status():
    response = requests.get(f"{BASE_URL}/status")
    print(response.json())


def get_algorithm():
    response = requests.get(f"{BASE_URL}/algorithm")
    print(response.json())


def get_resources():
    response = requests.get(f"{BASE_URL}/resources")
    print(response.json())


def get_interval():
    response = requests.get(f"{BASE_URL}/interval")
    print(response.json())


def add_dqn_agent():
    response = requests.post(f"{BASE_URL}/add_dqn_agent")
    print(response.json())


def add_ppo_agent():
    response = requests.post(f"{BASE_URL}/add_ppo_agent")
    print(response.json())


def add_ddpg_agent():
    response = requests.post(f"{BASE_URL}/add_ddpg_agent")
    print(response.json())
