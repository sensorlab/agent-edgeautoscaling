import os

import numpy as np
import time
import threading
from pydantic import BaseModel
from fastapi import FastAPI

from infer import initialize_agents
from envs import set_available_resource, set_other_priorities, set_other_utilization


class Application:
    def __init__(self):
        self.resources = 1000
        self.debug = False
        self.action_interval = 1
        self.current_algorithm = None
        self.set_ppo()
        self.infer_thread = None
        self.stop_signal = threading.Event()

    def infer_mdqn(self):
        other_envs = [[env for env in self.envs if env != self.envs[i]] for i in range(len(self.envs))]

        states = [np.array(env.reset()).flatten() for env in self.envs]
        while not self.stop_signal.is_set():
            start_time = time.time()
            actions = [agent.get_action(state) for state, agent in zip(states, self.agents)]
            states, rewards, dones, _ = [], [], [], []
            for i, action in enumerate(actions):
                set_other_utilization(self.envs[i], other_envs[i])
                set_other_priorities(self.envs[i], other_envs[i])

                state, reward, done, _ = self.envs[i].step(action, 2)
                set_available_resource(self.envs, self.resources)
                states.append(np.array(state).flatten())
                rewards.append(reward)
                dones.append(done)
                if self.debug:
                    print(
                        f"{self.envs[i].id}: ACTION: {action}, LIMIT: {self.envs[i].ALLOCATED}, "
                        f"{self.envs[i].last_cpu_percentage: .2f}%, AVAILABLE: {self.envs[i].AVAILABLE}, "
                        f"reward: {reward} state(limit, usage, others): {self.envs[i].state[-1]}")
            if self.debug:
                print()

            elapsed_time = time.time() - start_time
            if elapsed_time < self.action_interval:
                time.sleep(self.action_interval - elapsed_time)

            if self.stop_signal.is_set():
                break

    def start_inference(self):
        if self.infer_thread is None or not self.infer_thread.is_alive():
            self.stop_signal.clear()
            self.infer_thread = threading.Thread(target=self.infer_mdqn)
            self.infer_thread.start()
            return {"message": "Inference started"}
        else:
            return {"message": "Inference already running"}

    def stop_inference(self):
        if self.infer_thread is not None and self.infer_thread.is_alive():
            self.stop_signal.set()
            return {"message": "Inference stopped"}
        else:
            return {"message": "Inference not running"}

    def set_resources(self, new_resources):
        self.resources = new_resources
        for env in self.envs:
            env.MAX_CPU_LIMIT = new_resources
        set_available_resource(self.envs, new_resources)
        if self.debug:
            print(f"Resources set to {new_resources}")
        return {"message": f"Resources set to {new_resources}"}

    def set_dqn(self):
        if self.current_algorithm != 'dqn':
            self.envs, self.agents = initialize_agents(n_agents=3, resources=1000, tl_agent=2,
                                                       model='trained/dqn/mdqn1000ep1000m25inc2_rf_20rps5.0alpha1000res',
                                                       algorithm='mdqn', independent=False, priorities=[1.0, 1.0, 1.0])
            self.current_algorithm = 'dqn'
            return {"message": "DQN algorithm set"}
        else:
            return {"message": "DQN algorithm is already set"}

    def set_ppo(self):
        if self.current_algorithm != 'ppo':
            self.envs, self.agents = initialize_agents(n_agents=3, resources=1000, tl_agent=0,
                                                       model='trained/ppo/1000ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources',
                                                       algorithm='ppo', independent=False, priorities=[1.0, 1.0, 1.0])
            self.current_algorithm = 'ppo'
            return {"message": "PPO algorithm set"}
        else:
            return {"message": "PPO algorithm is already set"}



app = Application()
elasticity_app = FastAPI()


class IntervalRequest(BaseModel):
    interval: int


class ResourceRequest(BaseModel):
    resources: int


@elasticity_app.post("/start")
def start_inference():
    return app.start_inference()


@elasticity_app.post("/stop")
def stop_inference():
    return app.stop_inference()


@elasticity_app.post("/set_resources")
def set_resources(item: ResourceRequest):
    return app.set_resources(item.resources)


@elasticity_app.post("/set_interval")
def set_interval(request: IntervalRequest):
    app.action_interval = request.interval
    return {"message": f"Interval set to {request.interval}"}


@elasticity_app.post("/set_dqn_algorithm")
def set_dqn():
    return app.set_dqn()


@elasticity_app.post("/set_ppo_algorithm")
def set_ppo():
    return app.set_ppo()


@elasticity_app.get("/status")
def get_status():
    return {"status": app.infer_thread.is_alive() if app.infer_thread is not None else False}


@elasticity_app.get("/algorithm")
def get_algorithm():
    return {"algorithm": app.current_algorithm}


@elasticity_app.get("/resources")
def get_resources():
    return {"resources": app.resources}


@elasticity_app.get("/interval")
def get_interval():
    return {"interval": app.action_interval}
