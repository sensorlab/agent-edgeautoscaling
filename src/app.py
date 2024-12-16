import numpy as np
import time
import threading
import pickle

from pydantic import BaseModel
from fastapi import FastAPI

from infer import initialize_agents, initialize_agent
from envs import set_available_resource, set_other_priorities, set_other_utilization


class Application:
    def __init__(self):
        # Metrics collection, turn on to collect resoruce changes for each agent
        self.collect_metrics = False
        self.deltas_for_alg = []
        self.first_agent_delta = 0

        self.n_agents = 3
        self.resources = 1000
        self.debug = False
        self.action_interval = 1
        self.current_algorithm = None
        self.set_dqn()
        self.infer_thread = None
        self.stop_signal = threading.Event()
        self.other_envs = self._update_other_envs()
        self.lock = threading.Lock()

    def _update_other_envs(self):
        return [
            [env for env in self.envs if env != self.envs[i]]
            for i in range(len(self.envs))
        ]

    def infer_mdqn(self):
        states = [np.array(env.reset()).flatten() for env in self.envs]
        while not self.stop_signal.is_set():
            with self.lock:
                if len(self.envs) > len(states):
                    states.append(np.array(self.envs[-1].reset()).flatten())

            start_time = time.time()
            actions = []
            with self.lock:
                for state, agent in zip(states, self.agents):
                    actions.append(agent.get_action(state))
            # actions = [agent.get_action(state) for state, agent in zip(states, self.agents)]
            states, rewards, dones, _ = [], [], [], []
            for i, action in enumerate(actions):
                with self.lock:
                    if i >= len(self.envs) or i >= len(self.other_envs):
                        continue  # Skip if index is out of range
                    set_other_utilization(self.envs[i], self.other_envs[i])
                    set_other_priorities(self.envs[i], self.other_envs[i])

                    state, reward, done, _ = self.envs[i].step(action, 2)
                    set_available_resource(self.envs, self.resources)
                    states.append(np.array(state).flatten())
                    rewards.append(reward)
                    dones.append(done)
                    if self.debug:
                        print(
                            f"{self.envs[i].id}: ACTION: {action}, LIMIT: {self.envs[i].ALLOCATED}, "
                            f"{self.envs[i].last_cpu_percentage: .2f}%, AVAILABLE: {self.envs[i].AVAILABLE}, "
                            f"reward: {reward} state(limit, usage, others): {self.envs[i].state[-1]}"
                        )
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
            self.infer_thread.join()

            if self.collect_metrics:
                deltas = []
                deltas.append(self.first_agent_delta)
                for env in self.envs:
                    deltas.append(env.cummulative_delta)
                    env.cummulative_delta = 0

                self.deltas_for_alg.append(deltas)
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
        if self.collect_metrics and self.current_algorithm != "dqn":
            pickle.dump(
                self.deltas_for_alg,
                open(
                    f"results/generated/scalable_agents/{self.current_algorithm}_deltas.p",
                    "wb",
                ),
            )
            self.deltas_for_alg = []

        self.envs, self.agents = initialize_agents(
            n_agents=self.n_agents,
            resources=1000,
            tl_agent=2,
            model="trained/dqn/mdqn1000ep1000m25inc2_rf_20rps5.0alpha1000res",
            algorithm="mdqn",
            independent=False,
            priorities=[1.0, 1.0, 1.0],
        )
        self.current_algorithm = "dqn"
        self.other_envs = self._update_other_envs()
        return {"message": "DQN algorithm set"}

    def set_ppo(self):
        if self.collect_metrics and self.current_algorithm != "ppo":
            pickle.dump(
                self.deltas_for_alg,
                open(
                    f"results/generated/scalable_agents/{self.current_algorithm}_deltas.p",
                    "wb",
                ),
            )
            self.deltas_for_alg = []

        self.envs, self.agents = initialize_agents(
            n_agents=self.n_agents,
            resources=1000,
            tl_agent=0,
            model="trained/ppo/1000ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources",
            algorithm="ppo",
            independent=False,
            priorities=[1.0, 1.0, 1.0],
            scale_action=100,
        )
        self.current_algorithm = "ppo"
        self.other_envs = self._update_other_envs()
        return {"message": "PPO algorithm set"}

    def set_ddpg(self):
        if self.collect_metrics and self.current_algorithm != "ddpg":
            pickle.dump(
                self.deltas_for_alg,
                open(
                    f"results/generated/scalable_agents/{self.current_algorithm}_deltas.p",
                    "wb",
                ),
            )
            self.deltas_for_alg = []

        self.envs, self.agents = initialize_agents(
            n_agents=self.n_agents,
            resources=1000,
            tl_agent=0,
            model="trained/ddpg/1000ep_2rf_20rps5.0alpha_50scale1000resources",
            algorithm="ddpg",
            independent=False,
            priorities=[1.0, 1.0, 1.0],
        )
        self.current_algorithm = "ddpg"
        self.other_envs = self._update_other_envs()
        return {"message": "DDPG algorithm set"}

    def add_agent(self):
        match self.current_algorithm:
            case "ppo":
                new_env, new_agent = initialize_agent(
                    id=len(self.envs) + 1,
                    resources=1000,
                    tl_agent=0,
                    model="trained/ppo/1000ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources",
                    algorithm="ppo",
                    independent=False,
                    priority=1.0,
                    scale_action=100,
                )
            case "dqn":
                new_env, new_agent = initialize_agent(
                    id=len(self.envs) + 1,
                    resources=1000,
                    tl_agent=2,
                    model="trained/dqn/mdqn1000ep1000m25inc2_rf_20rps5.0alpha1000res",
                    algorithm="mdqn",
                    independent=False,
                    priority=1.0,
                )
            case "ddpg":
                new_env, new_agent = initialize_agent(
                    id=len(self.envs) + 1,
                    resources=1000,
                    tl_agent=0,
                    model="trained/ddpg/1000ep_2rf_20rps5.0alpha_50scale1000resources",
                    algorithm="ddpg",
                    independent=False,
                    priority=1.0,
                )
            case _:
                return {"message": "No algorithm set"}

        self.envs.append(new_env)
        self.agents.append(new_agent)
        self.other_envs = self._update_other_envs()
        return {"message": "Agent added"}

    def remove_agent(self):
        with self.lock:
            if len(self.envs) > 1:
                self.envs[-1].patch(self.envs[-1].MIN_CPU_LIMIT)
                self.envs.pop(-1)
                self.agents.pop(-1)
                self.other_envs = self._update_other_envs()
                return {"message": "Last agent removed"}
            else:
                return {"message": "Cannot remove the only agent"}

    def remove_first_agent(self):
        with self.lock:
            if len(self.envs) > 1:
                self.envs[0].patch(
                    self.envs[0].MIN_CPU_LIMIT
                )  # Reset the first agent to its initial state
                self.first_agent_delta = self.envs[0].cummulative_delta
                self.envs.pop(0)
                self.agents.pop(0)
                self.other_envs = self._update_other_envs()
                return {"message": "First agent removed"}
            else:
                return {"message": "Cannot remove the only agent"}

    def set_default_limits(self):
        for env in self.envs:
            env.patch(env.MIN_CPU_LIMIT)
        return {"message": "Default limits set"}


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


@elasticity_app.post("/set_default_limits")
def set_default_limits():
    return app.set_default_limits()


@elasticity_app.post("/set_ddpg_algorithm")
def set_ddpg():
    return app.set_ddpg()


@elasticity_app.post("/add_agent")
def add_agent():
    return app.add_agent()


@elasticity_app.post("/remove_agent")
def remove_agent():
    return app.remove_agent()


@elasticity_app.post("/remove_first_agent")
def remove_first_agent():
    return app.remove_first_agent()


@elasticity_app.get("/status")
def get_status():
    return {
        "status": app.infer_thread.is_alive() if app.infer_thread is not None else False
    }


@elasticity_app.get("/algorithm")
def get_algorithm():
    return {"algorithm": app.current_algorithm}


@elasticity_app.get("/resources")
def get_resources():
    return {"resources": app.resources}


@elasticity_app.get("/interval")
def get_interval():
    return {"interval": app.action_interval}
