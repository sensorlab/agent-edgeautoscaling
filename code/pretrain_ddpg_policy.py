import time
import subprocess
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from pod_controller import set_container_cpu_values
from utils import save_training_data
from spam_cluster import get_response_times

from envs import ContinuousElasticityEnv
from train_ddpg import DDPGagent, set_available_resource


if __name__ == '__main__':
    DEBUG = True
    custom_app_label = "app=localization-api"
    scale_action = 50
    resources = 1000
    max_cpu = resources
    min_cpu = 50
    UPPER = 60
    LOWER = 30
    action_interval = 1
    AVAILABLE = resources
    print_output = False
    episodes = 100
    reqs_per_second = 50
    interval = 1000
    alpha = 0.75
    gamma_latency = 0.5
    min_rps = 10
    bs = 4
    SAVE_WEIGHTS = True
    debug = True

    url = f"http://localhost:30888/predict"
    USERS = 10

    n_agents = 3
    envs = [ContinuousElasticityEnv(i) for i in range(1, n_agents + 1)]
    for env in envs:
        env.MAX_CPU_LIMIT = resources
        env.MIN_CPU_LIMIT = min_cpu
        env.DEBUG = False
        env.scale_action = scale_action
    agents = [DDPGagent(env, hidden_size=64, max_memory_size=60000) for env in envs]
    parent_dir = 'code/model_metric_data/ddpg/pretrained'
    MODEL = f'{episodes}ep{resources}resources{reqs_per_second}rps{interval}interval{alpha}alpha{scale_action}scale_a{gamma_latency}gl'
    os.makedirs(f'{parent_dir}/{MODEL}', exist_ok=True)

   
    print(f"Pretraining tresholding models with {n_agents} agents for {episodes} episodes with {resources} resources, {reqs_per_second} requests per second, {interval} ms interval, {alpha} alpha, {bs} batch size\nModel name {MODEL}\n")

    rewards = []
    avg_rewards = []
    mean_latencies = []
    agents_summed_rewards = [[] for _ in range(n_agents)]

    set_container_cpu_values(100)
    set_available_resource(envs, resources)

    for episode in tqdm(range(episodes)):
        random_rps = np.random.randint(10, reqs_per_second)
        spam_process = subprocess.Popen(['python', 'code/spam_cluster.py', '--users', str(random_rps), '--interval', str(interval)])
        print(f"Loading cluster with {random_rps} requests per second")

        states = [np.array(env.reset()).flatten() for env in envs]
        set_available_resource(envs, resources)

        ep_latencies = []
        ep_rewards = []
        agents_ep_reward = [[] for _ in range(n_agents)]

        for step in range(envs[0].MAX_STEPS):
            start_time = time.time()
            agents_step_rewards = []

            latencies = get_response_times(USERS, url)
            latency = np.mean([latency for latency in latencies if latency is not None])
            ep_latencies.append(latency)

            shared_reward = 1 - latency * 10
            # latency = min(latency, gamma_latency)
            # shared_reward = (gamma_latency - latency) / gamma_latency

            new_states, dones = [], []
            for i, env in enumerate(envs):
                # action
                (cpu_limit, cpu, cpu_p), (_, _, _), (_, _) = env.get_container_usage()
                if cpu_p > UPPER:
                    action = np.random.uniform(0.8, 1)
                elif cpu_p < LOWER:
                    action = np.random.uniform(-0.8, -1)
                else:
                    action = np.random.uniform(-0.1, 0.1)

                scale_for = scale_action * action
                if env.AVAILABLE >= scale_for:
                    env.patch(int(max(env.ALLOCATED + scale_for, env.MIN_CPU_LIMIT)))

                set_available_resource(envs, AVAILABLE)

                new_state, agent_reward, done, _ = env.mimic_step()
                new_state = np.array(new_state).flatten()

                reward = alpha * agent_reward + (1 - alpha) * shared_reward

                agents_ep_reward[i].append(reward)
                agents_step_rewards.append(reward)

                action = np.array([action], dtype=np.float32)

                agents[i].memory.push(states[i], action, reward, new_state, done)
                new_states.append(new_state)
                dones.append(done)
                if len(agents[i].memory) > bs:
                    agents[i].update(bs)

                if debug:
                    print(f"Agent {env.id}, ACTION: {action}, LIMIT: {env.ALLOCATED}, AVAILABLE: {env.AVAILABLE}, reward: {reward} state(limit, usage, others): {env.state[-1]}, shared_reward: {shared_reward}, agent_reward: {agent_reward}")
            if debug:
                print()
            if step % 30 == 0 and step != 0:
                print(f"Shared: {agents_step_rewards}, latency: {latency}")
                for env in envs:
                    print(f"Agent {env.id}: {env.last_cpu_percentage} % CPU, AVAILABLE: {env.AVAILABLE}", end=". ")
                print()

            ep_rewards.append(np.mean(agents_step_rewards))

            states = new_states
            elapsed_time = time.time() - start_time
            if elapsed_time < action_interval:
                time.sleep(action_interval - elapsed_time)
            
            if any(dones):
                break
        
        mean_latencies.append(np.mean(ep_latencies))
        rewards.append(sum(ep_rewards))
        [agents_summed_rewards[i].append(np.sum(reward)) for i, reward in enumerate(agents_ep_reward)]

        spam_process.terminate()
        set_container_cpu_values(1000)
        for i in range(n_agents):
            while True:
                (_, _, cpu_percentage), (_, _, _), (_, _) = envs[i].node.get_container_usage(envs[i].container_id)
                if cpu_percentage > 20:
                    time.sleep(5)
                else:
                    break
        
        for env in envs:
            env.set_last_limit()
        
        print(f"Episode {episode} reward: {rewards[-1]} mean latency: {np.mean(ep_latencies)}")
                    
    if SAVE_WEIGHTS:
        for i, agent in enumerate(agents):
            agent.save_model(f"{parent_dir}/{MODEL}/agent_{i}")
        
        save_training_data(f'{parent_dir}/{MODEL}', rewards, mean_latencies, agents_summed_rewards)
