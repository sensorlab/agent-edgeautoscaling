import numpy as np
import time
import subprocess
import multiprocessing
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils import init_nodes
from spam_cluster import get_response_times
from pod_controller import get_loadbalancer_external_port, set_container_cpu_values
from infer import initialize_agents, infer
from threshold_elasticity import run_thold_elasticity

'''
This script generates results for the priority evaluation for MARLISE agents
and is meant to run with the separate_services configuration: localization-api1, localization-api2...
'''
if __name__ == '__main__':
    # Test Settings
    algorithms = ['thold', 'dqn', 'ppo', 'ddpg']
    
    # Results are the same for thold (lets say)
    # algorithms = ['dqn', 'ppo', 'ddpg']
    # algorithms = ['ddpg']
    # algorithms = ['ppo']
    recordings = 20
    num_iterations = 21
    rps = 100

    # priorities = [0.1, 1.0, 0.5] # LHM
    # priorities = [0.5, 0.1, 1.0] # MLH
    priorities = [1.0, 0.5, 0.1] # HML
    # second_priorities = [0.1, 1.0, 0.1]

    first_loads = [0.24879369, 0.08459699, 0.66660932]
    second_loads = [0.24550783, 0.72418413, 0.03030803]
    third_loads = [0.47402381, 0.07211604, 0.45386015]

    folder_path = f'results/generated/priority/ddpg_monte_carlo_{num_iterations}_hml'
    os.makedirs(folder_path, exist_ok=True)

    # Misc Settings
    interval = 1000

    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}"
    USERS = 5
    time_step = 1

    initial_container_cpu = 50

    nodes = init_nodes(debug=True, custom_label='app=localization-api')
    apis = 3

    crec_alg = {alg: [] for alg in algorithms}
    rts_alg = {alg: [] for alg in algorithms}

    deltas_0 = {algorithm: [] for algorithm in algorithms}
    deltas_1 = {algorithm: [] for algorithm in algorithms}
    deltas_2 = {algorithm: [] for algorithm in algorithms}

    all_response_times = {}

    # command = ['python', 'src/spam_cluster.py', '--users', str(rps), '--interval', str(interval), '--all']

    manager = multiprocessing.Manager()

    for alg in algorithms:
        iterations_all_response_times = {api_id: [] for api_id in range(1, apis + 1)}

        for iteration in range(num_iterations):
            print(f"Running {alg} iteration {iteration}/{num_iterations}")

            set_container_cpu_values(cpus=1000)
            for node in nodes:
                for container_id, (_, _, _) in list(node.get_containers().items()):
                    (_, _, cpu_percentage), (_, _, _), (_, _), _ = node.get_container_usage(container_id)
                    if cpu_percentage > 20:
                        time.sleep(1.5)
                    else:
                        break
            set_container_cpu_values(cpus=initial_container_cpu)
            time.sleep(5)

            match alg:
                case 'thold':
                    queue = multiprocessing.Queue()
                    infer_process = multiprocessing.Process(target=run_thold_elasticity, args=(queue,))
                    infer_process.start()
                case 'dqn':
                    envs, agents = initialize_agents(n_agents=3, resources=1000, tl_agent=2,
                                        model='trained/dqn/mdqn1000ep1000m25inc2_rf_20rps5.0alpha1000res',
                                        algorithm='mdqn', independent=False, priorities=priorities)
                case 'ppo':
                    envs, agents = initialize_agents(n_agents=3, resources=1000, tl_agent=0,
                                        model='trained/ppo/1000ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources',
                                        algorithm='ppo', independent=False, priorities=priorities, scale_action=100)
                case 'ddpg':
                    # priorities = [1.0, 0.75, 0.85]
                    # priorities = [0.85, 1.0, 0.75]
                    # second_priorities = [1.0, 0.75, 1.0]
                    envs, agents = initialize_agents(n_agents=3, resources=1000, tl_agent=0,
                                        model='trained/ddpg/1000ep_2rf_20rps5.0alpha_50scale1000resources',
                                        algorithm='ddpg', independent=False, priorities=priorities)

            if alg != 'thold':
                for env in envs:
                    env.cummulative_delta = 0
                # Changing on the fly priorities on the thread with a shared env manager
                shared_envs = manager.list([manager.dict({'priority': env.priority, 'cummulative_delta': env.cummulative_delta}) 
                                            for env in envs])
                infer_process = multiprocessing.Process(target=infer, args=(agents, envs, 1000, False, 1, shared_envs))
                infer_process.start()
                time.sleep(15)

            commands = []
            for i, load in enumerate(first_loads):
            # for i, load in enumerate(first_loads[iteration]):
                load = int(rps * load)
                if load > 0:
                    commands.append(['python', 'src/spam_cluster.py', '--users', str(load), 
                                    '--interval', str(interval), '--service', str(i + 1)])
            subprocesses = [subprocess.Popen(command) for command in commands]

            container_recordings = []
            response_times = {api_id: [] for api_id in range(1, apis + 1)}

            try:
                for i in range(recordings):
                    # Change the priorities at half and see what happens
                    # if alg != 'thold':
                        # if i == (recordings // 2):
                        #     for i, priority in enumerate(second_priorities):
                        #         shared_envs[i]['priority'] = priority
                        #         # envs[i].priority = priority
                        #     print("Changed priorities!")
                    # I also want to see what happens when the load is disabled, which service is going to scale first/last
                    # if i == (recordings // 2):
                    #     spam_process.terminate()
                    #     spam_process.wait()
                    if i == (recordings // 3):
                        for proc in subprocesses:
                            proc.terminate()
                            proc.wait()
                        subprocesses.clear()
                        for api_id, load in enumerate(second_loads):
                        # for api_id, load in enumerate(second_loads[iteration]):
                            load = int(rps * load)
                            if load > 0:
                                subprocesses.append(subprocess.Popen(
                                    ['python', 'src/spam_cluster.py', '--users', str(load), 
                                    '--interval', str(interval), '--service', str(api_id + 1)]))
                    if i == (2 * recordings // 3):
                        for proc in subprocesses:
                            proc.terminate()
                            proc.wait()
                        subprocesses.clear()
                        for api_id, load in enumerate(second_loads):
                        # for api_id, load in enumerate(third_loads[iteration]):
                            load = int(rps * load)
                            if load > 0:
                                subprocesses.append(subprocess.Popen(
                                    ['python', 'src/spam_cluster.py', '--users', str(load), 
                                    '--interval', str(interval), '--service', str(api_id + 1)]))
                    
                    start_time = time.time()

                    for api_id in range(1, apis + 1):
                        rts = [rt for rt in get_response_times(USERS, f'{url}/api{api_id}/predict') if rt is not None]
                        iterations_all_response_times[api_id].extend(rts)
                        mean_rt = np.mean(rts) if rts else float('nan')
                        response_times[api_id].append(mean_rt)

                    node_recordings = []
                    # What the fuck is this shitty hack
                    for api_id in range(1, apis + 1):
                        for node in nodes:
                            for container_id, (pod_name, _, _) in list(node.get_containers().items()):
                                if f'api{api_id}' in pod_name:
                                    (cpu_limit, cpu, cpu_percentage), (_, _, _), (_, _), _ = node.get_container_usage(container_id)
                                    node_recordings.append((cpu_limit, cpu, cpu_percentage))
                    container_recordings.append(node_recordings)

                    elapsed_time = time.time() - start_time
                    time.sleep(max(0, time_step - elapsed_time))
            finally:
                for proc in subprocesses:
                    proc.terminate()
                    proc.wait()
                infer_process.terminate()
                infer_process.join()

                crec_alg[alg].append(container_recordings)
                rts_alg[alg].append(response_times)

            if alg != 'thold':
                deltas_0[alg].append(shared_envs[0]['cummulative_delta'])
                deltas_1[alg].append(shared_envs[1]['cummulative_delta'])
                deltas_2[alg].append(shared_envs[2]['cummulative_delta'])
                print(deltas_0)
            else:
                if not queue.empty():
                    result = queue.get()
                    deltas_0[alg].append(result['localization-api1'])
                    deltas_1[alg].append(result['localization-api2'])
                    deltas_2[alg].append(result['localization-api3'])
        
        all_response_times[alg] = iterations_all_response_times
    

    mean_crec_alg = {algorithm: [[] for _ in range(recordings)] for algorithm in algorithms}
    mean_rts_alg = {algorithm: {api_id: [] for api_id in range(1, apis + 1)} for algorithm in algorithms}

    for algorithm in algorithms:
        for i in range(recordings):
            service_recordings = [[] for _ in range(apis)]
            for iteration in range(num_iterations):
                for service_idx, node_recording in enumerate(crec_alg[algorithm][iteration][i]):
                    service_recordings[service_idx].append(node_recording)
            for service_idx, service_recording in enumerate(service_recordings):
                cpu_limits = [rec[0] for rec in service_recording]
                cpu_usages = [rec[1] for rec in service_recording]
                cpu_percentages = [rec[2] for rec in service_recording]
                mean_crec_alg[algorithm][i].append((
                    np.mean(cpu_limits),
                    np.mean(cpu_usages),
                    np.mean(cpu_percentages)
                ))

        for api_id in range(1, apis + 1):
            for i in range(recordings):
                mean_rts = []
                for iteration in range(num_iterations):
                    mean_rts.append(rts_alg[algorithm][iteration][api_id][i])
                mean_rts_alg[algorithm][api_id].append(np.mean(mean_rts))
    
    for alg in algorithms:
        print(alg, np.mean(np.array(list(mean_rts_alg[str(alg)].values())).flatten()))


    pickle.dump(mean_crec_alg, open(os.path.join(folder_path, 'mean_crec_alg.p'), 'wb'))
    pickle.dump(mean_rts_alg, open(os.path.join(folder_path, 'mean_rts_alg.p'), 'wb'))

    pickle.dump(all_response_times, open(os.path.join(folder_path, 'all_response_times.p'), 'wb'))
    pickle.dump(deltas_0, open(os.path.join(folder_path, 'deltas_0.p'), 'wb'))
    pickle.dump(deltas_1, open(os.path.join(folder_path, 'deltas_1.p'), 'wb'))
    pickle.dump(deltas_2, open(os.path.join(folder_path, 'deltas_2.p'), 'wb'))
