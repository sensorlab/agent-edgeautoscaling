'''
Test scenario:
25 25
50
50

4 services:
request to first two

add another agent, request to the 3

remove the frist one agent
'''
import numpy as np
import time
import subprocess
import multiprocessing
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from tqdm import tqdm

from utils import init_nodes
from spam_cluster import get_response_times
from pod_controller import get_loadbalancer_external_port, set_container_cpu_values
from thold_eval import run_thold_elasticity
from local_api_calls import *


'''
How to run: (get rid of multithreading) run with uvicorn app.py in the background,
for the API calls from local_api_calls.py

The script generated scalibility results for MARLISE agents
and is meant to run with the separate_services configuration: localization-api1, localization-api2...
'''
if __name__ == '__main__':
    folder_path = f'results/generated/scalable_agents'
    os.makedirs(folder_path, exist_ok=True)

    recordings = 60
    time_step = 1
    num_iterations = 21

    apis = 4
    rps = [30, 40, 50, 50]
    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}"
    USERS = 5

    action_interval = 1
    resources = 1000
    initial_container_cpu = 50

    nodes = init_nodes(debug=True, custom_label='app=localization-api')

    priorities = [1.0, 1.0, 1.0, 1.0]

    # Algorithms to run and which settings (meant for RL) for them to use
    settings = {
        'ppo': {
            'model': 'trained/ppo/1000ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources',
            'algorithm': 'ppo',
            'tl_agent': 0,
            'scale_action': 100
        },        
        'dqn': {
            'model': 'trained/dqn/mdqn1000ep1000m25inc2_rf_20rps5.0alpha1000res',
            'algorithm': 'mdqn',
            'tl_agent': 2,
            'scale_action': None
        },
        'ddpg': {
            'model': 'trained/ddpg/1000ep_2rf_20rps5.0alpha_50scale1000resources',
            'algorithm': 'ddpg',
            'tl_agent': 0,
            'scale_action': None
        },
        'thold': {
        },
    }
    algorithms = settings.keys()
    print(algorithms)

    crec_alg = {algorithm: [] for algorithm in algorithms}
    rts_alg = {algorithm: [] for algorithm in algorithms}
    all_response_times = {}
    response_times_part1 = {algorithm: {} for algorithm in algorithms}
    response_times_part2 = {algorithm: {} for algorithm in algorithms}
    response_times_part3 = {algorithm: {} for algorithm in algorithms}
    response_times_part4 = {algorithm: {} for algorithm in algorithms}

    for alg in algorithms:
        response_times_part1[alg] = {api_id: [] for api_id in range(1, apis + 1)}
        response_times_part2[alg] = {api_id: [] for api_id in range(1, apis + 1)}
        response_times_part3[alg] = {api_id: [] for api_id in range(1, apis + 1)}
        response_times_part4[alg] = {api_id: [] for api_id in range(1, apis + 1)}

    deltas_0 = {algorithm: [] for algorithm in algorithms}
    deltas_1 = {algorithm: [] for algorithm in algorithms}
    deltas_2 = {algorithm: [] for algorithm in algorithms}
    deltas_3 = {algorithm: [] for algorithm in algorithms}

    for alg in tqdm(algorithms, desc="Algorithms"):
        iterations_all_response_times = {api_id: [] for api_id in range(1, apis + 1)}
        variable_repsonse_times_for_parts = {api_id: [] for api_id in range(1, apis + 1)}

        for iteration in tqdm(range(num_iterations), desc=f"Iterations for {alg}", leave=False):
            print(f'Iteration {iteration + 1}/{num_iterations} for {alg}')

            set_container_cpu_values(cpus=1000, n=apis)
            for node in nodes:
                for container_id, (_, _, _) in list(node.get_containers().items()):
                    (_, _, cpu_percentage), (_, _, _), (_, _), _ = node.get_container_usage(container_id)
                    if cpu_percentage > 20:
                        time.sleep(1.5)
                    else:
                        break
            set_container_cpu_values(cpus=initial_container_cpu, n=apis)
            time.sleep(5)

            response_times = {api_id: [] for api_id in range(1, apis + 1)}
            container_recordings = []

            try:
                match alg:
                    case 'thold':
                        queue = multiprocessing.Queue()
                        infer_process = multiprocessing.Process(target=run_thold_elasticity, args=(queue,))
                        infer_process.start()
                    case 'dqn':
                        set_dqn_algorithm()
                        start_inference()
                        time.sleep(10)
                    case 'ppo':
                        set_ppo_algorithm()
                        start_inference()
                        time.sleep(10)
                    case 'ddpg':
                        set_ddpg_algorithm()
                        start_inference()
                        time.sleep(10)

                commands = []
                for service in range(2):
                    commands.append(['python', 'src/spam_cluster.py', '--users', str(rps[service]), 
                                                '--interval', str(1000), '--service', str(service + 1)])

                subprocesses = [subprocess.Popen(command) for command in commands]

                for recording in range(recordings):
                    if recording == (recordings // 4):
                        for api_id in range(1, apis + 1):
                            response_times_part1[alg][api_id].extend(variable_repsonse_times_for_parts[api_id]) 
                        variable_repsonse_times_for_parts = {api_id: [] for api_id in range(1, apis + 1)}
                        if alg != 'thold':
                            stop_inference() # To write the deltas, and then start it again
                            add_agent()
                            start_inference()
                        else:
                            infer_process.terminate()
                            infer_process.join()
                            if not queue.empty():
                                result = queue.get()
                                deltas_0[alg].append(result['localization-api1'])
                                deltas_1[alg].append(result['localization-api2'])
                                deltas_2[alg].append(result['localization-api3'])
                                deltas_3[alg].append(result['localization-api4'])
                                queue = multiprocessing.Queue()
                                infer_process = multiprocessing.Process(target=run_thold_elasticity, args=(queue,))
                                infer_process.start()
                        subprocesses.append(subprocess.Popen(['python', 'src/spam_cluster.py', '--users', str(rps[2]), 
                                                '--interval', str(1000), '--service', str(3)]))
                    
                    if recording == (2 * recordings // 4):
                        for api_id in range(1, apis + 1):
                            response_times_part2[alg][api_id].extend(variable_repsonse_times_for_parts[api_id]) 
                        variable_repsonse_times_for_parts = {api_id: [] for api_id in range(1, apis + 1)}
                        if alg != 'thold':
                            stop_inference()
                            add_agent()
                            start_inference()
                        else:
                            infer_process.terminate()
                            infer_process.join()
                            if not queue.empty():
                                result = queue.get()
                                deltas_0[alg].append(result['localization-api1'])
                                deltas_1[alg].append(result['localization-api2'])
                                deltas_2[alg].append(result['localization-api3'])
                                deltas_3[alg].append(result['localization-api4'])
                                queue = multiprocessing.Queue()
                                infer_process = multiprocessing.Process(target=run_thold_elasticity, args=(queue,))
                                infer_process.start()
                        subprocesses.append(subprocess.Popen(['python', 'src/spam_cluster.py', '--users', str(rps[3]), 
                                                '--interval', str(1000), '--service', str(4)]))

                    if recording == (3 * recordings // 4):
                        for api_id in range(1, apis + 1):
                            response_times_part3[alg][api_id].extend(variable_repsonse_times_for_parts[api_id])
                        variable_repsonse_times_for_parts = {api_id: [] for api_id in range(1, apis + 1)}
                        if alg != 'thold':
                            stop_inference()
                            remove_first_agent()
                            start_inference()
                        else:
                            from pod_controller import patch_pod
                            patch_pod('localization-api1', cpu_request='50m', cpu_limit='50m', 
                                      container_name='localization-api', debug=True, print_output=True)
                            infer_process.terminate()
                            infer_process.join()
                            if not queue.empty():
                                result = queue.get()
                                deltas_0[alg].append(result['localization-api1'])
                                deltas_1[alg].append(result['localization-api2'])
                                deltas_2[alg].append(result['localization-api3'])
                                deltas_3[alg].append(result['localization-api4'])
                                queue = multiprocessing.Queue()
                                infer_process = multiprocessing.Process(target=run_thold_elasticity, args=(queue,))
                                infer_process.start()
                        subprocesses[0].terminate()

                    start_time = time.time()

                    for api_id in range(1, apis + 1):
                        rts = [rt for rt in get_response_times(USERS, f'{url}/api{api_id}/predict') if rt is not None]
                        iterations_all_response_times[api_id].extend(rts)
                        variable_repsonse_times_for_parts[api_id].extend(rts)
                        mean_rt = np.mean(rts) if rts else float('nan')
                        response_times[api_id].append(mean_rt)

                    node_recordings = []
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
                for process in subprocesses:
                    process.terminate()
                    process.wait()
                if alg == 'thold':
                    infer_process.terminate()
                    infer_process.join()
                else:
                    stop_inference()
                rts_alg[alg].append(response_times)
                crec_alg[alg].append(container_recordings)
            
            if alg == 'thold':
                if not queue.empty():
                    result = queue.get()
                    print(f"Got result: {result}")
                    deltas_0[alg].append(result['localization-api1'])
                    deltas_1[alg].append(result['localization-api2'])
                    deltas_2[alg].append(result['localization-api3'])
                    deltas_3[alg].append(result['localization-api4'])
        all_response_times[alg] = iterations_all_response_times
        for api_id in range(1, apis + 1):
            response_times_part4[alg][api_id].extend(variable_repsonse_times_for_parts[api_id])
    
    set_dqn_algorithm() # Reset to default algorithm, so that deltas get saved :D 

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
    pickle.dump(response_times_part1, open(os.path.join(folder_path, 'response_times_part1.p'), 'wb'))
    pickle.dump(response_times_part2, open(os.path.join(folder_path, 'response_times_part2.p'), 'wb'))
    pickle.dump(response_times_part3, open(os.path.join(folder_path, 'response_times_part3.p'), 'wb'))
    pickle.dump(response_times_part4, open(os.path.join(folder_path, 'response_times_part4.p'), 'wb'))

    pickle.dump(deltas_0, open(os.path.join(folder_path, 'deltas_0.p'), 'wb'))
    pickle.dump(deltas_1, open(os.path.join(folder_path, 'deltas_1.p'), 'wb'))
    pickle.dump(deltas_2, open(os.path.join(folder_path, 'deltas_2.p'), 'wb'))
    pickle.dump(deltas_3, open(os.path.join(folder_path, 'deltas_3.p'), 'wb'))
    print(f'Results for {num_iterations} over {recordings} recordings for {algorithms} saved in {folder_path}')
