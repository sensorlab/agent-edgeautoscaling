# Multi Agent Reinforcement Learning-based In-place Scaling Engine (MARLISE)

In this project, the scaling approach involves deploying an agent per pod, which is a vertical 
scaling strategy. This method is particularly well-suited for stateful applications. 
The system is highly scalable because we can add an agent for every pod, and the interval of 
scaling can be as low as 1 second, allowing for rapid adjustments to resource demands.

## File Structure

Scripts are meant to run from the project root folder.

## Usage

### Prerequisites

#### Feature Gates

- Usage of InPlacePodVerticalScaling feature gate is required for agent functionalities to ensure seamless scaling.

Example:
```
kubectl patch pod localization-api1 --patch '{"spec":{"containers":[{"name":"localization-api", "resources":{"requests":{"cpu":"500m"}, "limits":{"cpu":"500m"}}}]}}'
```

#### Python Requirements

- Ensure all Python dependencies are installed. You can use a `requirements.txt` file or a similar method to install dependencies.

#### Tested Environments

- This project has been tested on k3s and microk8s.

## Installation of Cluster

To install the `microk8s` container orchestration platform, run the following script on every node:

```bash
sudo bash scripts/microk8s/setup.sh
```

After running the script, join the nodes using:

```bash
microk8s add-node
```

## Deployments

Deploy the necessary configurations and the use-case service of localization:

```bash
sudo bash scripts/microk8s/deploy_all.sh
```

More info in [demo readme](/docs/demo_setup.md).

## Inference

To run the inference script, use the following command:

```bash
python src/infer.py
```

### Inference Script Usage

```plaintext
usage: infer.py [-h] [--n_agents N_AGENTS] [--resources RESOURCES] [--load_model LOAD_MODEL] [--action_interval ACTION_INTERVAL] [--priorities PRIORITIES [PRIORITIES ...]] [--algorithm ALGORITHM] [--hack HACK] [--debug]

options:
  -h, --help            show this help message and exit
  --n_agents N_AGENTS
  --resources RESOURCES
  --load_model LOAD_MODEL
  --action_interval ACTION_INTERVAL
  --priorities PRIORITIES [PRIORITIES ...]
                        List of priorities (0.0 < value <= 1.0), default is 1.0 for all agents. Example: 1.0 1.0 1.0
  --algorithm ALGORITHM
                        Algorithm to use: ppo, ippo (instant ppo), dppo (discrete ppo), ddpg, iddpg (instant ddpg), mdqn, dmdqn, ddmdqn
  --hack HACK           Transfer learning agent, so every agent will be loaded from this agent's saved weights
  --debug
```

## Training

Three multi-agent deep reinforcement learning algorithms are supported:
- DQN: Actions are {increase, maintain, decrease}
- PPO: Outputs a continuous number [-1, 1], scaled and applied. Can also use discrete actions.
- DDPG: Similar to PPO, outputs a continuous number.

## Application Backend
Avaialbe as a [Docker image](https://hub.docker.com/repository/docker/wrathchild14/elasticity/general)
or run locally, refer to [other readme](/src/readme.md).

The Backend handles the elasticity by itself, but offers API intreface for control and information about the system.

### API Endpoints that are supported
| Endpoint                | Method | Description                                      | Request Body                                                                 | Response                                                                 |
|-------------------------|--------|--------------------------------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `/start`                | POST   | Starts the inference process                     | None                                                                         | `{ "message": "Inference started" }`                                     |
| `/stop`                 | POST   | Stops the inference process                      | None                                                                         | `{ "message": "Inference stopped" }`                                     |
| `/set_resources`        | POST   | Sets the maximum CPU resources                   | `{ "resources": int }`                                                       | `{ "message": "Resources set to {resources}" }`                          |
| `/set_interval`         | POST   | Sets the interval between actions                | `{ "interval": int }`                                                        | `{ "message": "Interval set to {interval}" }`                            |
| `/set_dqn_algorithm`    | POST   | Sets the algorithm to DQN                        | None                                                                         | `{ "message": "DQN algorithm set" }`                                     |
| `/set_ppo_algorithm`    | POST   | Sets the algorithm to PPO                        | None                                                                         | `{ "message": "PPO algorithm set" }`                                     |
| `/set_ddpg_algorithm`   | POST   | Sets the algorithm to DDPG                       | None                                                                         | `{ "message": "DDPG algorithm set" }`                                    |
| `/set_default_limits`   | POST   | Sets the default CPU limits                      | None                                                                         | `{ "message": "Default limits set" }`                                    |
| `/status`               | GET    | Gets the status of the inference process         | None                                                                         | `{ "status": bool }`                                                     |
| `/algorithm`            | GET    | Gets the current algorithm being used            | None                                                                         | `{ "algorithm": "ppo" OR "dqn" OR "ddpg" }`                              |
| `/resources`            | GET    | Gets the current maximum CPU resources           | None                                                                         | `{ "resources": int }`                                                   |
| `/interval`             | GET    | Gets the current action interval                 | None                                                                         | `{ "interval": int }`                                                    |


## Frontend

For the frontend that is connected to the above aforementioned Backend, refer to the following [repository](https://github.com/wrathchild14/resource-elastisity-nancy-visualization/).

## Notes

- Ensure all containers are running the same Python version.
- In Kubernetes, it is recommended to disable swap with `sudo swapoff -a` on all nodes to maximize 
resource utilization. All deployments should have CPU and memory limits set.
