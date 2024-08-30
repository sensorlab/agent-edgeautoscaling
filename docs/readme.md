# Resouce Elasticity NANCY

## File Structure

Scripts are meant to run from the project root folder.

## Usage

### Prerequisites

- Disable swap for optimal performance and stability on all Kubernetes nodes:
```bash
sudo swapoff -a
```

#### Feature Gates

- Usage of InPlacePodVerticalScaling feature gate is required for certain functionalities.

#### Python Requirements

- Ensure all Python dependencies are installed. You can use a `requirements.txt` file or a similar method to install dependencies.

#### Tested Environments

- This project has been tested on k3s and microk8s.

## Installation of Cluster

To set up the cluster, run the following script:

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

## Inference

To run the inference script, use the following command:

```bash
python src/infer.py --help
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

Three multi-agent reinforcement learning algorithms are supported:
- DQN
- PPO
- DDPG

## Application Backend

TBD (elasticity_mdqn_module)

## Frontend

For the frontend, refer to the following repository:
- [Resource Elasticity Nancy Visualization](https://github.com/wrathchild14/resource-elastisity-nancy-visualization/)

## Notes

- Ensure all containers are running the same Python version.
- In Kubernetes, it is recommended to disable swap on all nodes to maximize resource utilization. All deployments should have CPU and memory limits set.