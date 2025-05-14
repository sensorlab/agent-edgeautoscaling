# Use Case: 2 Node Cluster with Snapd through Ethernet

This guide provides step-by-step instructions for setting up a demonstration deployment with one Virtual Machine (VM) as the master (control plane) node and one Raspberry Pi 4 as a worker node. The nodes should be connected to a network to create a Kubernetes cluster.

## Prerequisites

- A Virtual Machine (VM) with a Linux operating system.
- A Raspberry Pi 4 with a Linux operating system.
- Both nodes connected to the same network.
- SSH access to both nodes.
- Basic knowledge of Kubernetes and command-line interface.

## Access to Worker Node

To access the worker node (Raspberry Pi), use the following SSH command:

```sh
ssh pi@raspberrypi.local
```

## Deployment of Demonstration with MicroK8s Kubernetes Cluster

### Setup MicroK8s

1. For every node, run the following script `scripts/microk8s/setup.sh` to install MicroK8s and set up the feature gate. Note that the script uses sudo access and requests to disable swap.

### Deploy Services

2. For every node:
    * On the master node, run `microk8s add-node`.
    * On other (worker) nodes, paste the join command provided by the master node.

3. On the master node, run the following script `scripts/microk8s/deploy_all.sh` to deploy all services:
    * Note: Labeling is used for container scheduling to appropriate nodes. The user will be prompted to enter the substring or whole string of the master node and worker nodes.
      Nodes available can be seen with `microk8s kubectl get nodes`. Worker nodes can be multiple and will be labeled like e.g.:
      rasp1, rasp2, rasp3...
    * This step deploys the whole system including: cAdvisor, Prometheus, Grafana for monitoring, MADRL Elasticity backend for scaling, and LaaS Frontend for visualization.

### Add Grafana Dashboard

4. To add the Grafana dashboard, open Grafana on `<master-ip>:32000`, and log in with the credentials `admin:prom-operator`. Then import the dashboard from: `configs/prometheus_stack/grafana_dashboards/Embedded dashboard-1722329590789.json`.

The demo should run on `<master-ip>:30000` or `localhost:30000`.
