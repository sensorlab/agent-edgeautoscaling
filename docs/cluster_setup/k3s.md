# K3s Cluster Setup Guide

This guide provides step-by-step instructions for setting up a K3s cluster, enabling InPlacePodVerticalScaling, 
configuring kubeconfig, adding nodes, labeling nodes, and safely shutting down the cluster.

## Prerequisites

- A Linux machine with `curl` installed.
- Basic knowledge of Kubernetes and command-line interface.

## Install K3s

### Server

To install the K3s server with the InPlacePodVerticalScaling feature enabled, run the following command:

```shell
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.27.1%2Bk3s1 sh -s - --kube-apiserver-arg feature-gates=InPlacePodVerticalScaling=true
```

### Agent

#### Requirements

- IP address of the server host.
- Token from the server: `sudo cat /var/lib/rancher/k3s/server/node-token`

#### Fresh Install

To install the K3s agent and connect it to the server, run the following command:

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="agent --server https://IP_ADDRESS:6443 --token K3S_TOKEN" sh -s -
```

#### Adding Node Labels

Node labels can be added when connecting to the cluster:

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="agent --server https://IP_ADDRESS:6443 --token K3S_TOKEN --node-label foo=bar" sh -s -
```

#### Adding Feature Gate Flag to Kubelet

To add the feature gate flag to the kubelet argument, run the following command:

```shell
curl -sfL https://get.k3s.io | K3S_URL=https://IP_ADDRESS:6443 K3S_TOKEN=K3S_TOKEN INSTALL_K3S_VERSION=v1.27.1%2Bk3s1 sh -s - --kubelet-arg feature-gates=InPlacePodVerticalScaling=true --node-label foo=bar
```

## Miscellaneous

### Important

To use `kubectl` commands, configure the KUBECONFIG environment variable:

```shell
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
sudo chmod 444 /etc/rancher/k3s/k3s.yaml
```

### Run K3s Server

To start the K3s server, run:

```shell
sudo k3s server
```

### Bring Down Cluster

To uninstall K3s and bring down the cluster, run:

```shell
sudo /usr/local/bin/k3s-uninstall.sh
```

## Safely Shutdown

To safely shut down the cluster, drain the nodes to evict all pods:

```shell
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data --force --grace-period=0
```

Then, stop K3s on each node:

```shell
sudo /usr/local/bin/k3s-killall.sh
```
