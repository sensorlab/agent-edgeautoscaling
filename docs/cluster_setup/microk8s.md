# MicroK8s Cluster Setup Guide

This guide provides step-by-step instructions for setting up a MicroK8s cluster, 
enabling InPlacePodVerticalScaling, configuring kubeconfig, adding nodes, labeling nodes, and safely shutting down the cluster.

## Prerequisites

- A Linux machine with `snap` installed.
- Basic knowledge of Kubernetes and command-line interface.

## Install MicroK8s

To install MicroK8s, run the following command:

```shell
sudo snap install microk8s --classic --channel=1.28
```

## Enable InPlacePodVerticalScaling

To enable seamless scaling without the need for container restart, activate the feature on 
every node by modifying the Kube API server configuration:

1. Open the Kube API server configuration file:

```shell
sudo nano /var/snap/microk8s/current/args/kube-apiserver
```

2. Add the following line to enable the feature:

```
--feature-gates=InPlacePodVerticalScaling=true
```

3. Restart the MicroK8s cluster for the changes to take effect:

```shell
sudo microk8s stop
sudo microk8s start
```

## Configure Kubeconfig

To allow interaction with the Kube API server, apply the proper authentication:

```shell
microk8s.kubectl config view --raw > ~/.kube/config
```

## Add Nodes

### On Master Node

Generate a join token:

```shell
microk8s add-node
```

If needed, add worker nodes to the `/etc/hosts` file of the master node. Example:

```
ip_addr hostname
```

### On Worker Nodes

Join the worker nodes to the cluster using the generated token:

```shell
microk8s join IP_ADDRESS:25000/TOKEN
```

## Label Nodes

To label nodes, use the following command:

```shell
microk8s kubectl label nodes <your-node-name> foo=bar
```

## Safely Shutdown

To safely shut down the cluster, drain the nodes to evict all pods:

```shell
microk8s kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data --force --grace-period=0
```

Then, stop MicroK8s on each node:

```shell
sudo microk8s stop
```
