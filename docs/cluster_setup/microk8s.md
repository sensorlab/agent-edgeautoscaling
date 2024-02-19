# Install

```shell
sudo snap install microk8s --classic --channel=1.27
```

## InPlacePodVerticalScaling

- on every node where microk8s is installed in `/var/snap/microk8s/current/args/kube-apiserver` add:

```
--feature-gates=InPlacePodVerticalScaling=true
```

- restart the cluster

```shell
sudo microk8s stop
sudo microk8s start
```

## kube config

```shell
microk8s.kubectl config view --raw > ~/.kube/config
```

## Add nodes

- on master node
```shell
microk8s add-node
```

worker nodes should be added into `/etc/hotsts` of master. Example:
```
ip_addr hostname
```

- other nodes:
```shell
microk8s join IP_ADDRESS:25000/TOKEN
```

### Label nodes
```shell
microk8s kubectl label nodes <your-node-name> foo=bar
```

# Safely shutdown

drain nodes to evict all pods
```shell
microk8s kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data --force --grace-period=0
```

on each node
```shell
sudo microk8s stop
```
