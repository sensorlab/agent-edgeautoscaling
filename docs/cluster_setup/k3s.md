# Install

## server

```shell
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.27.1%2Bk3s1 sh -s - --kube-apiserver-arg feature-gates=InPlacePodVerticalScaling=true
```

## agent

Requirements

- IP address of host
- token: `sudo cat /var/lib/rancher/k3s/server/node-token`

### fresh install

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="agent --server https://IP_ADDRESS:6443 --token K3S_TOKEN" sh -s -
```

- node labels can be added when connecting to the cluster

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="agent --server https://IP_ADDRESS:6443 --token K3S_TOKEN --node-label foo=bar" sh -s -
```

- add feature gate flag to kubelet-arg

```shell
curl -sfL https://get.k3s.io | K3S_URL=https://IP_ADDRESS:6443 K3S_TOKEN=K3S_TOKEN INSTALL_K3S_VERSION=v1.27.1%2Bk3s1 sh -s - --kubelet-arg feature-gates=InPlacePodVerticalScaling=true --node-label foo=bar
```

# Misc

## Important

- use `kubectl` commands

```shell
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
sudo chmod 444 /etc/rancher/k3s/k3s.yaml
```

## run:

```shell
sudo k3s server
```

## bring down cluster

```shell
sudo /usr/local/bin/k3s-uninstall.sh
```
