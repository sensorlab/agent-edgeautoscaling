Uses the python client for kubernetes, to use it, this command is necessary to load the correct credentials and cluster information for the master node.

- k3s

```shell
sudo cat /etc/rancher/k3s/k3s.yaml > ~/.kube/config
```

- microk8s

```shell
kubectl config view --raw > ~/.kube/config
```
