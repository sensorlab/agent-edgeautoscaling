Uses the python client for kubernetes, to use

- k3s

```shell
sudo cat /etc/rancher/k3s/k3s.yaml > ~/.kube/config
```

- microk8s

```shell
kubectl config view --raw > ~/.kube/config
```
