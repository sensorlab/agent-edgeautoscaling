Run with `Python 3.10.13` and `Ray 3.9.0`. Python version is important, as all containers running have to be the same

In Kubernetes, it is recommended to disable swap on all nodes. The goal of Kubernetes is to maximize resource utilization, so all deployments should have CPU and memory limits set.
```bash
sudo swapoff -a
```
