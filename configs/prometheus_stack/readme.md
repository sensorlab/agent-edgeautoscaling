# Deploy Prometheus + Grafana

```shell
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install [NAME] prometheus-community/kube-prometheus-stack --namespace [NAMESPACE_NAME]
```

# Change configuration values

```shell
helm upgrade [NAME] prometheus-community/kube-prometheus-stack -f values.yaml
```

# Example usage

```shell
microk8s helm install stack prometheus-community/kube-prometheus-stack --namespace metrics
```

```shell
microk8s helm upgrade stack prometheus-community/kube-prometheus-stack -f values.yaml -n metrics
```

```shell
microk8s helm upgrade stack prometheus-community/kube-prometheus-stack -f embeddings_values.yaml -n metrics
```
