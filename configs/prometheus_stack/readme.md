# Deploy Prometheus + Grafana

```shell
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install [NAME] prometheus-community/kube-prometheus-stack
```

# Change scrape interval

```bash
helm upgrade [NAME] prometheus-community/kube-prometheus-stack -f values.yaml
```
