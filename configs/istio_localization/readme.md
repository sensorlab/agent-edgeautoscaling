<!-- # Istio service mesh
- Used for weighted service routing
```shell
microk8s helm repo add istio https://istio-release.storage.googleapis.com/charts
microk8s helm repo update
microk8s helm install istio-base istio/base -n istio-system --create-namespace
microk8s helm install istiod istio/istiod -n istio-system
microk8s helm install istio-ingress istio/gateway -n istio-system
```

- Optional
```shell
microk8s helm install istio-egress istio/gateway -n istio-system
``` -->

# Install

- if needed enable `microk8s enable community`, then run command:

```shell
microk8s enable istio
```

# Setup

- with `microk8s kubectl get svc -n istio-system` get the port from the `istio-ingressgateway` to use for the requests.
