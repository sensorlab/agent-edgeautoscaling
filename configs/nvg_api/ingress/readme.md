# Deploying Ingress controller

```shell
microk8s helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
microk8s helm repo update
microk8s helm install ingress-nginx ingress-nginx/ingress-nginx
```


# Edit configuraiton

```shell
microk8s kubectl edit deployment ingress-nginx-controller
```