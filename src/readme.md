# Uses the python client for kubernetes, to use it, this command is necessary to load the correct credentials and cluster information for the master node.

- k3s

```shell
sudo cat /etc/rancher/k3s/k3s.yaml > ~/.kube/config
```

- microk8s

```shell
kubectl config view --raw > ~/.kube/config
```

# Debug Elasticity FastAPI application

```shell
uvicorn --app-dir src/ app:elasticity_app --reload
```

## Build it locally

```shell
docker build -t elasticity-app -f src/Dockerfile .
sudo docker run -d -p 8000:8000 elasticity-app
```

## Build an multi-arch image and push to dockerhub

```shell
sudo docker buildx build --platform linux/amd64,linux/arm64 -t <username>/<docker-image-name> -f src/Dockerfile . --push
```

