# Back-End configuration setup

The application uses the Python client for Kubernetes and to use it, the correct credentials and cluster information for the master node have to be loaded.

### k3s

```bash
sudo cat /etc/rancher/k3s/k3s.yaml > ~/.kube/config
```

### microk8s

```bash
kubectl config view --raw > ~/.kube/config
```

# Running the Elasticity FastAPI Application Locally

To run the Elasticity FastAPI application locally for debug purposes, use the following command:

```bash
uvicorn --app-dir src app:elasticity_app --reload
```

# Building the Application Locally with Docker

To build the application locally using Docker, follow these steps:

1. Build the Docker image:
    ```bash
    docker build -t elasticity-app -f src/Dockerfile .
    ```

2. Run the Docker container:
    ```bash
    sudo docker run -d -p 8000:8000 elasticity-app
    ```

# Building a Multi-Arch Image and Pushing to DockerHub

To build a multi-architecture Docker image and push it to DockerHub, use the following command:

```bash
sudo docker buildx build --platform linux/amd64,linux/arm64 -t <username>/<docker-image-name> -f src/Dockerfile . --push
```

Replace `<username>` and `<docker-image-name>` with your DockerHub username and the desired image name, respectively.

---
