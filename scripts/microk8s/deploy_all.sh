#!/bin/bash

set -e
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1"
}

log "Setting up Kubernetes configuration"
microk8s kubectl config view --raw > ~/.kube/config

log "Labeling nodes for cluster identification"
microk8s kubectl label nodes raspberrypi1 cluster=rasp1 --overwrite
microk8s kubectl label nodes raspberrypi2 cluster=rasp2 --overwrite
microk8s kubectl label nodes e6-orancloud cluster=vm --overwrite

log "Deploying cAdvisor for container monitoring"
microk8s kubectl apply -f configs/cadvisor/config.yaml

log "Installing NGINX Ingress Controller using Helm"
microk8s helm repo update
microk8s helm install ingress-nginx ingress-nginx/ingress-nginx

log "Creating a namespace for metrics"
microk8s kubectl create namespace metrics || true

log "Installing Prometheus stack for monitoring"
microk8s helm install stack prometheus-community/kube-prometheus-stack --namespace metrics

log "Upgrading Prometheus stack with custom values"
microk8s helm upgrade stack prometheus-community/kube-prometheus-stack -f configs/prometheus_stack/values.yaml -n metrics
microk8s helm upgrade stack prometheus-community/kube-prometheus-stack -f configs/prometheus_stack/embeddings_values.yaml -n metrics

log "Deploying localization services"
microk8s kubectl apply -f configs/localization/separate_services.yaml

log "Deployment completed successfully"
