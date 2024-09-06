#!/bin/bash

set -e
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1"
}

log "Setting up Kubernetes configuration"
microk8s config > ~/.kube/config

log "Labeling nodes for cluster identification"
microk8s kubectl label nodes raspberrypi1 cluster=rasp1 --overwrite
microk8s kubectl label nodes raspberrypi2 cluster=rasp2 --overwrite
microk8s kubectl label nodes e6-orancloud cluster=vm --overwrite

log "Deploying cAdvisor for container monitoring"
microk8s kubectl apply -f configs/cadvisor/config.yaml

log "Adding Helm repositories"
microk8s helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
microk8s helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
microk8s helm repo update

log "Installing NGINX Ingress Controller using Helm"
microk8s helm install ingress-nginx ingress-nginx/ingress-nginx

log "Creating a namespace for metrics"
microk8s kubectl create namespace metrics || true

log "Installing Prometheus stack for monitoring"
microk8s helm install stack prometheus-community/kube-prometheus-stack --namespace metrics --values configs/prometheus_stack/values.yaml


log "Deploying localization services"
microk8s kubectl apply -f configs/localization/separate_services.yaml

log "Deployment completed successfully"
