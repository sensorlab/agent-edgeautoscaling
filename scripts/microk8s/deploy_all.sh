#!/bin/bash

set -e
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1"
}

log "Setting up Kubernetes configuration"
microk8s config > ~/.kube/config

# log "Labeling nodes for cluster identification"
# microk8s kubectl label nodes raspberrypi1 cluster=rasp1 --overwrite
# microk8s kubectl label nodes raspberrypi2 cluster=rasp2 --overwrite
# microk8s kubectl label nodes e6-orancloud cluster=vm --overwrite

log "Labeling nodes for cluster identification, NOTE: Looking for substrings of the node names"

read -p "INPUT the master node name or a substring: " master_node
read -p "INPUT the substring of the edge nodes: " edge_nodes
i=0
nodes=$(microk8s kubectl get nodes -o jsonpath='{.items[*].metadata.name}')
for node in $nodes; do
    case $node in
        $edge_nodes*)
            microk8s kubectl label nodes $node cluster=rasp$((++i)) --overwrite
            log "Labeling $node with cluster=rasp$i"
            ;;
        $master_node*)
            microk8s kubectl label nodes $node cluster=vm --overwrite
            log "Labeling $node with cluster=vm"
            ;;
        *)
            log "Unknown node: $node"
            ;;
    esac
done

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
