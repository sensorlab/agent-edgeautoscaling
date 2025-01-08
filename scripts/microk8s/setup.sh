#!/bin/bash

set -e

log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1"
}

if [ "$EUID" -ne 0 ]; then
    log "Please run as root or use sudo"
    exit 1
fi

log "Removing existing microk8s installation"
sudo snap remove --purge microk8s

read -p "Do you want to disable swap? (yes/no): " disable_swap
if [ "$disable_swap" == "yes" ]; then
    log "Disabling swap"
    sudo swapoff -a
else
    log "Skipping swap disable"
fi

log "Installing microk8s"
sudo snap install microk8s --classic --channel=1.28

log "Activating feature gate for InPlacePodVerticalScaling"
echo "--feature-gates=InPlacePodVerticalScaling=true" | sudo tee -a /var/snap/microk8s/current/args/kube-apiserver

log "Restarting microk8s to apply changes"
/snap/bin/microk8s stop
/snap/bin/microk8s start

log "microk8s setup completed successfully"
