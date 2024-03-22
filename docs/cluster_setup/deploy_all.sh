#!/bin/bash

cd ../../configs

microk8s.kubectl config view --raw > ~/.kube/config

microk8s kubectl label nodes raspberrypi1 cluster=rasp1
microk8s kubectl label nodes raspberrypi2 cluster=rasp2
microk8s kubectl label nodes e6-orancloud cluster=vm

microk8s kubectl apply -f cadvisor/config.yaml

microk8s helm install ingress-nginx ingress-nginx/ingress-nginx

microk8s kubectl create namespace metrics

microk8s helm install stack prometheus-community/kube-prometheus-stack --namespace metrics

microk8s helm upgrade stack prometheus-community/kube-prometheus-stack -f prometheus_stack/values.yaml -n metrics
microk8s helm upgrade stack prometheus-community/kube-prometheus-stack -f prometheus_stack/embeddings_values.yaml -n metrics

microk8s kubectl apply -f localization/config.yaml
