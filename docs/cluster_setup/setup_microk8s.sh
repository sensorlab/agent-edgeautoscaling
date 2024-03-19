#!/bin/bash

# remove purge microk8s
sudo snap remove --purge microk8s

# install and activate feature gate
sudo snap install microk8s --classic --channel=1.28
sudo echo "--feature-gates=InPlacePodVerticalScaling=true" >> /var/snap/microk8s/current/args/kube-apiserver

# restart microk8s for the gate to take effect
/snap/bin/microk8s stop
/snap/bin/microk8s start
