FROM python:3.10.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/app.py .
COPY src/node.py .
COPY src/utils.py .
COPY src/envs.py .
COPY src/pod_controller.py .
COPY src/infer.py .
COPY src/train_ddpg.py .
COPY src/train_ppo.py .
COPY src/train_mdqn.py .
COPY src/spam_cluster.py .
COPY configs/elasticity_config.yaml configs/elasticity_config.yaml

COPY trained/dqn/mdqn1000ep1000m25inc2_rf_20rps5.0alpha1000res trained/dqn/mdqn1000ep1000m25inc2_rf_20rps5.0alpha1000res
COPY trained/ppo/1000ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources trained/ppo/1000ep_rf_2_20rps10kepochs5alpha10epupdate50scale_a_1000resources

EXPOSE 8000

# COPY kubeconfig.yaml /root/.kube/config
# ENV KUBECONFIG=/root/.kube/config

CMD ["uvicorn", "app:elasticity_app", "--host", "0.0.0.0", "--port", "8000"]