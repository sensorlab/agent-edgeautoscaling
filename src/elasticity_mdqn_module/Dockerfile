FROM python:3.10.13

WORKDIR /app

ADD . /app

RUN pip install numpy fastapi uvicorn pandas torch torhcvision gymnasium kubernetes

EXPOSE 8000

CMD ["uvicorn", "elasticity:app", "--host", "0.0.0.0", "--port", "8001", "--limit-concurrency", "4000", "--log-level", "critical", "--backlog", "4000"]

# sudo docker buildx build --platform linux/amd64,linux/arm64 -t [image] . --push
