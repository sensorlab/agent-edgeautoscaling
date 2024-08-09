FROM python:3.10.13

WORKDIR /app

COPY regressor.py /app
COPY spring.pkl /app
COPY trained_model.pkl /app

RUN pip install numpy fastapi uvicorn scikit-learn joblib pandas

EXPOSE 8000

CMD ["uvicorn", "regressor:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--limit-concurrency", "4000", "--log-level", "critical", "--backlog", "4000"]

# sudo docker buildx build --platform linux/amd64,linux/arm64 -t [image] . --push
