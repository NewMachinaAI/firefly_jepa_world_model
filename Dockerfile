FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY container_harness_src/ container_harness_src/
COPY container_encoder_predictor_train_src/ container_encoder_predictor_train_src/
COPY container_encoder_inference_src/ container_encoder_inference_src/
COPY container_predicter_inference_src/ container_predicter_inference_src/
COPY container_policy_inference_src/ container_policy_inference_src/

# Training data and model weights live in ./data, mounted at /app/data by docker-compose
ENTRYPOINT ["python"]
CMD ["container_harness_src/firefly_harness.py"]
