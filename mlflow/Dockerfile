FROM python:3.7-slim-buster

# Install python packages
RUN pip install mlflow

# Install curl for healthcheck
RUN apt update && apt install -y curl && rm -rf /var/lib/apt/lists/*

CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "sqlite:////opt/mlflow/database/mlruns.db", "--default-artifact-root", "/opt/better_categories_4all/data/mlruns"]
