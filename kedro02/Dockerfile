FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY conf/ ./conf/
COPY data/06_models/ ./data/06_models/
COPY logs/ ./logs/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        kedro>=0.19.0 \
        pandas \
        numpy \
        scikit-learn \
        mlflow \
        pyarrow \
        fastapi \
        "uvicorn[standard]" \
        "pydantic>=2.0"

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "kedro02.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
