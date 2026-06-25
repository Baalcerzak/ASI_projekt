FROM python:3.10-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        kedro>=0.19.0 \
        pandas \
        numpy \
        scikit-learn \
        mlflow \
        pyarrow \
        "pycaret==3.3.2" \
        fastapi \
        "uvicorn[standard]" \
        "pydantic>=2.0"

COPY src/ ./src/
RUN pip install --no-cache-dir .

FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY conf/ ./conf/
COPY data/06_models/ ./data/06_models/

RUN mkdir -p /app/logs && chown -R 1000:1000 /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd -u 1000 -m appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "kedro02.api.app:app", "--host", "0.0.0.0", "--port", "8000"]