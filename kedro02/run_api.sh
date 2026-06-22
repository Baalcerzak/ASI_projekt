#!/bin/bash

PORT=${1:-8000}
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

echo "=== Car Price Prediction API ==="
echo "Uruchamianie na porcie $PORT..."
echo "Dokumentacja: http://localhost:$PORT/docs"
echo "Health check: http://localhost:$PORT/health"
echo ""

uvicorn kedro02.api.app:app --host 0.0.0.0 --port "$PORT" --reload
