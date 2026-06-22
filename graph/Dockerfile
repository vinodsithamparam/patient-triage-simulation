FROM python:3.12-slim

WORKDIR /app

# Install system utilities required for compiling some database adapters if necessary
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY preprocessor.joblib .
COPY xgb_model.joblib .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8123

# Start the native server instance
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8123", "--allow-blocking"]