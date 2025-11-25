FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY retrofitkit ./retrofitkit
COPY config ./config
COPY recipes ./recipes
COPY docs ./docs

ENV P4_CONFIG=/app/config/config.yaml

EXPOSE 8000
CMD ["uvicorn", "retrofitkit.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
