FROM python:3.12.7-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 3.prometheus_exporter.py ./prometheus_exporter.py
COPY 7.inference.py ./inference.py
COPY model/ model/

EXPOSE 8000
CMD ["uvicorn", "prometheus_exporter:app", "--host", "0.0.0.0", "--port", "8000"]
