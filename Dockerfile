FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-api-constraints.txt ./
RUN pip install --no-cache-dir cython setuptools wheel && pip install --no-cache-dir --no-build-isolation -r requirements.txt -c requirements-api-constraints.txt
COPY config.py api_service.py ./
COPY alignment ./alignment
COPY database ./database
COPY detection ./detection
COPY enrollment ./enrollment
COPY recognition ./recognition
COPY scripts ./scripts
ENV PYTHONUNBUFFERED=1
EXPOSE 5000
CMD ["python", "-m", "uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "5000"]
