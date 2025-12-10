# Change this line to 3.12
FROM python:3.12-slim

WORKDIR /app

# Install build essentials (needed for some vector DB extensions)
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Create directories so permissions are correct when mounting
RUN mkdir -p /app/data /app/chroma_db

ENTRYPOINT ["python", "app.py"]