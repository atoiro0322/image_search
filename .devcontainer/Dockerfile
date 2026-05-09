FROM mcr.microsoft.com/devcontainers/python:3.11

# chromadb が内部で使う sqlite3 の新しいバージョンを確保
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt
