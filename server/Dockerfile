FROM python:3.12-slim

# Build context must be the repo root: docker build -f server/Dockerfile .
WORKDIR /app

# Install the local mem0 fork (replaces mem0ai from PyPI)
WORKDIR /app/packages
COPY pyproject.toml .
COPY README.md .
COPY mem0 ./mem0
RUN pip install --no-cache-dir .[graph]

# Install remaining server dependencies
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
