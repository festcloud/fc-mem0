FROM python:3.12-slim

WORKDIR /app

COPY server/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install local mem0 from repo source (pyproject.toml is in repo root, mem0/ is the package)
COPY pyproject.toml /tmp/mem0_source/
COPY README.md /tmp/mem0_source/
COPY mem0 /tmp/mem0_source/mem0
RUN pip install --no-cache-dir "/tmp/mem0_source[graph]"

COPY server .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
