FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY mem0 ./mem0
COPY server/requirements.txt ./server/requirements.txt

WORKDIR /app/server
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
