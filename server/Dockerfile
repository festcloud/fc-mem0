FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mem0_package /tmp/mem0_package
RUN pip install --no-cache-dir --force-reinstall --no-deps "/tmp/mem0_package" \
    && rm -rf /app/mem0

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
