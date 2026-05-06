FROM python:3.12-slim

WORKDIR /app

# 1. Встановлюємо всі залежності (включно з mem0ai з PyPI)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Правильно затягуємо локальний форк через додатковий контекст
COPY --from=mem0_local . /app/mem0

# 3. Зносимо PyPI-версію і ставимо твій форк у режимі editable
RUN pip uninstall -y mem0ai && pip install --no-cache-dir -e /app/mem0

# 4. Копіюємо код сервера
COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
