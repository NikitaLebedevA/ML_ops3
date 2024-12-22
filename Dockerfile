# 1. Базовый образ с установкой зависимостей для MinIO и DVC
FROM python:3.11.7 as dvc_container

# Устанавливаем необходимые зависимости
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем DVC с поддержкой S3
RUN pip install "dvc[s3]"

# 2. Основной контейнер для FastAPI
FROM python:3.11.7 as fastapi_container

# Устанавливаем необходимые зависимости для Poetry
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем проект в контейнер
COPY . .

# Устанавливаем зависимости с помощью Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Устанавливаем точку входа для приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# 3. Контейнер для MinIO
FROM quay.io/minio/minio as minio_base

# Установим только MinIO для контейнера MinIO
# Можете добавить любые дополнительные настройки, которые вам нужны

# Копируем необходимые конфиги для MinIO и запускаем сервис MinIO
ENTRYPOINT ["/bin/sh", "-c", "minio server /data & sleep 10 && mc alias set myminio http://localhost:9000 minioadmin minioadmin && mc mb myminio/datasets || echo 'Bucket already exists' && wait"]
