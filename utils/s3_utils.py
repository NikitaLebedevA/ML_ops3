import boto3
from botocore.exceptions import BotoCoreError, ClientError
import os
import json

# Настройки S3
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "datasets")

# Инициализация клиента
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

def upload_data_to_s3(data: dict, filename: str) -> str:
    """
    Загружает данные в бакет S3.
    """
    try:
        json_data = json.dumps(data)
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=json_data,
            ContentType="application/json",
        )
        return f"{S3_ENDPOINT}/{S3_BUCKET_NAME}/{filename}"
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Ошибка при загрузке данных в S3: {e}")


def download_data_from_s3(filename: str) -> dict:
    """
    Загружает данные из S3.
    """
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        data = response["Body"].read().decode("utf-8")
        return json.loads(data)
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Ошибка при загрузке данных из S3: {e}")
