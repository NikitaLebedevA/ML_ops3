import boto3
import os
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

logging.basicConfig(level=logging.INFO)

class S3Client:
    def __init__(self, endpoint_url=None, access_key=None, secret_key=None, bucket_name=None):
        """
        Инициализация клиента S3.
        """
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        self.access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")

        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            raise RuntimeError("S3 configuration is incomplete. Check environment variables.")

        try:
            self.s3 = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            )
            logging.info("Успешное подключение к S3.")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logging.error(f"Ошибка авторизации S3: {e}")
            raise

    def upload_file(self, file_path, object_name=None):
        """
        Загрузка файла в S3.
        """
        try:
            if object_name is None:
                object_name = os.path.basename(file_path)

            if not self.check_bucket():
                self.create_bucket()

            self.s3.upload_file(file_path, self.bucket_name, object_name)
            logging.info(f"Файл {file_path} успешно загружен в {self.bucket_name}/{object_name}.")
        except ClientError as e:
            logging.error(f"S3 ошибка при загрузке файла {file_path}: {e}")
            raise
        except Exception as e:
            logging.error(f"Неизвестная ошибка при загрузке файла {file_path}: {e}")
            raise

    def download_file(self, object_name, file_path):
        """
        Загрузка файла из S3.
        """
        try:
            self.s3.download_file(self.bucket_name, object_name, file_path)
            logging.info(f"Файл {object_name} из {self.bucket_name} успешно скачан в {file_path}.")
        except ClientError as e:
            logging.error(f"S3 ошибка при загрузке файла {object_name}: {e}")
            raise
        except Exception as e:
            logging.error(f"Неизвестная ошибка при загрузке файла {object_name}: {e}")
            raise

    def list_objects(self):
        """
        Список объектов в S3 bucket.
        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name)
            objects = response.get("Contents", [])
            logging.info(f"Найдено {len(objects)} объектов в bucket {self.bucket_name}.")
            return [obj["Key"] for obj in objects]  # Возвращаем только ключи
        except ClientError as e:
            logging.error(f"S3 ошибка при получении списка объектов: {e}")
            return []
        except Exception as e:
            logging.error(f"Неизвестная ошибка при получении списка объектов: {e}")
            return []

    def delete_object(self, object_name):
        """
        Удалить объект из S3.
        """
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=object_name)
            logging.info(f"Объект {object_name} удалён из {self.bucket_name}.")
        except ClientError as e:
            logging.error(f"S3 ошибка при удалении объекта {object_name}: {e}")
            raise
        except Exception as e:
            logging.error(f"Неизвестная ошибка при удалении объекта {object_name}: {e}")
            raise

    def check_bucket(self):
        """
        Проверить существование bucket.
        """
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logging.info(f"Bucket {self.bucket_name} доступен.")
            return True
        except ClientError as e:
            logging.warning(f"Bucket {self.bucket_name} недоступен: {e}")
            return False

    def create_bucket(self):
        """
        Создать bucket, если он отсутствует.
        """
        try:
            self.s3.create_bucket(Bucket=self.bucket_name)
            logging.info(f"Bucket {self.bucket_name} успешно создан.")
        except ClientError as e:
            logging.error(f"S3 ошибка при создании bucket {self.bucket_name}: {e}")
            raise
