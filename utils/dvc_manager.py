import subprocess
import logging
import os

logging.basicConfig(level=logging.INFO)

class DVCManager:
    def __init__(self, remote_name="myremote"):
        self.remote_name = remote_name
        self.s3_endpoint = os.getenv("S3_ENDPOINT", "http://localhost:9000")
        self.s3_access_key = os.getenv("S3_ACCESS_KEY", "minioadmin")
        self.s3_secret_key = os.getenv("S3_SECRET_KEY", "minioadmin")
        self.s3_bucket = os.getenv("S3_BUCKET", "datasets")

    def add_and_push(self, file_path):
        """Добавить файл в DVC и отправить в удаленный репозиторий."""
        try:
            logging.info(f"Добавление файла {file_path} в DVC...")
            subprocess.run(["dvc", "add", file_path], check=True)
            logging.info(f"Файл {file_path} добавлен в DVC. Отправка в удаленный репозиторий...")
            subprocess.run(["dvc", "push"], check=True)
            logging.info(f"Файл {file_path} успешно отправлен в remote.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Ошибка работы с DVC: {e}")
            raise

    def pull(self, file_path):
        """Скачать файл из DVC remote."""
        try:
            logging.info(f"Скачивание файла {file_path} из DVC...")
            subprocess.run(["dvc", "pull", file_path], check=True)
            logging.info(f"Файл {file_path} успешно скачан.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Ошибка скачивания файла через DVC: {e}")
            raise

    def status(self):
        """Получить статус DVC."""
        try:
            logging.info("Получение статуса DVC...")
            result = subprocess.run(["dvc", "status"], check=True, capture_output=True, text=True)
            logging.info("Статус DVC получен.")
            return result.stdout
        except subprocess.CalledProcessError as e:
            logging.error(f"Ошибка получения статуса DVC: {e}")
            return "Error"

    def set_remote(self):
        """Настройка Minio как удаленного хранилища."""
        try:
            remote_url = f"s3://{self.s3_bucket}"
            logging.info(f"Настройка удаленного репозитория {self.remote_name} с URL: {remote_url}...")
            
            # Добавление удаленного хранилища
            subprocess.run(["dvc", "remote", "add", "--default", self.remote_name, remote_url], check=True)
            
            # Модификация параметров подключения
            subprocess.run(["dvc", "remote", "modify", self.remote_name, "endpointurl", self.s3_endpoint], check=True)
            subprocess.run(["dvc", "remote", "modify", self.remote_name, "access_key_id", self.s3_access_key], check=True)
            subprocess.run(["dvc", "remote", "modify", self.remote_name, "secret_access_key", self.s3_secret_key], check=True)
            
            logging.info(f"Удаленный репозиторий {self.remote_name} настроен и готов к использованию.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Ошибка настройки удаленного репозитория: {e}")
            raise
