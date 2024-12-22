import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from clearml import Task, Logger
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

# Инициализация загрузки переменных окружения
load_dotenv("utils/s3_env.env")

# Конфигурация MinIO
S3_ENDPOINT = "minio:9000"
S3_BUCKET = os.getenv("S3_BUCKET", "datasets")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")

# Настраиваем клиента MinIO
client = Minio(
    S3_ENDPOINT,
    access_key=S3_ACCESS_KEY,
    secret_key=S3_SECRET_KEY,
    secure=False,
)

# Словарь доступных моделей
available_models = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
}


class Models:
    def __init__(self, task_name="Model Training"):
        self.models = {}
        self.task = Task.init(project_name="ML Models", task_name=task_name)

    def add_model(self, model_name: str, hyperparameters: dict, version: str = "1.0"):
        """Добавить модель с указанными гиперпараметрами."""
        if model_name not in available_models:
            raise KeyError(f"Model {model_name} is not available.")
        
        model_class = available_models[model_name]
        model = model_class(**hyperparameters)
        model_versioned_name = f"{model_name}_v{version}"
        
        self.models[model_versioned_name] = model
        Logger.current_logger().report_text(
            f"Added model {model_versioned_name} with hyperparameters: {hyperparameters}"
        )

    def train(self, model_name: str, X_train, y_train):
        """Обучение указанной модели."""
        if model_name not in self.models:
            raise KeyError(f"Model {model_name} does not exist.")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        Logger.current_logger().report_text(f"Training completed for model {model_name}")
        Logger.current_logger().report_scalar("Training", "Model", model_name)
        self.task.flush()

    def predict(self, model_name: str, X_test):
        """Получение предсказаний от модели."""
        if model_name not in self.models:
            raise KeyError(f"Model {model_name} does not exist.")
        
        predictions = self.models[model_name].predict(X_test)
        Logger.current_logger().report_text(f"Predictions made for model {model_name}")
        return predictions

    def save_model(self, model_name: str, version: str = "1.0"):
        """Сохранение модели в MinIO."""
        model_versioned_name = f"{model_name}_v{version}"
        if model_versioned_name not in self.models:
            raise KeyError(f"Model {model_versioned_name} does not exist.")
        
        save_path = f"{model_versioned_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.models[model_versioned_name], f)
        
        self.upload_to_minio(save_path, f"models/{save_path}")
        os.remove(save_path)
        Logger.current_logger().report_text(f"Model {model_versioned_name} saved to MinIO")

    def load_model(self, model_name: str, version: str = "1.0"):
        """Загрузка модели из MinIO."""
        model_versioned_name = f"{model_name}_v{version}"
        model_path = f"models/{model_versioned_name}.pkl"

        temp_path = f"/tmp/{model_versioned_name}.pkl"
        self.download_from_minio(model_path, temp_path)

        with open(temp_path, "rb") as f:
            self.models[model_versioned_name] = pickle.load(f)
        os.remove(temp_path)
        Logger.current_logger().report_text(f"Model {model_versioned_name} loaded from MinIO")

    def save_dataset(self, dataset_path: str):
        """Сохранение датасета в MinIO."""
        object_name = f"datasets/{os.path.basename(dataset_path)}"
        self.upload_to_minio(dataset_path, object_name)
        Logger.current_logger().report_text(f"Dataset {dataset_path} uploaded to MinIO")

    def download_dataset(self, object_name: str, download_path: str):
        """Загрузка датасета из MinIO."""
        self.download_from_minio(object_name, download_path)
        Logger.current_logger().report_text(f"Dataset {object_name} downloaded from MinIO to {download_path}")

    def upload_to_minio(self, file_path: str, object_name: str):
        """Загрузка файла в MinIO."""
        try:
            with open(file_path, "rb") as file_data:
                client.put_object(S3_BUCKET, object_name, file_data, length=os.stat(file_path).st_size)
            Logger.current_logger().report_text(f"Uploaded {file_path} to MinIO bucket {S3_BUCKET}")
        except S3Error as e:
            Logger.current_logger().report_text(f"Failed to upload {file_path} to MinIO: {str(e)}")

    def download_from_minio(self, object_name: str, download_path: str):
        """Загрузка файла из MinIO."""
        try:
            client.fget_object(S3_BUCKET, object_name, download_path)
            Logger.current_logger().report_text(f"Downloaded {object_name} from MinIO to {download_path}")
        except S3Error as e:
            Logger.current_logger().report_text(f"Failed to download {object_name} from MinIO: {str(e)}")
