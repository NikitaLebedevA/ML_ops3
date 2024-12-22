from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Dict, List
from model_manager import Models
from schemas import AddModelRequest, TrainingRequest, PredictionRequest, DropModelRequest
from dotenv import load_dotenv
import os

# Инициализация загрузки переменных окружения
load_dotenv("utils/s3_env.env")

# Проверка конфигурации MinIO
S3_BUCKET_NAME = os.getenv("S3_BUCKET")
if not S3_BUCKET_NAME:
    raise RuntimeError("MinIO bucket name is not configured. Check environment variables.")

# Инициализация менеджера моделей
model_manager = Models()


def register_endpoints(app: FastAPI):
    """
    Регистрирует все конечные точки в приложении FastAPI.
    """

    @app.get("/health", summary="Check API Health")
    def health_check() -> Dict[str, str]:
        """
        Проверяет состояние API.
        """
        return {"status": "OK"}

    @app.post("/add_model", summary="Add a new ML model")
    def add_model(request: AddModelRequest):
        """
        Добавляет новую модель машинного обучения.
        """
        try:
            model_manager.add_model(request.model_name, request.hyperparameters)
            config_path = f"/tmp/{request.model_name}_config.json"
            with open(config_path, "w") as config_file:
                config_file.write(str({"model_name": request.model_name, "hyperparameters": request.hyperparameters}))

            model_manager.upload_to_minio(config_path, f"configs/{request.model_name}_config.json")
            os.remove(config_path)

            return {"message": f"Model {request.model_name} added successfully."}
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/train", summary="Train a ML model")
    def train_model(request: TrainingRequest):
        """
        Обучает модель машинного обучения.
        """
        try:
            dataset_path = f"/tmp/{request.model_name}_data.pkl"
            model_manager.download_from_minio(f"datasets/{request.model_name}_data.pkl", dataset_path)

            with open(dataset_path, "rb") as f:
                training_data = pickle.load(f)
            os.remove(dataset_path)

            X_train, y_train = training_data["X_train"], training_data["y_train"]
            model_manager.train(request.model_name, X_train, y_train)

            return {"message": f"Model {request.model_name} trained successfully."}
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @app.post("/predict", summary="Get predictions from a ML model")
    def predict(request: PredictionRequest):
        """
        Получает предсказания от обученной модели.
        """
        try:
            predictions = model_manager.predict(request.model_id, request.input_data)
            return {"predictions": predictions.tolist()}
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @app.post("/drop_model", summary="Remove a ML model")
    def drop_model(request: DropModelRequest):
        """
        Удаляет модель из менеджера.
        """
        try:
            model_manager.models.pop(request.model_id, None)
            return {"message": f"Model {request.model_id} removed successfully."}
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/upload_to_minio", summary="Upload file to MinIO")
    async def upload_file_to_minio(file: UploadFile = File(...)):
        """
        Загружает файл в MinIO.
        """
        try:
            file_path = f"/tmp/{file.filename}"
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            model_manager.upload_to_minio(file_path, f"uploads/{file.filename}")
            os.remove(file_path)

            return {"message": f"File {file.filename} uploaded to MinIO successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MinIO error: {str(e)}")

    @app.get("/list_minio_files", summary="List all files in MinIO bucket")
    def list_minio_files():
        """
        Получает список файлов из хранилища MinIO.
        """
        try:
            objects = model_manager.client.list_objects(S3_BUCKET_NAME)
            files = [obj.object_name for obj in objects]
            return {"files": files}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")
