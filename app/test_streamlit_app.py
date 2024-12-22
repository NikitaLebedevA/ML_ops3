import pytest
from unittest.mock import MagicMock
from minio.error import S3Error
import json
import os
import joblib
import pandas as pd
from io import BytesIO

from clearml import Task
import logging
from minio import Minio

# Фикстура для мокирования клиента MinIO
@pytest.fixture
def mock_minio_client():
    # Мокируем методы клиента MinIO
    mock_client = MagicMock()
    
    # Мокируем успешную загрузку данных в MinIO
    mock_client.put_object.return_value = None
    mock_client.get_object.return_value = BytesIO(json.dumps({'model_name': 'LogisticRegression_1.0', 'hyperparameters': {'random_state': 0}}).encode())
    mock_client.list_objects.return_value = []
    mock_client.bucket_exists.return_value = True
    mock_client.make_bucket.return_value = None
    
    return mock_client

def test_add_data_to_storage(mock_minio_client):
    # Переопределяем клиент MinIO на мокированный
    client = mock_minio_client

    # Мокируем данные для загрузки
    dataset_name = "example_dataset"
    dataset_data = [{"feature1": 0.5, "feature2": 0.7, "Target": 1}, {"feature1": 0.2, "feature2": 0.1, "Target": 0}]
    
    # Преобразуем данные в JSON и создаем объект BytesIO
    dataset_json = json.dumps(dataset_data)
    dataset_bytes = BytesIO(dataset_json.encode())

    # Мокируем загрузку данных в хранилище (MinIO)
    mock_minio_client.put_object('datasets', f'{dataset_name}.json', dataset_bytes,
                                  length=len(dataset_json), content_type='application/json')
    
    # Проверяем, что метод put_object был вызван с правильными параметрами
    mock_minio_client.put_object.assert_called_once_with(
        'datasets', f'{dataset_name}.json', dataset_bytes,
        length=len(dataset_json), content_type='application/json'
    )

    # Дополнительно можно проверить, что данные были записаны корректно
    # Для этого можно воссоздать запрос get_object, чтобы убедиться, что данные загружены
    mock_minio_client.get_object.return_value = dataset_bytes
    obj = mock_minio_client.get_object('datasets', f'{dataset_name}.json')
    assert json.loads(obj.read().decode()) == dataset_data  # Проверяем, что данные соответствуют оригинальным

# Тест с реальным S3
@pytest.fixture(scope="module")
def real_minio_client():
    # Для настоящего теста можно использовать реальное подключение к S3 (например, MinIO).
    client = Minio(
        "localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False
    )
    return client

def test_train_model_with_real_s3(real_minio_client):
    # Переопределяем клиент MinIO на реальный
    client = real_minio_client

    model_name = "LogisticRegression"
    model_version = "1.0"
    hyperparameters = '{"random_state": 0}'
    dataset_name = "real_example_dataset"
    target_field = "Target"
    
    # Загружаем модель и данные
    model_info = {
        "model_name": model_name,
        "hyperparameters": json.loads(hyperparameters),
    }
    dataset_data = [{"feature1": 0.5, "feature2": 0.7, "Target": 1}, {"feature1": 0.2, "feature2": 0.1, "Target": 0}]
    
    # Загрузка данных в S3
    dataset_path = f"datasets/{dataset_name}.json"
    model_path = f"models/{model_name}_{model_version}.json"
    
    # Мокируем загрузку данных
    data_stream = BytesIO(json.dumps(dataset_data).encode())
    client.put_object(
        "datasets", dataset_path, data_stream, length=len(json.dumps(dataset_data)),
        content_type="application/json"
    )
    
    # Проверка успешной загрузки данных
    objects = list(client.list_objects("datasets", prefix="datasets/", recursive=True))
    assert any(obj.object_name == dataset_path for obj in objects)
    
    # Моделируем обучение и проверяем успешность
    from sklearn.linear_model import LogisticRegression
    
    # Создаем модель и обучаем ее
    model = LogisticRegression(random_state=0)
    df = pd.DataFrame(dataset_data)
    X = df.drop(columns=[target_field])
    y = df[target_field]
    model.fit(X, y)
    
    # Проверяем, что модель обучена
    assert model.coef_.shape[1] == 2  # Количество признаков в модели

    # Загружаем модель в реальное хранилище
    model_file = f"{model_name}_{model_version}_model.pkl"
    joblib.dump(model, model_file)
    
    client.put_object(
                        "datasets",
                        f'models/{model_file}',
                        data=open(model_file, "rb"),
                        length=os.path.getsize(model_file),
                        content_type="application/octet-stream"
                    )


    # Проверка успешной загрузки модели в S3 (корректируем путь)
    objects = client.list_objects("datasets", prefix="models/", recursive=True)
    print(objects)
    model_list = [obj.object_name for obj in objects]    
    print(model_file)
    print(model_list)
    check_model = f'models/{model_file}'
    assert any(obj == check_model for obj in model_list)

    # Очищаем тестовые данные
    client.remove_object("datasets", dataset_path)
    client.remove_object("datasets", model_file)
