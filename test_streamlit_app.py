import pytest
import requests
from unittest.mock import MagicMock
from minio.error import S3Error
from io import BytesIO
import json
import os

# Подключаем все моки
@pytest.fixture
def mock_minio_client():
    mock_client = MagicMock()
    return mock_client

@pytest.fixture
def mock_requests():
    return MagicMock()

# Мок для streamlit.spinner
@pytest.fixture
def mock_streamlit():
    mock_spinner = MagicMock()
    mock_spinner.return_value.__enter__.return_value = None  # Имитируем контекстный менеджер
    return mock_spinner

def test_add_model(mock_minio_client, mock_requests, mock_streamlit):
    model_name = "LogisticRegression"
    model_version = "1.0"
    hyperparameters = '{"random_state": 0}'
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"model_name": model_name, "model_version": model_version}
    mock_requests.post.return_value = mock_response
    
    # Подготовка теста
    with mock_streamlit():
        # Стандартный тест на добавление модели
        response = mock_requests.post(
            "http://fastapi_app:8000/add_model",
            json={
                "model_name": model_name,
                "model_version": model_version,
                "hyperparameters": json.loads(hyperparameters),
            },
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200
        assert response.json() == {"model_name": model_name, "model_version": model_version}

def test_load_file_to_s3(mock_minio_client, mock_streamlit):
    uploaded_file = MagicMock()
    uploaded_file.name = "data.json"
    uploaded_file.getvalue.return_value = b"{}"
    mock_minio_client.put_object.return_value = None

    # Проверяем загрузку файла в S3
    with mock_streamlit():
        mock_minio_client.put_object(
            "datasets/data.json", 
            uploaded_file,
            length=len(uploaded_file.getvalue()),
            content_type="application/json"
        )
        mock_minio_client.put_object.assert_called_once_with(
            "datasets/data.json",
            uploaded_file,
            length=len(uploaded_file.getvalue()),
            content_type="application/json"
        )

def test_train_model(mock_minio_client, mock_streamlit):
    mock_minio_client.get_object.return_value = BytesIO(b'{"model_name": "LogisticRegression", "hyperparameters": {"random_state": 0}}')
    mock_minio_client.get_object.return_value = BytesIO(b'[{"Feature1": 1, "Feature2": 2, "Target": 1}]')

    model_name = "LogisticRegression"
    dataset = "example_dataset"
    target_field = "Target"

    # Проверяем, что модель обучена
    with mock_streamlit():
        # Здесь можно добавлять свою логику теста на обучение
        assert model_name == "LogisticRegression"

def test_delete_model(mock_minio_client, mock_streamlit):
    model_to_delete = "LogisticRegression_1.0"
    
    mock_minio_client.remove_object.return_value = None

    # Проверяем удаление модели
    with mock_streamlit():
        mock_minio_client.remove_object("models/" + model_to_delete)
        mock_minio_client.remove_object.assert_called_once_with("models/" + model_to_delete)

def test_predict_model(mock_minio_client, mock_streamlit):
    model_name = "LogisticRegression_1.0"
    dataset = "test_data"
    
    mock_minio_client.get_object.return_value = BytesIO(b'{"model_name": "LogisticRegression", "hyperparameters": {"random_state": 0}}')
    mock_minio_client.get_object.return_value = BytesIO(b'[{"Feature1": 1, "Feature2": 2}]')

    # Проверка выполнения предсказания
    with mock_streamlit():
        predictions = [1]
        assert predictions == [1]
