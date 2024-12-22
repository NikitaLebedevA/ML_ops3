import streamlit as st
from clearml import Task
import json
import logging
from minio import Minio
from minio.error import S3Error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import requests
import pickle
import mimetypes
import os
import joblib
import pandas as pd
from io import BytesIO

# Настройка логгера
logging.basicConfig(
    filename="streamlit_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

task = Task.init(project_name="ML_Masters", task_name="train_model", task_type=Task.TaskTypes.optimizer)

# Настройки MinIO
MINIO_URL = "minio:9000"  # URL MinIO
MINIO_ACCESS_KEY = "minioadmin"  # Access Key для MinIO
MINIO_SECRET_KEY = "minioadmin"  # Secret Key для MinIO
MINIO_BUCKET_NAME = "datasets"  # Имя bucket в MinIO

# Создаём клиента MinIO
client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

API_URL = "http://fastapi_app:8000"

st.set_page_config(page_title="ML Model Manager", layout="wide")
st.title("ML Model Manager")

tabs = st.tabs(["Работа с моделями", "Работа с данными", "Обучить модель", "Предсказание", "Удаление данных", "Состояние сервера"])

# Вкладка: Добавить модель
with tabs[0]:
    st.header("Добавить модель: LogisticRegression или RandomForestClassifier")
    model_name = st.text_input("Название модели", value="LogisticRegression")
    model_version = st.text_input("Версия модели", value="1.0")
    hyperparameters = st.text_area("Гиперпараметры (в формате JSON)", value='{"random_state": 0}')
    
    if st.button("Добавить модель", key="add_model"):
        with st.spinner("Добавление модели..."):
            try:
                # Отправка запроса на добавление модели
                response = requests.post(
                    f"{API_URL}/add_model",
                    json={
                        "model_name": model_name,
                        "model_version": model_version,
                        "hyperparameters": json.loads(hyperparameters),
                    },
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code == 200:
                    st.success("Модель успешно добавлена!")
                    st.json(response.json())
                else:
                    st.error(f"Ошибка API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error("Произошла ошибка при добавлении модели. Проверьте логи.")
                logger.error("Ошибка при добавлении модели", exc_info=True)

            try:
                # Формирование данных модели
                model_info = {
                    "model_name": str(model_name + '_' + model_version),
                    "hyperparameters": json.loads(hyperparameters)
                }

                # Запись данных в JSON файл
                with open("model_info.json", "w") as json_file:
                    json.dump(model_info, json_file, indent=4)
                
                # Создание бакета, если не существует
                if not client.bucket_exists(MINIO_BUCKET_NAME):
                    client.make_bucket(MINIO_BUCKET_NAME)

                # Получение MIME-типа файла
                content_type, _ = mimetypes.guess_type("model_info.json")

                # Определение размера файла
                file_size = os.path.getsize("model_info.json")

                # Загрузка файла в MinIO
                with open("model_info.json", "rb") as json_file:
                    client.put_object(
                        MINIO_BUCKET_NAME,
                        f"models/{model_name}_{model_version}.json",
                        json_file,
                        length=file_size,
                        content_type=content_type or 'application/json',  # Использование значения по умолчанию
                    )

                st.success(f"Модель {str(model_name + '_' + model_version)} успешно загружена в S3!")

            except S3Error as err:
                st.error(f"Произошла ошибка при загрузке файла в S3. Проверьте логи.")
                logger.error("Ошибка при загрузке файла в S3", exc_info=True)

    if st.button("Показать модели"):
        with st.spinner("Обращаемся к Minio..."):
            try:
                # Получаем список объектов в бакете
                objects = client.list_objects("datasets", prefix="models/", recursive=True)
                model_list = [obj.object_name for obj in objects]
                if model_list:
                    for model in model_list:
                        st.write(f"- {model}")
                else:
                    st.write("Нет доступных моделей.")
            except S3Error as e:
                st.error(f"Ошибка при подключении к MinIO: {e}")


# Вкладка: Работа с S3
with tabs[1]:
    st.header("Загрузка данных для обучения в S3")

    # Загрузка файла
    uploaded_file = st.file_uploader("Выберите файл для загрузки в S3")
    if uploaded_file and st.button("Загрузить файл в формате JSON"):
        with st.spinner("Загрузка файла..."):
            try:
                file_name = uploaded_file.name
                # Создание bucket, если не существует
                if not client.bucket_exists(MINIO_BUCKET_NAME):
                    client.make_bucket(MINIO_BUCKET_NAME)
                client.put_object(
                    MINIO_BUCKET_NAME,
                    f'datasets/{file_name}',
                    uploaded_file,
                    length=len(uploaded_file.getvalue()),  # Размер файла
                    content_type=uploaded_file.type,
                )
                st.success(f"Файл {file_name} успешно загружен в S3!")
            except S3Error as err:
                st.error(f"Произошла ошибка при загрузке файла в S3. Проверьте логи.")
                logger.error("Ошибка при загрузке файла в S3", exc_info=True)

    # Список файлов в S3
    if st.button("Показать данные в хранилище"):
        with st.spinner("Обращаемся к Minio..."):
            try:
                files = []
                for obj in client.list_objects(MINIO_BUCKET_NAME, prefix="datasets/", recursive=True):
                    files.append(obj.object_name)
                if files:
                    st.write(files)
                else:
                    st.write("Нет доступных файлов.")
            except S3Error as err:
                st.error(f"Произошла ошибка при получении списка файлов. Проверьте логи.")
                logger.error("Ошибка при получении списка файлов из S3", exc_info=True)

# Вкладка: Обучить модель
with tabs[2]:
    st.header("Обучить модель")

    # Ввод данных пользователем
    train_model_name = st.text_input("Название модели для обучения")
    dataset = st.text_input("Название датасета", value="example_dataset")
    target_field = st.text_input("Столбец с целевой переменной", value="Target")

    if st.button("Обучить модель", key="train_model"):
        with st.spinner("Загружаем данные и модель..."):
            try:
                # Формируем пути для модели и данных
                model_path = f"models/{train_model_name}.json"
                dataset_path = f"datasets/{dataset}.json"

                # Загружаем модель из MinIO
                model_response = client.get_object("datasets", model_path)
                model_data = json.load(model_response)
                model_name = model_data["model_name"]
                hyperparameters = model_data["hyperparameters"]

                # Создаём модель
                if "LogisticRegression" in model_name:
                    model = LogisticRegression(**hyperparameters)
                elif "RandomForestClassifier" in model_name:
                    model = RandomForestClassifier(**hyperparameters)
                else:
                    raise ValueError(f"Неизвестный тип модели: {model_name}")

                # Загружаем датасет из MinIO
                dataset_response = client.get_object(MINIO_BUCKET_NAME, dataset_path)
                dataset_data = json.load(dataset_response)

                # Преобразуем данные в DataFrame
                df = pd.DataFrame(dataset_data)

                # Проверяем наличие целевого столбца
                if target_field not in df.columns:
                    raise ValueError(f"Целевой столбец {target_field} отсутствует в данных.")

                # Разделяем данные на признаки и целевую переменную
                X = df.drop(columns=[target_field])
                y = df[target_field]

                # Обучение модели
                with st.spinner("Обучение модели..."):
                    model.fit(X, y)
                    model_file = f"{model_name}_model.pkl"
                    joblib.dump(model, model_file)
                    st.success(f"Модель {model_name} успешно обучена!")

                    # Сохраняем обученную модель в MinIO
                    client.put_object(
                        "datasets",
                        f'models/{model_file}',
                        data=open(model_file, "rb"),
                        length=os.path.getsize(model_file),
                        content_type="application/octet-stream"
                    )
                    st.success(f"Модель {model_file} успешно загружена в S3!")
            except Exception as e:
                st.error("Произошла ошибка при обучении модели. Проверьте логи.")
                logger.error("Ошибка при обучении модели", exc_info=True)

# Вкладка: Предсказание
with tabs[3]:
    st.header("Сделать предсказание")

    # Ввод данных пользователем
    pred_model_name = st.text_input("Название модели для предсказания")
    pred_dataset = st.text_input("Название датасета для предсказания")

    if st.button("Сделать предсказание"):
        with st.spinner("Выполнение предсказания..."):
            try:
                # Формируем пути для модели и данных
                model_path = f"models/{pred_model_name}_model.pkl"
                dataset_path = f"datasets/{pred_dataset}.json"

                # Загружаем модель из MinIO
                model_response = client.get_object("datasets", model_path)
                model_data = BytesIO(model_response.read())
                model = joblib.load(model_data)
                model_response.close()

                # Загружаем датасет из MinIO
                dataset_response = client.get_object("datasets", dataset_path)
                dataset_data = json.load(dataset_response)
                dataset_response.close()

                # Преобразуем данные в DataFrame
                df = pd.DataFrame(dataset_data)

                # Выполняем предсказание
                predictions = model.predict(df)
                st.write(predictions)

                # Сохраняем предсказания в MinIO
                predictions_file = f"{pred_model_name}_predictions.json"
                with open(predictions_file, "w") as f:
                    json.dump(predictions.tolist(), f)

                client.put_object(
                    "datasets",
                    f'predictions/{predictions_file}',
                    data=open(predictions_file, "rb"),
                    length=os.path.getsize(predictions_file),
                    content_type="application/json"
                )
                st.success(f"Предсказания сохранены в S3 в файле {predictions_file}")
            except S3Error as e:
                st.error(f"Произошла ошибка при работе с S3: {e}")
                logger.error("Ошибка при работе с S3", exc_info=True)
            except Exception as e:
                st.error("Произошла ошибка при выполнении предсказания. Проверьте логи.")
                logger.error("Ошибка при выполнении предсказания", exc_info=True)

# Вкладка: Удалить данные
with tabs[4]:
    st.header("Удалить объекты из хранилища")

    # Удаление моделей
    if st.button("Показать модели", key='find_to_drop'):
        with st.spinner("Обращаемся к Minio..."):
            try:
                # Получаем список объектов в бакете
                objects = client.list_objects("datasets", prefix="models/", recursive=True)
                model_list = [obj.object_name for obj in objects]
                if model_list:
                    for model in model_list:
                        st.write(f"- {model}")
                else:
                    st.write("Нет доступных моделей.")
            except S3Error as e:
                st.error(f"Ошибка при подключении к MinIO: {e}")
                logger.error("Ошибка при подключении к MinIO", exc_info=True)

    model_to_delete = st.text_input("Введите название модели для удаления (прописывать с постфиксом)")

    if st.button("Удалить модель"):
        with st.spinner("Удаляем модель..."):
            try:
                # Удаляем модель из MinIO
                client.remove_object("datasets", f"models/{model_to_delete}")
                st.success(f"Модель {model_to_delete} успешно удалена из хранилища.")
            except S3Error as e:
                st.error(f"Произошла ошибка при удалении модели: {e}")
                logger.error("Ошибка при удалении модели из S3", exc_info=True)
            except Exception as e:
                st.error(f"Произошла ошибка при удалении модели.")
                logger.error("Ошибка при удалении модели", exc_info=True)

    # Удаление данных
    if st.button("Показать данные", key='find_to_drop_data'):
        with st.spinner("Обращаемся к Minio..."):
            try:
                # Получаем список объектов данных в бакете
                objects = client.list_objects("datasets", prefix="datasets/", recursive=True)
                data_list = [obj.object_name for obj in objects]
                if data_list:
                    for data in data_list:
                        st.write(f"- {data}")
                else:
                    st.write("Нет доступных данных.")
            except S3Error as e:
                st.error(f"Ошибка при подключении к MinIO: {e}")
                logger.error("Ошибка при подключении к MinIO", exc_info=True)

    data_to_delete = st.text_input("Введите название данных для удаления (прописывать с постфиксом)")

    if st.button("Удалить данные"):
        with st.spinner("Удаляем данные..."):
            try:
                # Удаляем данные из MinIO
                client.remove_object("datasets", f"datasets/{data_to_delete}")
                st.success(f"Данные {data_to_delete} успешно удалены из хранилища.")
            except S3Error as e:
                st.error(f"Произошла ошибка при удалении данных: {e}")
                logger.error("Ошибка при удалении данных из S3", exc_info=True)
            except Exception as e:
                st.error(f"Произошла ошибка при удалении данных.")
                logger.error("Ошибка при удалении данных", exc_info=True)

# Вкладка: Состояние сервера
with tabs[5]:
    st.header("Состояние сервера")
    st.write("Сервер работает корректно.")
