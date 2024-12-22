import sys
import os
import threading
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from endpoints import register_endpoints

# Добавляем путь к родительскому каталогу
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Загружаем переменные окружения из файла .env
load_dotenv(os.path.join(parent_dir, "utils/s3_env.env"))

# Проверка, что все необходимые переменные окружения для S3 заданы
required_env_vars = ["S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET_NAME"]

if not all([os.environ.get(var) for var in required_env_vars]):
    raise EnvironmentError("Не все переменные окружения S3 настроены.")

# FastAPI приложение
app = FastAPI()

# Регистрируем эндпоинты
register_endpoints(app)

# Запуск FastAPI
def run_fastapi():
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Ошибка при запуске FastAPI: {e}")
        sys.exit(1)

# Запуск Streamlit
def run_streamlit():
    try:
        os.system("streamlit run streamlit_app.py")
    except Exception as e:
        print(f"Ошибка при запуске Streamlit: {e}")
        sys.exit(1)

# Запуск приложений в потоках
if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi)
    streamlit_thread = threading.Thread(target=run_streamlit)
    
    # Запускаем FastAPI и Streamlit параллельно
    fastapi_thread.start()
    streamlit_thread.start()
    
    # Ожидаем завершения обоих потоков
    fastapi_thread.join()  # Дождаться завершения FastAPI
    streamlit_thread.join()  # Дождаться завершения Streamlit
