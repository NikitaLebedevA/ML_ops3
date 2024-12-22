# ml_server_with_miniO

Запускаем раз
docker-compose -f docker-compose.yml -f docker-compose.clearml.yml up --build

Запускаем два
docker exec -it fastapi_app streamlit run /app/streamlit_app.py --server.address 0.0.0.0

test data - для обучения моделей

predict data - для предсказания

Тесты запускаем как: /n
cd app /n
pytest test_streamlit_app.py
