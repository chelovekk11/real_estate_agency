# app_fastapi.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from category_encoders import TargetEncoder

app = FastAPI(
    title="LightGBM Prediction API",
    description="API для предсказания стоимости недвижимости с использованием LightGBM",
    version="1.0.0"
)

# Определение схемы данных для запроса
class PredictionRequest(BaseModel):
    baths: float
    sqft_log: float
    beds: float
    year_built: int
    stories: float
    city: str
    zipcode: str
    state: str
    # Добавьте другие признаки, если необходимо

# Загрузка модели при запуске приложения
# Путь к директории модели
model_dir = r'C:\IDE\final_projectr_real_estate_agency\models'

# Путь к файлу модели
MODEL_PATH = os.path.join(model_dir, 'LightGBM_Tuned.joblib')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель: {e}")

# Загрузка и подготовка Target Encoder
TE_PATH = os.path.join(model_dir, 'target_encoder.joblib')  
if not os.path.exists(TE_PATH):
    raise FileNotFoundError(f"Target Encoder не найден по пути: {TE_PATH}")

try:
    te = joblib.load(TE_PATH)
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить Target Encoder: {e}")

@app.post("/predict", summary="Предсказание стоимости недвижимости", description="Принимает данные о недвижимости и возвращает предсказанную стоимость.")
def predict(request: PredictionRequest):
    try:
        data = request.dict()
        df = pd.DataFrame([data])

        # Применение Target Encoding
        high_cardinality_cols = ['city', 'zipcode', 'state']
        df[high_cardinality_cols] = te.transform(df[high_cardinality_cols])

        # Предсказание
        pred_log = model.predict(df)[0]
        original_value = np.exp(pred_log)  

        return {"prediction": original_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {e}")

@app.get("/", summary="Корневой маршрут", description="Возвращает приветственное сообщение.")
def read_root():
    return {"message": "Добро пожаловать в LightGBM Prediction API! Перейдите по адресу /docs для доступа к документации Swagger UI."}
