from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import os

# ============================================
# НАСТРОЙКИ
# ============================================

# Меняем рабочую директорию
os.chdir(r'C:\ai-projects\ml-pipeline')

# Загружаем модели
model_package = joblib.load('models/model_package.pkl')
model_character_v2 = joblib.load('models/model_character_v2.pkl')
character_gender_map = joblib.load('models/character_gender_map.pkl')

# Создаем приложение
app = FastAPI(
    title="Анимэль ML API",
    description="Рекомендации по package и персонажам для детских праздников",
    version="1.0.0"
)

# ============================================
# МОДЕЛИ ЗАПРОСОВ
# ============================================

class RecommendationRequest(BaseModel):
    age: int
    kids_count: int
    gender: Optional[str] = None  # 'M', 'F', или None
    top_k: int = 3

class PackageRecommendation(BaseModel):
    name: str
    probability: float

class CharacterRecommendation(BaseModel):
    name: str
    probability: float
    gender: str

class RecommendationResponse(BaseModel):
    age: int
    age_group: str
    kids_count: int
    kids_group: str
    gender: Optional[str]
    top_packages: List[PackageRecommendation]
    top_characters: List[CharacterRecommendation]

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def get_age_group(age: int) -> str:
    if age <= 3:
        return '1-3'
    elif age <= 6:
        return '4-6'
    elif age <= 9:
        return '7-9'
    elif age <= 12:
        return '10-12'
    else:
        return '13+'

def get_kids_group(kids_count: int) -> str:
    if kids_count <= 5:
        return '1-5'
    elif kids_count <= 10:
        return '6-10'
    elif kids_count <= 15:
        return '11-15'
    else:
        return '15+'

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Анимэль ML API работает!",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "ok",
        "models_loaded": True,
        "package_classes": len(model_package.classes_),
        "character_classes": len(model_character_v2.classes_)
    }

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Получить рекомендации по package и character
    
    - **age**: возраст именинника (1-15)
    - **kids_count**: количество детей (1-30)
    - **gender**: пол ребенка ('M' или 'F', опционально)
    - **top_k**: количество рекомендаций (по умолчанию 3)
    """
    
    # Валидация входных данных
    if not (1 <= request.age <= 15):
        raise HTTPException(status_code=400, detail="Возраст должен быть от 1 до 15")
    
    if not (1 <= request.kids_count <= 30):
        raise HTTPException(status_code=400, detail="Количество детей должно быть от 1 до 30")
    
    if request.gender and request.gender not in ['M', 'F']:
        raise HTTPException(status_code=400, detail="Пол должен быть 'M' или 'F'")
    
    # Определяем группы
    age_group = get_age_group(request.age)
    kids_group = get_kids_group(request.kids_count)
    gender_input = request.gender if request.gender else 'unknown'
    
    # Создаем DataFrame для предсказания
    input_data = pd.DataFrame({
        'age': [request.age],
        'age_group': [age_group],
        'kids_count': [request.kids_count],
        'kids_group': [kids_group],
        'gender': [gender_input]
    })
    
    # Предсказания для package
    pkg_proba = model_package.predict_proba(input_data)[0]
    pkg_indices = np.argsort(pkg_proba)[::-1][:request.top_k]
    top_packages = [
        PackageRecommendation(
            name=model_package.classes_[i],
            probability=float(pkg_proba[i])
        )
        for i in pkg_indices
    ]
    
    # Предсказания для character с фильтрацией по полу
    char_proba = model_character_v2.predict_proba(input_data)[0]
    char_indices = np.argsort(char_proba)[::-1]
    
    filtered_chars = []
    for idx in char_indices:
        char_name = model_character_v2.classes_[idx]
        char_prob = float(char_proba[idx])
        
        # Определяем пол персонажа
        char_clean = char_name.lower().strip()
        char_gender = character_gender_map.get(char_clean, 'U')
        
        # Фильтруем по полу (если указан)
        if request.gender and char_gender != 'U' and char_gender != request.gender:
            continue
        
        filtered_chars.append(
            CharacterRecommendation(
                name=char_name,
                probability=char_prob,
                gender=char_gender
            )
        )
        
        if len(filtered_chars) >= request.top_k:
            break
    
    return RecommendationResponse(
        age=request.age,
        age_group=age_group,
        kids_count=request.kids_count,
        kids_group=kids_group,
        gender=request.gender,
        top_packages=top_packages,
        top_characters=filtered_chars
    )

# ============================================
# ЗАПУСК СЕРВЕРА
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)