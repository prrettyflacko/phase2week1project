import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import time
import os

# --- ОБЩАЯ НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Weather App (ResNet18)", page_icon="🌤️", layout="wide")

# --- ОПРЕДЕЛЕНИЕ ПУТЕЙ (АБСОЛЮТНЫЕ) ---
# Путь к папке, где лежит этот скрипт (предполагаем, что он внутри pages/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Корень проекта (на уровень выше, если pages/ внутри model_StP)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Пути к графикам (лежат в той же папке, что и скрипт, т.е. в pages/)
LEARNING_CURVES_PATH = os.path.join(SCRIPT_DIR, "learning_curves.png")
CONFUSION_MATRIX_PATH = os.path.join(SCRIPT_DIR, "confusion_matrix.png")
# Путь к весам (лежат в папке weights в корне проекта)
WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights", "resnet18.pth")

# --- КЭШИРОВАНИЕ И ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ---
@st.cache_resource
def load_weather_model():
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    
    classes = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
    
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(num_ftrs, len(classes))
    )
    
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    else:
        st.sidebar.error(f"⚠️ Файл весов не найден по пути: {WEIGHTS_PATH}")
        
    model.eval()
    return model, classes

# Вспомогательные функции для инференса
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        st.error(f"Не удалось загрузить изображение по ссылке. Ошибка: {e}")
        return None

# --- БОКОВОЕ МЕНЮ НАВИГАЦИИ ---
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел проекта:",
    ["Аналитика и метрики", "🌤️ Тестирование модели"]
)

# Загружаем модель один раз (кэшируется)
model, class_names = load_weather_model()

# ====================================================================
# РАЗДЕЛ 1: АНАЛИТИКА И МЕТРИКИ
# ====================================================================
if page == "Аналитика и метрики":
    st.title("Панель мониторинга и аналитики: Weather Image Recognition")
    st.write("Детальная статистика и результаты процесса обучения архитектуры ResNet18.")
    st.markdown("---")

    # Метрики верхнего уровня
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Размер датасета", value="6,862")
    with col2:
        st.metric(label="Количество классов", value="11 классов")
    with col3:
        st.metric(label="Время обучения (5 эпох)", value="4 мин 23 сек")
    with col4:
        st.metric(label="Итоговая точность (Accuracy)", value="91.62%")

    st.markdown("---")
    st.subheader("Результаты и кривые обучения")

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(LEARNING_CURVES_PATH):
            st.image(LEARNING_CURVES_PATH, caption="Кривые потерь (Loss) и точности (Accuracy) по эпохам")
        else:
            st.warning(f"⚠️ Файл 'learning_curves.png' не найден. Ожидаемый путь: {LEARNING_CURVES_PATH}")

    with c2:
        if os.path.exists(CONFUSION_MATRIX_PATH):
            st.image(CONFUSION_MATRIX_PATH, caption="Матрица ошибок (Confusion Matrix Heatmap)")
        else:
            st.warning(f"⚠️ Файл 'confusion_matrix.png' не найден. Ожидаемый путь: {CONFUSION_MATRIX_PATH}")

# ====================================================================
# РАЗДЕЛ 2: ТЕСТИРОВАНИЕ МОДЕЛИ
# ====================================================================
elif page == "🌤️ Тестирование модели":
    st.title("🌤️ Распознавание типов погоды — Проверка модели")
    st.write("Загружайте собственные изображения локально или используйте прямые ссылки для проверки качества работы нейросети.")
    
    # Проверка статуса весов (теперь с абсолютным путём)
    if os.path.exists(WEIGHTS_PATH):
        st.success(f"Веса модели успешно загружены из `{WEIGHTS_PATH}`")
    else:
        st.error(f"Веса не найдены по пути: {WEIGHTS_PATH}. Проверьте структуру папок.")
    
    st.markdown("---")
    st.subheader("Загрузка данных")
    upload_type = st.radio("Выберите метод передачи изображения:", ("Локальный файл (Мультизагрузка)", "Прямая ссылка (URL)"))

    images_to_process = []

    if upload_type == "Локальный файл (Мультизагрузка)":
        uploaded_files = st.file_uploader(
            "Перетащите файлы сюда или нажмите обзор", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            for file in uploaded_files:
                images_to_process.append(Image.open(file).convert('RGB'))

    elif upload_type == "Прямая ссылка (URL)":
        url_input = st.text_input("Вставьте прямую ссылку на изображение (.jpg / .png):")
        if url_input:
            img = load_image_from_url(url_input)
            if img:
                images_to_process.append(img)

    # Инференс и вывод
    if images_to_process:
        st.markdown("---")
        st.subheader("Результаты работы вашей модели")
        
        cols = st.columns(min(len(images_to_process), 3))
        transform = get_transform()
        
        for idx, img in enumerate(images_to_process):
            with cols[idx % 3]:
                st.image(img, use_container_width=True, caption=f"Изображение #{idx+1}")
                
                tensor = transform(img).unsqueeze(0)
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(tensor)
                    _, preds = torch.max(outputs, 1)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    prob = probs[0, preds.item()].item()
                end_time = time.time()
                
                inf_time = end_time - start_time
                label = class_names[preds.item()]
                
                st.success(f"**Класс:** `{label}`")
                st.info(f"**Доверие сети:** {prob:.2%}")
                st.warning(f"**Время инференса:** {inf_time:.4f} сек")
