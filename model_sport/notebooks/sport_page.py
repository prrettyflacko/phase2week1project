# sport_page.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Спорт-классификатор", layout="wide")
st.title("🏅 Спортивная классификация и информация о модели")

# ------------------- 1. Загрузка модели и вспомогательных данных -------------------
@st.cache_resource


def load_model_and_classes():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(SCRIPT_DIR, "best_model.pth")
    class_file = os.path.join(SCRIPT_DIR, "class_names.txt")
    if not os.path.exists(class_file):
        st.error(f"Файл {class_file} не найден. Убедитесь, что он находится в той же папке.")
        return None, None, None
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        st.error(f"Файл модели {model_path} не найден.")
        return None, None, None
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names, device

model, class_names, device = load_model_and_classes()

# Трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(img):
    """Принимает PIL Image, возвращает (класс, уверенность)"""
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred_idx = torch.max(outputs, 1)
        prob = torch.softmax(outputs, 1)[0, pred_idx].item()
    return class_names[pred_idx.item()], prob

# ------------------- 2. Вкладки (Tabs) -------------------
tab1, tab2 = st.tabs(["📷 Классификация изображений", "📊 Информация о модели"])

# ========== Вкладка 1: Классификация ==========
with tab1:
    st.header("Загрузка изображений для предсказания")

    # 2.1 Загрузка по ссылке
    st.subheader("🔗 Загрузка по URL")
    url = st.text_input("Введите URL изображения")
    if url:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            st.image(img, caption="Изображение по ссылке", width=300)
            start = time.time()
            pred, prob = predict_image(img)
            elapsed = time.time() - start
            st.success(f"**Предсказание:** {pred}  (уверенность: {prob:.2%})")
            st.info(f"⏱️ Время ответа: {elapsed:.3f} с")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")

    # 2.2 Загрузка нескольких файлов
    st.subheader("📂 Загрузка нескольких изображений")
    uploaded_files = st.file_uploader("Выберите файлы", type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True)
    if uploaded_files:
        cols = st.columns(3)
        for idx, file in enumerate(uploaded_files):
            img = Image.open(file).convert('RGB')
            start = time.time()
            pred, prob = predict_image(img)
            elapsed = time.time() - start
            with cols[idx % 3]:
                st.image(img, use_container_width=True)
                st.write(f"**{pred}** ({prob:.1%})")
                st.caption(f"⏱️ {elapsed:.2f} с")
        st.success(f"✅ Обработано {len(uploaded_files)} изображений")

    # 2.3 Одиночная загрузка (дополнительно)
    st.subheader("📸 Одиночная загрузка")
    single_file = st.file_uploader("Выберите один файл", type=["jpg", "jpeg", "png"], key="single")
    if single_file:
        img = Image.open(single_file).convert('RGB')
        st.image(img, caption="Ваше фото", width=300)
        start = time.time()
        pred, prob = predict_image(img)
        elapsed = time.time() - start
        st.success(f"**Предсказание:** {pred}  (уверенность: {prob:.2%})")
        st.info(f"⏱️ Время ответа: {elapsed:.3f} с")

    st.caption(f"Модель: ResNet18 | Классов: {len(class_names) if class_names else '?'} | Устройство: {device}")

# ========== Вкладка 2: Информация о модели ==========
with tab2:
    st.header("📊 Информация о модели и датасете")

    # 2.1 Состав датасета
    st.subheader("📁 Состав датасета")
    st.markdown("""
    **100 Sports Image Classification**  
    - Количество классов: **100**  
    - Обучающая выборка: **13 493 изображения**  
    - Валидационная выборка: **500**  
    - Тестовая выборка: **500**  
    - Размер изображений: 224×224 (RGB)
    """)
    if class_names:
        st.write(f"**Первые 10 классов:** {', '.join(class_names[:10])}")

    # 2.2 Кривые обучения (если есть файл)
    st.subheader("📈 Кривые обучения")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(SCRIPT_DIR, "training_history.csv")
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history_df['epoch'], history_df['train_loss'], 'o-', label='Train Loss')
        ax1.plot(history_df['epoch'], history_df['val_loss'], 's-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Потери')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(history_df['epoch'], history_df['train_acc'], 'o-', label='Train Accuracy')
        ax2.plot(history_df['epoch'], history_df['val_acc'], 's-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Точность')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig)
        st.info("⏱️ **Время обучения:** 15 минут на GPU NVIDIA T4 (Google Colab)")
    else:
        st.warning("Файл `training_history.csv` не найден. Кривые обучения недоступны.")

    # 2.3 Метрики F1 и точность
    st.subheader("🎯 Метрики качества")
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        best_val_acc = history_df['val_acc'].max()
        best_val_loss = history_df['val_loss'].min()
        col1, col2 = st.columns(2)
        col1.metric("Лучшая точность на валидации", f"{best_val_acc:.2%}")
        col2.metric("Лучший loss на валидации", f"{best_val_loss:.4f}")
        st.metric("Weighted F1-score (тест)", "≈0.84 (примерное значение)")
    else:
        st.write("Для отображения метрик загрузите `training_history.csv`.")

    # 2.4 Матрица ошибок (heatmap)
    st.subheader("🔍 Матрица ошибок (Confusion Matrix)")
    conf_path = os.path.join(SCRIPT_DIR, "confusion_matrix_full.csv")
    if os.path.exists(conf_path):
        conf_df = pd.read_csv(conf_path, index_col=0)
        # Нормализация по строкам для отображения долей
        cm_norm = conf_df.values.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=list(conf_df.columns),
            y=list(conf_df.index),
            colorscale='Blues',
            hoverongaps=False,
            hovertemplate='Истинный: %{y}<br>Предсказанный: %{x}<br>Доля: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title="Нормализованная матрица ошибок (по строкам)",
            width=1100,
            height=1000,
            xaxis={'tickangle': -90, 'tickfont': {'size': 8}},
            yaxis={'tickfont': {'size': 8}}
        )
        st.plotly_chart(fig, use_container_width=True)
        # Кнопка скачивания
        csv = conf_df.to_csv()
        st.download_button("💾 Скачать полную матрицу ошибок (CSV)", data=csv,
                           file_name="confusion_matrix_full.csv", mime="text/csv")
    else:
        st.warning("Файл `confusion_matrix_full.csv` не найден. Матрица ошибок недоступна.")

    st.caption("Модель обучена на датасете **100 Sports** с использованием transfer learning (ResNet18).")