import os
import time
import torch
import torch.nn as nn
import streamlit as st
import requests  # 🎯 НАШ НОВЫЙ ИМПОРТ ДЛЯ РАБОТЫ С URL
from PIL import Image
from torchvision.models import resnet50
import torchvision.transforms as transforms

# 1. Архитектура класса модели
class AdvancedBloodClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(AdvancedBloodClassifier, self).__init__()
        self.base_model = resnet50(weights=None)
        
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),       # fc.0
            nn.BatchNorm1d(512),                # fc.1
            nn.ReLU(),                          # fc.2
            nn.Dropout(0.2),                    # fc.3
            nn.Linear(512, 128),                # fc.4
            nn.ReLU(),                          # fc.5
            nn.Dropout(0.2),                    # fc.6
            nn.Linear(128, num_classes)         # fc.7
        )

    def forward(self, x):
        return self.base_model(x)

# 2. Функция загрузки модели
def load_my_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AdvancedBloodClassifier(num_classes=4)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, 'weights', 'best_blood_resnet50.pth')
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device

# Алфавитный порядок классов
CLASS_NAMES = [
    'Эозинофил (Eosinophil)', 
    'Лимфоцит (Lymphocyte)', 
    'Моноцит (Monocyte)', 
    'Нейтрофил (Neutrophil)'
]

# Синхронизированный конвейер трансформаций (192x192)
test_transforms = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])

# Вычисляем пути к графикам аналитики
current_dir = os.path.dirname(os.path.abspath(__file__))
curves_path = os.path.join(current_dir, 'weights', 'learning_curves.png')
cm_path = os.path.join(current_dir, 'weights', 'confusion_matrix.png')

# Создаем вкладки
tab1, tab2 = st.tabs(["🔬 Классификатор", "📊 Аналитика и Метрики"])

# ==============================================================================
# ВКЛАДКА 1: КЛАССИФИКАЦИЯ
# ==============================================================================
with tab1:
    st.title("🔬 Веб-анализатор клеток крови")
    st.subheader("Модуль автоматической классификации лейкоцитов (ResNet50)")
    st.write("---")

    # 🎯 Выбор способа загрузки снимка
    source_select = st.radio(
        "Выберите способ загрузки изображения клетки:",
        ["Загрузить файл с компьютера 💻", "Вставить ссылку (URL) из интернета 🌐"]
    )
    
    image = None  # Сюда запишем итоговый PIL-объект для нейросети

    if source_select == "Загрузить файл с компьютера 💻":
        uploaded_file = st.file_uploader("Перетащите сюда снимок клетки крови", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
    else:
        url_input = st.text_input("Вставьте прямую ссылку на JPG/PNG картинку клетки:")
        if url_input:
            try:
                # Безопасно скачиваем картинку по сети с таймаутом в 10 секунд
                with st.spinner("Скачивание изображения по ссылке..."):
                    response = requests.get(url_input, timeout=10, stream=True)
                    if response.status_code == 200:
                        image = Image.open(response.raw).convert("RGB")
                    else:
                        st.error(f"❌ Не удалось скачать картинку. Сервер вернул код ошибки: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке URL. Проверьте правильность ссылки. Техническая инфо: {e}")

    # Если картинка успешно получена (с ПК или по URL) — запускаем нейросеть
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Образец для анализа", use_container_width=True)
            
        with col2:
            st.write("### Результат анализа модели:")
            
            with st.spinner("Нейросеть обрабатывает снимок..."):
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    
                model, device = load_my_model()
                input_tensor = test_transforms(image).unsqueeze(0).to(device)
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, preds = torch.max(outputs, 1)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                inference_time = time.time() - start_time
                
                predicted_class = CLASS_NAMES[preds.item()]
                
            st.success(f"🤖 Обнаружен тип: **{predicted_class}**")
            st.metric(label="⏱️ Время ответа модели (Инференс)", value=f"{inference_time:.4f} сек")
            
            st.write("**Уверенность нейросети по всем типам клеток:**")
            for i, name in enumerate(CLASS_NAMES):
                percentage = float(probabilities[0, i].item())
                st.write(f"{name}")
                st.progress(percentage)
                st.caption(f"Вероятность: {percentage*100:.2f}%")

# ==============================================================================
# ВКЛАДКА 2: АНАЛИТИКА
# ==============================================================================
with tab2:
    st.title("📊 Аналитика и Метрики Обучения")
    st.write("---")
    if os.path.exists(curves_path):
        st.image(curves_path, use_container_width=True)
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)
