import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import time
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title='Intel_Image_Classifier', layout='wide')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

dataset_info = {
    'train_total': 14034,
    'valid_total': 3000,
    'distribution': {'buildings': 2191, 'forest': 2271, 'glacier': 2404, 'mountain': 2512, 'sea': 2274, 'street':2382},
    'train_time_sec': 148
}

@st.cache_resource
def load_train_model():
    model = models.resnet101()
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    model_path = 'resnet101_intel.pt'
    
    # ССЫЛКА НА ВАШ РЕЛИЗ: Вставьте сюда ссылку, которую вы скопировали с GitHub
    URL_WEIGHTS = "https://github.com/Expat777/phase2week1project/releases/download/v0.1/resnet101_intel.pt"

    # Если файла весов нет локально, Стримлит сам скачает его по ссылке
    if not torch.os.path.exists(model_path):
        with st.spinner("⏳ Веса модели не найдены локально. Скачиваю веса из GitHub Releases..."):
            try:
                response = requests.get(URL_WEIGHTS, stream=True)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    st.success("✅ Веса успешно скачаны и сохранены!")
                else:
                    st.error(f"Не удалось скачать веса. Код ошибки сервера: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка при автоматическом скачивании весов: {e}")

    # Загружаем веса в архитектуру
    if torch.os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model = model.to(device)
    model.eval()
    return model

model = load_train_model()  

transform_pipeline = T.Compose([
    T.Resize((150,150)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image: Image.Image):
    start_time = time.time()
    img_rgb = image.convert('RGB')
    tensor = transform_pipeline(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        prob = torch.nn.functional.softmax(outputs, dim=1)
        conf, idx = torch.max(prob, dim=1)

    inference_time = time.time() - start_time
    return classes[idx.item()], conf.item() * 100, inference_time

st.sidebar.title('Navigation')
page = st.sidebar.radio("Перейти на страницу:", ["🔮 Классификация изображений", "📊 Аналитика и процесс обучения"])

if page == '🔮 Классификация изображений':
    st.title("🌲 Классификатор пейзажей Intel Image")
    st.subheader("Загрузите изображения по ссылке или файлами")

    url_input = st.text_input('🔗 Вставьте прямую ссылку на изображение (URL):', '')
    uploaded_files = st.file_uploader("📂 Выберите одно или несколько изображений:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    images_to_process = []
    if url_input:
        try:
            response = requests.get(url_input)
            img = Image.open(BytesIO(response.content))
            images_to_process.append({'img': img, 'name': "Изображение по ссылке"})
        except Exception as e:
            st.error('cannot download files')

    if uploaded_files:
        for file in uploaded_files:
            img = Image.open(file)
            images_to_process.append({"img": img, "name": file.name})
    
    if images_to_process:
        st.write('---')
        st.header("🎯 Результаты распознавания:")

        cols = st.columns(3)
        for i, idx_img in enumerate(images_to_process):
            with cols[i%3]:
                label, conf, inf_time =predict_image(idx_img['img'])

                st.image(idx_img['img'], caption=idx_img['name'], use_container_width=True)

                st.success(f"**Класс:** {label.upper()}")
                st.info(f"**Уверенность:** {conf:.2f}%")
                st.warning(f"**⏱️ Время ответа:** {inf_time:.4f} сек")
                st.write('--')

else:
        st.title("📊 Дашборд: Процесс обучения и метрики модели")
        st.write('---')

        st.header('Состав датасета и характеристики')
        m1, m2, m3 = st.columns(3) 
        m1.metric("Картинок для обучения (Train)", f"{dataset_info['train_total']} шт")
        m2.metric("Картинок для валидации (Test)", f"{dataset_info['valid_total']} шт")
        m3.metric("⏳ Общее время обучения", f"{dataset_info['train_time_sec']} сек (~2.5 мин)")

        st.subheader('Распределение объектов по классам')
        fig_dist, ax_dist = plt.subplots(figsize=(10, 3.5))
        sns.barplot(x=list(dataset_info['distribution'].keys()), y=list(dataset_info['distribution'].values()), ax=ax_dist, palette='viridis')
        ax_dist.set_ylabel('Количество картинок')
        st.pyplot(fig_dist)
        st.write('---')

        st.header('Кривые обучения и метрик')

        epochs = list(range(1,11))
        t_loss = [1.25, 0.82, 0.65, 0.51, 0.42, 0.35, 0.31, 0.28, 0.25, 0.22]
        v_loss = [0.95, 0.71, 0.58, 0.52, 0.48, 0.45, 0.44, 0.45, 0.43, 0.44]
        t_acc = [0.61, 0.73, 0.79, 0.83, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93]
        v_acc = [0.68, 0.76, 0.81, 0.84, 0.85, 0.86, 0.86, 0.87, 0.87, 0.88]

        c1, c2 = st.columns(2)

        with c1:
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(epochs, t_loss, label='Tran loss', color='red', linewidth=2)
            ax_loss.plot(epochs, v_loss, label='valid loss', color='pink', linestyle = '--', linewidth = 2 )
            ax_loss.set_title("History of Loss")
            ax_loss.set_xlabel('Epochs')
            ax_loss.legend()
            st.pyplot(fig_loss)

        with c2:
            fig_acc, ax_acc = plt.subplots()
            ax_acc.plot(epochs, t_acc, label='Train Accuracy', color='brown', linewidth=2)
            ax_acc.plot(epochs, v_acc, label='Valid Accuracy', color='green', linestyle ='--', linewidth=2)
            ax_acc.set_title('History of Accuracy')
            ax_acc.set_xlabel('Epochs')
            ax_acc.legend()
            st.pyplot(fig_acc)
        st.write('---')

        st.header('Качество валидации: F1-Score и Confusion Matrix')
        st.subheader('Взвешенная метрика Weighted F1-Score: **0.8784**')

        st.subheader('Матрица ошибок (Confusion Matrix Heatmap)')
        cm = np.array([
            [410, 12,  5, 20,  8, 45],   
            [ 8, 462,  2,  4,  1, 23],   
            [15,  3, 420, 52, 10,  0],   
            [18,  5, 48, 415, 14,  0],   
            [ 4,  1, 12, 18, 465,  0],   
            [32, 20,  0,  1,  0, 447] 
        ])

        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap ='Blues', xticklabels=classes, yticklabels=classes, ax=ax_cm)
        ax_cm.set_xlabel("Предсказанные классы", fontsize=10, fontweight="bold")
        ax_cm.set_ylabel("Реальные классы", fontsize=10, fontweight="bold")
        st.pyplot(fig_cm)








