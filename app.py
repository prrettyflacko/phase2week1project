import streamlit as st
import os

# 1. Глобальная настройка всего приложения
st.set_page_config(
    page_title="Командный ИИ-Помощник (4 Модели)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🧭 Навигация по проекту")
st.sidebar.write("Выберите интересующий ИИ-модуль:")

# 2. Проверяем наличие файлов страниц всех участников
blood_page_exists = os.path.exists("model_blood/blood_page.py")
sport_page_exists = os.path.exists("model_sport/sport_page.py")
intel_page_exists = os.path.exists("model_intel/intel_page.py")
weather_page_exists = os.path.exists("model_StP/pages/weather_page.py") # Путь к 4-й странице погоды

pages_to_load = []

# --- 1. Твоя страница (Клетки крови) ---
if blood_page_exists:
    pages_to_load.append(st.Page("model_blood/blood_page.py", title="🔬 Анализ клеток крови", icon="🩸"))
else:
    with st.sidebar:
         st.warning("⚠️ Модуль Крови не найден")

# --- 2. Страница Участника 2 (Спорт) ---
sport_page_exists = os.path.exists("model_sport/notebooks/sport_page.py")

if sport_page_exists:
    pages_to_load.append(st.Page("model_sport/notebooks/sport_page.py", title="⚽ Классификация спорта", icon="🏃‍♂️"))
else:
    def sport_placeholder():
        st.title("🏃‍♂️ Модуль: Классификация видов спорта")
        st.info("Разработка ведется Участником 2. Страница появится сразу после пуша файла `sport_page.py`.")
    pages_to_load.append(st.Page(sport_placeholder, title="⚽ Классификация спорта (В разработке)", icon="⏳"))

# --- 3. Страница Участника 3 (Intel/Природа) ---
if intel_page_exists:
    pages_to_load.append(st.Page("model_intel/intel_page.py", title="🏔️ Снимки Intel (Природа)", icon="🌲"))
else:
    def intel_placeholder():
        st.title("🌲 Модуль: Классификация снимков Intel")
        st.info("Разработка ведется Участником 3. Страница появится сразу после пуша файла `intel_page.py`.")
    pages_to_load.append(st.Page(intel_placeholder, title="🏔️ Снимки Intel (В разработке)", icon="⏳"))

# --- 4. 🆕 Страница Участника 4 (Классификация погоды) ---
if weather_page_exists:
    pages_to_load.append(st.Page("model_StP/pages/weather_page.py", title="🌈 Классификация погоды", icon="🌦️"))
else:
    def weather_placeholder():
        st.title("🌦️ Модуль: Классификация погоды")
        st.info("Разработка ведется Участником 4. Страница появится сразу после пуша файла `weather_page.py` в папку `model_StP`.")
    pages_to_load.append(st.Page(weather_placeholder, title="🌈 Классификация погоды (В разработке)", icon="⏳"))

# 3. Запускаем многостраничное меню на 4 вкладки
pg = st.navigation(pages_to_load)
pg.run()
