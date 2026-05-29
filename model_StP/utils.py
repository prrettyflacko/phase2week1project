import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import time


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        st.error(f"Не удалось загрузить изображение по ссылке. Ошибка: {e}")
        return None


def predict(image, model, class_names, device="cpu"):
    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)

        # Исправленный подсчет вероятности:
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prob = probs[0, preds.item()].item()

    end_time = time.time()

    inference_time = end_time - start_time
    return class_names[preds.item()], prob, inference_time
