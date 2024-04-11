from fastapi import FastAPI
import os
import torch
from faiss import read_index

import torch
import torchvision.transforms as T


dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

import pandas as pd

df = pd.read_json('mapping_vec_img.json')



import torchvision.transforms as T
import torch
from PIL import Image

transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    Convert single-channel grayscale images to three-channel RGB images.
    """
    img = Image.open(img)

    # Проверяем количество каналов в изображении
    if img.mode == 'L':
        img = img.convert('RGB')  # Если одноканальное изображение, преобразуем в трехканальное RGB

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img


# Определяем маршрут для обработки запросов
def run_inference(image_path: str, num_similar: int = 4):
    # Загрузка изображения
    query_image = load_image(image_path)
    query_image = query_image

    # Предобработка изображения и получение вложения
    query_embedding = dinov2_vits14(query_image).detach().numpy()

    # Поиск похожих изображений
    index = read_index("siberiada.index")

    _, I = index.search(query_embedding, num_similar)

    # Состав списка имен файлов похожих изображений
    similar_images = []
    for index in I[0]:
        filename = df.iloc[index]['filename']
        image_path = os.path.join(filename)
        image_path = os.path.normpath(image_path)
        similar_images.append(image_path)

    # Возвращаем результат
    return {"similar_images": similar_images}


query_image = "C:/Users/Arnold/Downloads/Сибириана_коллеекции/1/3_Взгляд снизу. Автор Константин Хилько.jpg"

print(run_inference(query_image, num_similar = 4))