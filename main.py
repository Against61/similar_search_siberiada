from fastapi import FastAPI
import os
import torch
from faiss import read_index
import asyncio
import pandas as pd

from preproccesing import load_image
from model import dinov2_vits14

device = torch.device("cpu")

df = pd.read_json('mapping_vec_img.json')

# Создаем экземпляр приложения FastAPI
app = FastAPI()

# Определяем маршрут для обработки запросов
@app.post("/inference")
async def run_inference(image_path: str, num_similar: int = 4):
    # Загрузка изображения
    query_image = await asyncio.get_event_loop().run_in_executor(None, load_image, image_path)
    query_image = query_image

    # Предобработка изображения и получение вставки
    query_embedding = await asyncio.get_event_loop().run_in_executor(None, dinov2_vits14, query_image)
    query_embedding = query_embedding.detach().cpu().numpy()

    # Поиск похожих изображений
    index = await asyncio.get_event_loop().run_in_executor(None, read_index, "siberiada.index")

    _, I = await asyncio.get_event_loop().run_in_executor(None, index.search, query_embedding, num_similar)

    # Состав списка путей к похожим изображениям
    similar_images = []
    for index in I[0]:
        filename = df.iloc[index]['filename']
        # image_path = await asyncio.get_event_loop().run_in_executor(None, os.path.join, filename)
        # image_path = os.path.normpath(image_path)
        similar_images.append(filename)

    # Возвращаем результат
    return {"similar_images": similar_images}


# Запускаем сервер FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=883)
