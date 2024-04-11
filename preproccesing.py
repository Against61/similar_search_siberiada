import cv2
import numpy as np
import torchvision.transforms as T
import torch
from PIL import Image

import requests
from io import BytesIO
from PIL import Image

device = torch.device("cpu")

transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

def load_image(image_url: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    Convert single-channel grayscale images to three-channel RGB images.
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Проверяем количество каналов в изображении
    if img.mode == 'L':
        img = img.convert('RGB')  # Если одноканальное изображение, преобразуем в трехканальное RGB

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img.to(device)
