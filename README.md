# Similar Image Search Service Documentation

## Description
This service leverages the DINOv2 (DIstillation with NO labels) model to find images similar to a given one. It is built using the FastAPI library and is designed for image processing.

## Installation

To use this service, you must install the following dependencies:

```bash
pip install fastapi uvicorn torch torchvision faiss-cpu numpy opencv-python-headless pandas requests Pillow
```

Additionally, the pretrained DINOv2 model will be automatically downloaded upon the first run.


# Server Launch
Use the following command to start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 883
```

After launching, the service will be available at http://localhost:883.


# Usage
To search for similar images, send a POST request to /inference with the parameters:

- image_path - path to the image or its URL;
- num_similar - the number of similar images to return.

Here's an example request using curl:

```bash
curl -X 'POST' \
  'http://localhost:883/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_path": "path_to_image",
  "num_similar": 4
}'
```

The service will respond with a JSON containing paths to similar images.


# Workflow

1. The image is loaded and preprocessed.
2. A feature vector for the image is obtained using the DINOv2 model.
3. Similar images are searched using the feature vector and a FAISS index.
4. A list of paths to similar images is returned.

# Note

Remember to replace path_to_image with the actual path to your image or the image URL.
