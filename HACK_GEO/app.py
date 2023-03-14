from fastapi import File, UploadFile
import numpy as np
from fastapi import FastAPI
from PIL import Image
import torch
import cv2
from io import BytesIO

app = FastAPI()
model = torch.load("model.pth")

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # Load the image
    content = await file.read()
    image = cv2.imdecode(np.fromstring(content, np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    # ...

    # Apply the segmentation model
    segmentation_mask = model(image)

    # Postprocess the segmentation mask
    # ...

    # Convert the segmentation mask to PNG and return it
    buffer = BytesIO()
    Image.fromarray(segmentation_mask).save(buffer, format="PNG")
    return {"segmentation_mask": buffer.getvalue()}
