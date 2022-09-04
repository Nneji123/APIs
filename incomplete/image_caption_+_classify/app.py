import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from PIL import Image
from utils import classify_image, self_caption

app = FastAPI(
    title="Image Captioning API",
    description="""Image Captioning API is a fastapi server that can generate captions for images.""",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=PlainTextResponse, tags=["home"])
async def home():
    note = """
    Image Captioning API 📚
    An API that can generate captions for images.
    Note: add "/redoc" to get the complete documentation.
    """
    return note


@app.post("/caption-image")
async def get_image(response_options: str, file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("images.jpg", img)
    try:
        image = Image.open("images.jpg")
        if response_options == "caption":
            data = self_caption(image)
            if os.path.exists("images.jpg"):
                os.remove("images.jpg")
            return data
        elif response_options == "classify":
            data = classify_image(image)
            if os.path.exists("images.jpg"):
                os.remove("images.jpg")
            return data
    except ValueError as e:
        e = "Error! Please upload a valid image type."
        return e
