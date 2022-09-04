import io
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from myfunc import image_caption
from PIL import Image

app = FastAPI(
    title="Image Caption Generator API",
    description="""An API for generating caption of images.""",
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
    Image Caption Generator API 📚
    An API for generating caption of images!
    Note: add "/redoc" to get the complete documentation.
    """
    return note


@app.post("/generate-caption")
async def generate_caption(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        image = Image.open("image.jpg")
        caption = image_caption(image)
        if os.path.exists("image.jpg"):
            os.remove("image.jpg")
        return caption
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals
