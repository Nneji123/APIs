import io
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from PIL import Image
from utility import caption

sys.path.append(os.path.abspath(os.path.join("..", "config")))


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
        image = cv2.imread("image.jpg")
        images = description = caption("image.jpg")
        #cv2.imwrite("output.jpg", images)
        return images
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals
