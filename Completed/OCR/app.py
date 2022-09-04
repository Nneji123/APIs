import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from PIL import Image
from utils import *

app = FastAPI(
    title="OCR(Optical Character Recognition) API",
    description="""An API for OCR(Optical Character Recognition)!""",
    version="0.0.1",
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
    OCR(Optical Character Recognition) API
    An API for OCR(Optical Character Recognition)!
    Note: add "/redoc" to get the complete documentation.
    """
    return note


@app.post("/ocr")
async def get_ocr_image(file: UploadFile = File(...), lang: str = "en"):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        inference("image.jpg", lang)
        print("Making prediction...")
        return FileResponse("output.jpg", media_type="image/jpg")
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals


@app.post("/text-ocr")
async def get_text_ocr_image(file: UploadFile = File(...), lang: str = "en") -> dict:

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        image = Image.open("image.jpg")
        text = get_text(image, lang)
        print("Making prediction...")
        return text
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals
