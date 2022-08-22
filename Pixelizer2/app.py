import io
import os
import sys

import cv2
import numpy as np
from blur_the_face_video import face_blurring, face_pixelate
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

sys.path.append(os.path.abspath(os.path.join("..", "config")))


app = FastAPI(
    title="Face Pixelizer API",
    description="""An API for Automatic Face Pixellization of Images""",
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
    Face Pixelizer API 📚
    An API for Automatic Face Pixellization of Images!
    Note: add "/redoc" to get the complete documentation.
    """
    return note


# endpoint for just enhancing the image
@app.post("/pixelize")
async def face_pixelize(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        image = cv2.imread("image.jpg")
        images = face_blurring(image, blur_type="pixelate", blocks=20)
        cv2.imwrite("out.jpg", images)
        return FileResponse("out.jpg", media_type="image/jpg")
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals


@app.post("/blur")
async def face_blur(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        image = cv2.imread("image.jpg")
        images = face_blurring(image, blur_type="simple", blocks=20)
        cv2.imwrite("output.jpg", images)
        return FileResponse("output.jpg", media_type="image/jpg")
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals
