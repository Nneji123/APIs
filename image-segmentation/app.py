import io
import os

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse
from utils import predict_mask

app = FastAPI(
    title="Image Segmentation API",
    description="""Image Captioning API is a fastapi server that can segment images.""",
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
    Image Segmentation API!
    This is a fastapi server that can segment images.
    Note: add "/redoc" to get the complete documentation.
    """
    return note


@app.post("/segment-image")
async def get_image(confidence:int, file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        image = Image.open("image.jpg")
        image = np.asarray(image)
        #questions = data.question
        predict_mask(im=image,confidence=confidence)
        return FileResponse("output.jpg", media_type="image/jpeg")
    except ValueError as e:
        e = "Error! Please upload a valid image type."
        return e 

