import io
import os
import time

import cv2
import numpy as np
from colorizer import colorization
from fastapi import FastAPI, File, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

app = FastAPI(
    title="Image Colorizer API",
    description="""An API for colorizing images.""",
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
    Image Colorizer API 
    An API for colorizing images!
    Note: add "/redoc" to get the complete documentation.
    """
    return note


# endpoint for just enhancing the image
@app.post("/colorize")
async def face_pixelize(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        image = "image.jpg"
        images = colorization(image)
        # cv2.imwrite("out.jpg", images)
        return FileResponse("output.jpg", media_type="image/jpg")
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals
