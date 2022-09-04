import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from PIL import Image
from utils import inference

app = FastAPI(
    title="Cartoonizer API",
    description="""An API for generating cartoonized images.""",
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
    Cartoonizer API 
    An API for generating cartoonized images!
    Note: add "/redoc" to get the complete documentation.
    """
    return note


# endpoint for just enhancing the image
@app.post("/cartoonize")
async def cartoonize_image(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        image = Image.open("image.jpg")
        image = np.asarray(image)
        inference(image)
        return FileResponse("output.jpg", media_type="image/jpg")
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals
