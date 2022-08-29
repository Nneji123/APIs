import io
import os
import sys

import cv2
import numpy as np
from pixelate import pixel
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

sys.path.append(os.path.abspath(os.path.join("..", "config")))


app = FastAPI(
    title="Pixelizer API",
    description="""An API for recognising vehicle number plates in images and video.""",
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
    Pixelizer API"
    An API for pixelizing images.
    Note: add "/redoc" to get the complete documentation.
    """
    return note


@app.post("/pixelate")
async def pixelate_image(file: UploadFile = File(...), downsample: int = 14, palette: int = 7, depth: int = 1, upscale: int = 14):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        # image = Image.open("image.jpg")
        # image = np.array(image)
        img = pixel("image.jpg", downsample, palette, depth, upscale)
        return FileResponse("output.jpg", media_type="image/jpg")
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals
