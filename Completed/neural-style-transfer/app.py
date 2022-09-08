import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from utils import inference

app = FastAPI(
    title="Neural Style Transfer API",
    description="""An API for transferring styles of images.""",
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
    Neural Style Transfer API!
    Note: add "/redoc" to get the complete documentation.
    """
    return note


@app.post(
    "/style-transfer",
    tags=["style-transfer"],
    description="Transfer style of an image. input_image is the image to be styled and style_image is the image to be used as style.",
)
async def get_image(
    input_image: UploadFile = File(...), style_image: UploadFile = File(...)
):

    contents = io.BytesIO(await input_image.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    contents2 = io.BytesIO(await style_image.read())
    file_bytes2 = np.asarray(bytearray(contents2.read()), dtype=np.uint8)
    img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
    cv2.imwrite("image2.jpg", img2)
    try:
        inference()
        return FileResponse("output.jpg", media_type="image/jpeg")
    except ValueError as e:
        e = "Error! Please upload a valid image type."
        return e
