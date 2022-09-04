import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from models import inference
from PIL import Image

app = FastAPI(
    title="Image Deblur API",
    description="""An API for deblurring images.""",
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
    Image Deblur API 
    An API for debluring images!
    Note: add "/redoc" to get the complete documentation.
    """
    return note


# endpoint for just enhancing the image
@app.post("/deblur")
async def deblur_image(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        image = Image.open("image.jpg")
        images = inference(image)
        print("Making prediction...")
        return FileResponse("results/1.png", media_type="image/png")
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals
