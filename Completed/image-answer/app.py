import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from PIL import Image
from pydantic import BaseModel
from utils import answer_question

app = FastAPI(
    title="Image Answering API",
    description="""Image Captioning API is a fastapi server that can answer questions about images.""",
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
    Image Answering API!
    This is a fastapi server that can answer questions about images.
    Note: add "/redoc" to get the complete documentation.
    """
    return note


class Question(BaseModel):
    question: str


@app.post("/answer-image")
async def get_image(question: str, file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    try:
        image = Image.open("image.jpg")
        # questions = data.question
        answer = answer_question(image=image, text=question)
        if os.path.exists("image.jpg"):
            os.remove("image.jpg")
        return answer
    except ValueError as e:
        e = "Error! Please upload a valid image type."
        return e
