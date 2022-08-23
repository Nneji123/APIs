import io
import os
import sys

import cv2
import numpy as np
from prefix_clip import download_pretrained_model, generate_caption
from gpt2_story_gen import generate_story
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
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

from pydantic import BaseModel

class GenerateStory(BaseModel):
    model: str
    genre: str
    n_stories: int

    class Config:
        schema_extra = {
            "example": {
                "model" : "coco",
                "genre": 100,
                "n_stories" :1
            }
        }

coco_weights = 'coco_weights.pt'
conceptual_weights = 'conceptual_weights.pt'

@app.post("/generate-story")
async def generate_storys(pil_image, data: GenerateStory, use_beam_search=False, file: UploadFile = File(...)):
    if data.model.lower()=='coco':
        model_file = coco_weights
    elif data.model.lower()=='conceptual':
        model_file = conceptual_weights

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    pil_image = Image.open("image.jpg")

    image_caption = generate_caption(
        model_path=model_file,
        pil_image=pil_image,
        use_beam_search=use_beam_search,
    )
    story = generate_story(image_caption, pil_image, data.genre.lower(), data.n_stories)
    return story