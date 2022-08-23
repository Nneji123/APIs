import io
import os
import sys

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import text2emotion as te

sys.path.append(os.path.abspath(os.path.join("..", "config")))
import nltk
nltk.download('omw-1.4')

app = FastAPI(
    title="Text2Emotion API",
    description="""An API for detecting the emotion in a block of text.""",
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
    Text2Emotion API 📚
    Note: add "/redoc" to get the complete documentation.
    """
    return note


class Emotion(BaseModel):
    doc: str

# endpoint for just enhancing the image
@app.post("/emotion")
async def get_emotion(data: Emotion):
    emo_text = te.get_emotion(data.doc)
    return emo_text