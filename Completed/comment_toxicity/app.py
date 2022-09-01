import io
import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join("..", "config")))
from utils import score_comment
app = FastAPI(
    title="Comment Toxicity Classifier API",
    description="""An API for classifying and detecting toxicity of comment.""",
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
    Comment Toxicity Classifier API 
    Note: add "/redoc" to get the complete documentation.
    """
    return note


class Comment(BaseModel):
    comment: str


# endpoint for just enhancing the image
@app.post("/comment-classify", response_class=PlainTextResponse)
async def get_comment(data: Comment):
    text = score_comment(data.comment)
    return text

if __name__=="__main__":
    uvicorn.run(app, port=8000) 