import io
import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from utils import check

sys.path.append(os.path.abspath(os.path.join("..", "config")))

app = FastAPI(
    title="Movie Recommender API",
    description="""An API for getting movie recommendations.""",
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
    Movie Recommender API
    Note: add "/redoc" to get the complete documentation.
    """
    return note


class MovieName(BaseModel):
    movie: str


# endpoint for just enhancing the image
@app.post("/movie", response_class=PlainTextResponse)
async def get_movie(data: MovieName):
    text = check(data.movie)
    return text


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
