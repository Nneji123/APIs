import io
import os
import sys

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from gpt_j.Basic_api import simple_completion
from gpt_j.gptj_api import Completion
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join("..", "config")))


app = FastAPI(
    title="Text Generation API",
    description="""An API for auto generating text from a prompt""",
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
    Text Generation API 📚
    An API for auto generating text from a prompt!
    Note: add "/redoc" to get the complete documentation.
    """
    return note


class TextGeneration(BaseModel):
    prompt: str
    max_length: int
    temperature: float
    top_probability: float
    top_k: int
    repetition: float

    class Config:
        schema_extra = {
            "example": {
                "prompt": "def perfect_square(num):",
                "max_length": 100,
                "temperature": 0.09,
                "top_probability": 1.0,
                "top_k": 40,
                "repetition": 0.216,
            }
        }


class AdvancedTextGeneration(BaseModel):
    prompt: str
    context: str
    max_tokens: int
    User: str
    Bot: str
    examples: dict
    max_length: int
    temperature: float
    top_probability: float
    top_k: int
    repetition: float

    class Config:
        schema_extra = {
            "example": {
                "prompt": "48/2",
                "context": "This is a calculator bot that will answer basic math questions",
                "max_length": 100,
                "User": "Student",
                "max_tokens": 50,
                "Bot": "Calculator",
                "examples": {
                    "5 + 5": "10",
                    "6 - 2": "4",
                    "4 * 15": "60",
                    "10 / 5": "2",
                    "144 / 24": "6",
                    "7 + 1": "8",
                },
                "temperature": 0.09,
                "top_probability": 1.0,
                "top_k": 40,
                "repetition": 0.216,
            }
        }


@app.post("/generate", response_class=PlainTextResponse, tags=["Simple Text"])
async def generate_text(data: TextGeneration):
    query = simple_completion(
        prompt=data.prompt,
        length=data.max_length,
        temp=data.temperature,
        top_p=data.top_probability,
        top_k=data.top_k,
        rep=data.repetition,
    )
    return query


@app.post("/completion", response_class=PlainTextResponse, tags=["Advanced Prompt"])
async def advanced_generate_text(data: AdvancedTextGeneration):
    context_setting = Completion(data.context, data.examples)
    response = context_setting.completion(
        prompt=data.prompt,
        user=data.User,
        bot=data.Bot,
        max_tokens=data.max_tokens,
        temperature=data.temperature,
        top_p=data.top_probability,
        top_k=data.top_k,
        rep=data.repetition,
    )
    return response
