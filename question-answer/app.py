from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from utils import get_answer
from pydantic import BaseModel

app = FastAPI(
    title="Question Answering API",
    description="""Question Answering API is a fastapi server that can answer questions from a given text file or text input.""",
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
    Question Answering API!
    This is a fastapi server that can answerr questions.
    Note: add "/redoc" to get the complete documentation.
    """
    return 

class Question(BaseModel):
    question: str
    context: str


@app.post("/question-answer")
async def get_answer(data: Question):
    try:
        answer = get_answer(data.question,data.context)
        return answer
    except Exception as e:
        e = "Cannot answer context."
        return e

@app.post("/question-answer-file")
async def get_answer_file(data:Question, file: UploadFile = File(...)):
    contents = await file.read()
    try:
        answer = get_answer(data.question,contents)
        return answer
    except Exception as e:
        e = "Cannot answer context."
        return e
