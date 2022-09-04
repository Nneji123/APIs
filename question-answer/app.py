import os
import PyPDF2
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
    return note

class Question(BaseModel):
    question: str
    context: str
    
    class Config:
        schema_extra = {
            "example": {
            "question": "What is the name of the first president of the United States?",
            "context": "George Washington was the first president of the United States of America."
        }
        }

@app.post("/question-answer", tags=["question-answer"])
async def get_answer_text(data: Question):
    try:
        answer = get_answer(question = data.question, context = data.context)
        if answer == "":
            return {"answer": "No answer found"}
        return answer
    except Exception as e:
        e = "Cannot answer context."
        return e

@app.post("/question-answer-file", tags=["text-file"], description="Upload a text file to answer questions.")
async def get_answer_textfile(myquestion:str, file: UploadFile = File(...)):
    files = await file.read()
    # save the file
    filename = "file.txt"
    with open(filename, "wb+") as f:
        f.write(files)
    try:
        with open(filename, "r") as f:
            txt = f.read()
        answer = get_answer(question = myquestion, context = txt)
        if os.path.exists("file.txt"):
            os.remove("file.txt")
        if answer == "":
            answer = "Cannot answer context."
        return answer
    except Exception as e:
        e = "Cannot answer context."
        return e
    
    
@app.post("/question-answer-pdf", tags=["pdf-file"], description="Upload a pdf file to answer questions.")
async def get_answer_pdf(myquestion:str, file: UploadFile = File(...)):
    files = await file.read()
    # save the file
    filename = "file.pdf"
    with open(filename, "wb+") as f:
        f.write(files)
    try:
        with open("file.pdf", "rb") as pdf:
            pdfReader = PyPDF2.PdfFileReader(pdf)
            pageObj = pdfReader.getPage(0)
            pdftext=pageObj.extractText()
        answer = get_answer(question = myquestion, context = pdftext)
        if os.path.exists("file.pdf"):
            os.remove("file.pdf")
        if answer == "":
            answer = "Cannot answer context."
        return answer
    except Exception as e:
        e = "Cannot answer context."
        return e