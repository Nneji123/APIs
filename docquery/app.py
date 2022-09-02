import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from utils import process_question, load_document
from pdf2image.exceptions import (PDFInfoNotInstalledError, PDFPageCountError,
                                  PDFSyntaxError)
from pydantic import BaseModel

#print(process_question("What is the job description?", load_document("filename.pdf")))

app = FastAPI(
    title="Document Query API",
    description="""An API for queirying documents.""",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str
    #image: bytes


@app.get("/", response_class=PlainTextResponse, tags=["home"])
async def home():
    note = """
    Document Query API 📚
    An API querying documents.
    Note: add "/redoc" to get the complete documentation.
    """
    return note


@app.post("/query-document")
async def get_document(data:Query, file: UploadFile = File(...)):
    files = await file.read()
    # save the file
    filename = "filename.pdf"
    with open(filename, "wb+") as f:
        f.write(files)
    # open the file and return the file name
    try:
        data = process_question(data.query, load_document("filename.pdf"))
        return data, F
    except (PDFInfoNotInstalledError, PDFPageCountError,
                                  PDFSyntaxError) as e:
        return "Unable to parse document! Please upload a valid PDF file."

@app.post("/query-image")
async def get_image(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("images.jpg", img)
    try:
        data = classify_image("images.jpg")
        return data
    except (ValueError) as e:
        vals = "Error! Please upload a valid image type."
        return vals 
