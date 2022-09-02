import io
import os

import cv2
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from utils import process_question, load_document, process_question_image
from pdf2image.exceptions import (PDFInfoNotInstalledError, PDFPageCountError,
                                  PDFSyntaxError)


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

# class Query(BaseModel):
#     query: str
#     #image: bytes


@app.get("/", response_class=PlainTextResponse, tags=["home"])
async def home():
    note = """
    Document Query API 📚
    An API querying documents.
    Note: add "/redoc" to get the complete documentation.
    """
    return note

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


@app.post("/query-document", tags=["query"])
async def get_document(type_of_response:str, question:str, file: UploadFile = File(...), ):
    files = await file.read()
    # save the file
    filename = "filename.pdf"
    with open(filename, "wb+") as f:
        f.write(files)
    # open the file and return the file name
    try:
        data = process_question(question, load_document("filename.pdf"))
        if type_of_response == "image":
            return FileResponse("output.jpg", media_type="image/jpg")
        elif type_of_response == "text":
            return data
        delete_file("filename.pdf")
        delete_file("output.jpg")
    except (PDFInfoNotInstalledError, PDFPageCountError,
                                  PDFSyntaxError) as e:
        e = "Unable to parse document! Please upload a valid PDF file."
        return e

@app.post("/query-image", tags=["query image"])
async def get_image(type_of_response:str, question:str, file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("images.jpg", img)
    try:
        im = Image.open("images.jpg")
        image = np.asarray(im)
        data = process_question_image(image, question)
        if type_of_response == "image":
            return FileResponse("output.jpg", media_type="image/jpg")
        elif type_of_response == "text":
            return data
        delete_file("images.jpg")
        delete_file("output.jpg")
    except ValueError as e:
        e = "Error! Please upload a valid image type."
        return e
    
