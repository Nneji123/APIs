import io

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from utils import self_caption


app = FastAPI(
    title="Image Captioning API",
    description="""Image Captioning API is a fastapi server that can generate captions for images.""",
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
    Image Captioning API 📚
    An API that can generate captions for images.
    Note: add "/redoc" to get the complete documentation.
    """
    return note


# @app.post("/generate", response_class=PlainTextResponse, tags=["generate"])
# async def get_document(file: UploadFile = File(...)):
#     files = await file.read()
#     # save the file
#     filename = "filename.pdf"
#     with open(filename, "wb+") as f:
#         f.write(files)
#     # open the file and return the file name
#     try:
#         data = convert_pdf("filename.pdf")
#         return data
#     except (PDFInfoNotInstalledError, PDFPageCountError,
#                                   PDFSyntaxError) as e:
#         return "Unable to parse document! Please upload a valid PDF file."

@app.post("/caption-image")
async def get_image(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("images.jpg", img)
    try:
        image = Image.open("images.jpg")
        data = self_caption(image)
        return data
    except ValueError as e:
        e = "Error! Please upload a valid image type."
        return e 
