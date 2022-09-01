import os
from pdf2image import convert_from_bytes, convert_from_path
from pdf2image.exceptions import (PDFInfoNotInstalledError, PDFPageCountError,
                                  PDFSyntaxError)
from PIL import Image
from transformers import pipeline

pipe = pipeline(task="image-classification", 
                model="microsoft/dit-base-finetuned-rvlcdip")

def convert_pdf(filename: str="filename.pdf"):
    images = convert_from_path("filename.pdf", dpi=500, single_file=True, jpegopt="optimized", output_file="image.jpg",output_folder="images")
    #Saving pages in jpeg format
    for page in images:
        page.save('image.jpg', 'JPEG')
    print("Converted pdf")
    image = Image.open("image.jpg")
    caption = pipe(image)
    if os.path.exists("image.jpg"):
        os.remove("image.jpg")
    if os.path.exists("filename.pdf"):
        os.remove("filename.pdf")
    if os.path.exists("images/image.jpg.ppm"):
        os.remove("images/image.jpg.ppm")
    return caption

def classify_image(image: str="images.jpg"):
    images = Image.open(image)
    caption = pipe(images)
    if os.path.exists("images.jpg"):
        os.remove("images.jpg")
    return caption 


