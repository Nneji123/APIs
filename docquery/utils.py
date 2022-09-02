from typing import List
from docquery import document, pipeline
from PIL import Image, ImageDraw
import traceback

import torch
from docquery.document import ImageDocument

p = pipeline.get_pipeline()



def get_document_answers(filepath: str, query: str):
    doc = document.load_document(filepath)
    for q in query:
        print(q, p(question=q, **doc.context))
        answers = (p(question=q, **doc.context))
        return answers
    
def get_image_answers(filepath: str, query: str):
    img = Image.open(filepath)
    doc = ImageDocument(img)
    for q in query:
        print(q, p(question=q, **doc.context))
        answers = (p(question=q, **doc.context))
        return answers
    
print(get_document_answers("filename.pdf", "What is the job description?"))
