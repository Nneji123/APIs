import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PIL import Image, ImageDraw
import traceback


import torch
from docquery.pipeline import get_pipeline
from docquery.document import load_bytes, load_document, ImageDocument


def ensure_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


CHECKPOINTS = {
    "LayoutLMv1 🦉": "impira/layoutlm-document-qa",
    "Donut 🍩": "naver-clova-ix/donut-base-finetuned-docvqa",
}

PIPELINES = {}


def construct_pipeline(model):
    global PIPELINES
    if model in PIPELINES:
        return PIPELINES[model]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ret = get_pipeline(checkpoint=CHECKPOINTS[model], device=device)
    PIPELINES[model] = ret
    return ret


def run_pipeline(model, question, document, top_k):
    pipeline = construct_pipeline(model)
    return pipeline(question=question, **document.context, top_k=top_k)


# TODO: Move into docquery
# TODO: Support words past the first page (or window?)
def lift_word_boxes(document, page):
    return document.context["image"][page][1]


def expand_bbox(word_boxes):
    if len(word_boxes) == 0:
        return None

    min_x, min_y, max_x, max_y = zip(*[x[1] for x in word_boxes])
    min_x, min_y, max_x, max_y = [min(min_x), min(min_y), max(max_x), max(max_y)]
    return [min_x, min_y, max_x, max_y]


# LayoutLM boxes are normalized to 0, 1000
def normalize_bbox(box, width, height, padding=0.005):
    min_x, min_y, max_x, max_y = [c / 1000 for c in box]
    if padding != 0:
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(max_x + padding, 1)
        max_y = min(max_y + padding, 1)
    return [min_x * width, min_y * height, max_x * width, max_y * height]


examples = [
    [
        "invoice.png",
        "What is the invoice number?",
    ],
    [
        "contract.jpeg",
        "What is the purchase amount?",
    ],
    [
        "statement.png",
        "What are net sales for 2020?",
    ],
]

question_files = {
    "What are net sales for 2020?": "statement.pdf",
}


def process_path(path):
    error = None
    if path:
        try:
            document = load_document(path)
            return document
        except Exception as e:
            traceback.print_exc()
            error = str(e)
    return None

def process_upload(file):
    if file:
        return process_path(file.name)
    else:
        return None


colors = ["#64A087", "green", "black"]


def process_question(question, document, model=list(CHECKPOINTS.keys())[0]):
    if document is None:
        return None, None, None

    text_value = None
    predictions = run_pipeline(model, question, document, 3)
    pages = [x.copy().convert("RGB") for x in document.preview]
    for i, p in enumerate(ensure_list(predictions)):
        if i == 0:
            text_value = p["answer"]
        else:
            # Keep the code around to produce multiple boxes, but only show the top
            # prediction for now
            break

        if "start" in p and "end" in p:
            image = pages[p["page"]]
            draw = ImageDraw.Draw(image, "RGBA")
            x1, y1, x2, y2 = normalize_bbox(
                expand_bbox(
                    lift_word_boxes(document, p["page"])[p["start"] : p["end"] + 1]
                ),
                image.width,
                image.height,
            )
            draw.rectangle(((x1, y1), (x2, y2)), fill=(0, 255, 0, int(0.4 * 255)))
            draw.save("output.jpg")
    return pages, predictions, text_value


def load_example_document(img, question, model):
    if img is not None:
        if question in question_files:
            print("using document")
            document = load_document(question_files[question])
        else:
            document = ImageDocument(Image.fromarray(img))
        preview, answer, answer_text = process_question(question, document, model)
        return document, question, preview, answer, answer_text
    else:
        return None, None, None, None, None

