import warnings

import easyocr
import pandas as pd
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")


def draw_boxes(image, bounds, color="yellow", width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image


def get_text(image, lang):
    reader = easyocr.Reader([lang])
    bounds = reader.readtext(image)
    vals = pd.DataFrame(bounds).iloc[:, 1:]
    vals = pd.DataFrame.to_json(vals)
    return vals


def inference(img, lang):
    reader = easyocr.Reader([lang])
    bounds = reader.readtext(img)
    im = Image.open(img)
    draw_boxes(im, bounds)
    im.save("output.jpg")


inference("image.jpg", "en")
print("Making prediction...")
