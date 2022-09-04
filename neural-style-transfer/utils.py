import os
import warnings
from datetime import datetime

import cv2
import numpy as np
import paddlehub as hub
from PIL import Image

warnings.filterwarnings("ignore")


os.system("hub install stylepro_artistic==1.0.1")
stylepro_artistic = hub.Module(name="stylepro_artistic")


def inference():
    image_style = Image.open("image.jpg").convert("RGB")
    image_style = np.array(image_style)
    image_style = image_style[:, :, ::-1].copy()
    content_image = image_style
    start = datetime.now()

    result = stylepro_artistic.style_transfer(
        images=[{"content": content_image, "styles": [cv2.imread("image2.jpg")]}]
    )

    print(f"Time spent at Style Transfer: {datetime.now()-start}")
    img = Image.fromarray(np.uint8(result[0]["data"])[:, :, ::-1]).convert("RGB")
    img.resize((512, 512), Image.ANTIALIAS).save("output.jpg")
    return "output.jpg"
