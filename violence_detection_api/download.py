import os
import gradio as gr
from model import classify_image
from PIL import Image
import numpy as np
if os.path.exists("bestmodel.h5"):
    print("This file exists!")
else:
    os.system("wget -O bestmodel.h5 https://huggingface.co/spaces/SIB/violence_api/resolve/main/bestmodel.h5")
    print("File downloaded!")

image = Image.open("images.jpeg")
image = image.resize((224, 224), Image.ANTIALIAS)
image = np.array(image)
print(classify_image(image))

