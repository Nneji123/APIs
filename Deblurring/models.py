import os

from PIL import Image

if os.path.exists("experiments/pretrained_models/deblur_GoPro_CMFNet.pth"):
    pass
else:
    os.system(
        "wget https://github.com/FanChiMao/CMFNet/releases/download/v0.0/deblur_GoPro_CMFNet.pth -P experiments/pretrained_models"
    )


def inference(img):
    if os.path.exists("test/"):
        pass
    else:
        os.system("mkdir test")
    basewidth = 512
    wpercent = basewidth / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.BILINEAR)
    img.save("test/1.png", "PNG")
    os.system(
        "python main_test_CMFNet.py --input_dir test --weights experiments/pretrained_models/deblur_GoPro_CMFNet.pth"
    )
    return "results/1.png"


image = Image.open("./images/Blur1.png")
inference(image)
print("Downloaded model")

# title = "Compound Multi-branch Feature Fusion for Image Restoration (Deblur)"
# description = "Gradio demo for CMFNet. CMFNet achieves competitive performance on three tasks: image deblurring, image dehazing and image deraindrop. Here, we provide a demo for image deblur. To use it, simply upload your image, or click one of the examples to load them. Reference from: https://huggingface.co/akhaliq"
# article = "<p style='text-align: center'><a href='https://' target='_blank'>Compound Multi-branch Feature Fusion for Real Image Restoration</a> | <a href='https://github.com/FanChiMao/CMFNet' target='_blank'>Github Repo</a></p> <center><img src='https://visitor-badge.glitch.me/badge?page_id=52Hz_CMFNet_deblurring' alt='visitor badge'></center>"

# examples = [['images/Blur1.png'], ['images/Blur2.png'], ['images/Blur5.png'],]
# gr.Interface(
#     inference,
#     [gr.inputs.Image(type="pil", label="Input")],
#     gr.outputs.Image(type="file", label="Output"),
#     title=title,
#     description=description,
#     article=article,
#     allow_flagging=False,
#     allow_screenshot=False,
#     examples=examples
# ).launch(debug=True)
