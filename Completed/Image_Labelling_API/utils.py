import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import pycocotools.mask as mask_util
from matplotlib.colors import hsv_to_rgb
from onnx import numpy_helper
from PIL import Image


def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize(
        (int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR
    )

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype("float32")

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math

    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, : image.shape[1], : image.shape[2]] = image
    image = padded_image

    return image


# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
sess = rt.InferenceSession("FasterRCNN-10.onnx")

outputs = sess.get_outputs()


classes = [line.rstrip("\n") for line in open("coco_classes.txt")]


def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.7):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12, 9))
    image = np.array(image)
    ax.imshow(image)

    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor="b",
                facecolor="none",
            )
            ax.annotate(
                classes[label] + ":" + str(np.round(score, 2)),
                (box[0], box[1]),
                color="w",
                fontsize=12,
            )
            ax.add_patch(rect)
    myclasses = []
    for label, score in zip(labels, scores):
        if score > score_threshold:
            names = f"{str(classes[label])}, confidence: {str(np.round(score, 2))}"
            myclasses.append(names)

    plt.axis("off")
    plt.savefig("output.jpg", bbox_inches="tight")
    return myclasses


def inference(img):
    input_image = Image.open(img)
    input_tensor = preprocess(input_image)

    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name

    boxes, labels, scores = sess.run(output_names, {input_name: input_tensor})
    display_objdetect_image(input_image, boxes, labels, scores)

    return "output.jpg"


def get_label(img):
    input_image = Image.open(img)
    input_tensor = preprocess(input_image)

    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name

    boxes, labels, scores = sess.run(output_names, {input_name: input_tensor})
    name = display_objdetect_image(input_image, boxes, labels, scores)
    return name
