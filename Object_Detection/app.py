import io
import os
import sys

import cv2
import numpy as np
from object_detection import *
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

sys.path.append(os.path.abspath(os.path.join("..", "config")))


app = FastAPI(
    title="Object Detection API",
    description="""An API for Detecting Objects in images.""",
)


@app.get("/")
async def running():
    note = """
    Object Detection API 📚
    An API for detecting objects in images!
    Note: add "/redoc" to get the complete documentation.
    """
    return note


# endpoint for just enhancing the image
@app.post("/detect-object")
async def detect_object(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    
    
    try:
        input_size = 416
        original_image = cv2.imread('image.jpg')
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        sess = rt.InferenceSession("models/model.onnx")

        outputs = sess.get_outputs()
        output_names = list(map(lambda output: output.name, outputs))
        input_name = sess.get_inputs()[0].name

        detections = sess.run(output_names, {input_name: image_data})
        print("Output shape:", list(map(lambda detection: detection.shape, detections)))

        
        ANCHORS = "models/anchors.txt"
        STRIDES = [8, 16, 32]
        XYSCALE = [1.2, 1.1, 1.05]

        ANCHORS = get_anchors(ANCHORS)
        STRIDES = np.array(STRIDES)

        pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
        bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
        bboxes = nms(bboxes, 0.213, method='nms')
        image = draw_bbox(original_image, bboxes)
        cv2.imwrite("output.jpg",image)
        return FileResponse("output.jpg", media_type="image/jpg")
    except ValueError:
        vals = "Error! Please upload a valid image type."
        return vals
