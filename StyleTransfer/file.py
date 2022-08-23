import cv2
import numpy as np
from onnx import numpy_helper
import onnx
import os
from PIL import Image
import onnxruntime as rt
from scipy import special
import colorsys
import random
from resizeimage import resizeimage



orig_img = Image.open("test6.jpg")
img = resizeimage.resize_cover(orig_img, [224,224], validate=False)
img_ycbcr = img.convert('YCbCr')
img_y_0, img_cb, img_cr = img_ycbcr.split()
img_ndarray = np.asarray(img_y_0)

img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
img_5 = img_4.astype(np.float32) / 255.0
img_5

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
ort_session = rt.InferenceSession("model.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: img_5}
output_name = ort_session.get_outputs()[0].name 
ort_outs = ort_session.run([output_name], [ort_inputs])
img_out_y = ort_outs[0]



# output_name = session.get_outputs()[0].name
# input_name = session.get_inputs()[0].name
# result = session.run([output_name], {input_name: x})[0][0] 

#Postprocessing Image

img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
cv2.imwrite('file.jpg',final_img)



