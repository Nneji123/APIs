import os
if os.path.exists('models/model.onnx'):
    pass
else:
    os.system("wget -O models/model.onnx https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx")