import os

if os.path.exists("FasterRCNN-10.onnx"):
    print("FasterRCNN-10.onnx exists")
else:
    os.system(
        "wget https://github.com/AK391/models/raw/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx"
    )
    print("Downloaded FasterRCNN-10.onnx")
