
from transformers import DetrFeatureExtractor, DetrForSegmentation
feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')
print("Model Downloaded!")
