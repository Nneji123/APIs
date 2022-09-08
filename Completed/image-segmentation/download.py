import os

from transformers import ViltForQuestionAnswering, ViltProcessor

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
os.system(
    "wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth -P /root/.cache/torch/hub/checkpoints/"
)

print("Model Downloaded!")
