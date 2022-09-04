
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def answer_question(image, text):
    encoding = processor(image, text, return_tensors="pt")
    
    # forward pass
    with torch.no_grad():
     outputs = model(**encoding)
     
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    predicted_answer = model.config.id2label[idx]
   
    return predicted_answer
 