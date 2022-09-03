import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from transformers import TapasTokenizer, TFTapasForQuestionAnswering

model_name = "google/tapas-base-finetuned-wtq"
model = TFTapasForQuestionAnswering.from_pretrained(model_name)
tokenizer = TapasTokenizer.from_pretrained(model_name)
print("Model loaded")