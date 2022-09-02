from PIL import Image

from transformers import ViTConfig, ViTForImageClassification
from transformers import ViTFeatureExtractor
from transformers import ImageClassificationPipeline, PerceiverForImageClassificationConvProcessing, PerceiverFeatureExtractor
from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore")


config = ViTConfig(num_hidden_layers=12, hidden_size=768)
model = ViTForImageClassification(config)

#print(config)

feature_extractor = ViTFeatureExtractor()

#the following gets called by classify_image() 
feature_extractor = PerceiverFeatureExtractor.from_pretrained("deepmind/vision-perceiver-conv")
model = PerceiverForImageClassificationConvProcessing.from_pretrained("deepmind/vision-perceiver-conv")
image_pipe = ImageClassificationPipeline(model=model, feature_extractor=feature_extractor)

# def create_story(text_seed):
#     #tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     #model = AutoModelForCausalLM.from_pretrained("gpt2")
    
#     #eleutherAI gpt-3 based
#     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
#     model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

#     # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
#     model.config.pad_token_id = model.config.eos_token_id

#     #input_prompt = "It might be possible to"
#     input_prompt = text_seed
#     input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

#     # instantiate logits processors
#     logits_processor = LogitsProcessorList(
#       [
#           MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),
#       ]
#     )
#     stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=100)])

#     outputs = model.greedy_search(
#       input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
#     )

#     result_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return result_text






def self_caption(image):
    repo_name = "ydshieh/vit-gpt2-coco-en"
    test_image = image
    feature_extractor2 = ViTFeatureExtractor.from_pretrained(repo_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model2 = VisionEncoderDecoderModel.from_pretrained(repo_name)
    pixel_values = feature_extractor2(test_image, return_tensors="pt").pixel_values
    # autoregressively generate text (using beam search or other decoding strategy)
    generated_ids = model2.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True)
    
    # decode into text
    preds = tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    pred_keys = ["Prediction"]
    pred_value = preds

    pred_dictionary = dict(zip(pred_keys, pred_value))
    #return(pred_dictionary)
    preds = ' '.join(preds)
    return {'caption': preds, "values": pred_dictionary}


def classify_image(image):
    results = image_pipe(image)
    # convert to format Gradio expects
    output = {}
    for prediction in results:
      predicted_label = prediction['label']
      score = prediction['score']
      output[predicted_label] = score
    print("OUTPUT")
    print(output)
    return output

# image = Image.open("cats.jpg")
# classify_image(image)
# def get_captions(image_path:str):
#     caption = Image.open(image_path)
#     val = self_caption(caption)
#     print(self_caption(image_path))
#     return val
