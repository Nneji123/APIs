import warnings

from transformers import (
    AutoTokenizer,
    ImageClassificationPipeline,
    PerceiverFeatureExtractor,
    PerceiverForImageClassificationConvProcessing,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTFeatureExtractor,
    ViTForImageClassification,
)

warnings.filterwarnings("ignore")


config = ViTConfig(num_hidden_layers=12, hidden_size=768)
model = ViTForImageClassification(config)


feature_extractor = ViTFeatureExtractor()

# the following gets called by classify_image()
feature_extractor = PerceiverFeatureExtractor.from_pretrained(
    "deepmind/vision-perceiver-conv"
)
model = PerceiverForImageClassificationConvProcessing.from_pretrained(
    "deepmind/vision-perceiver-conv"
)
image_pipe = ImageClassificationPipeline(
    model=model, feature_extractor=feature_extractor
)


def self_caption(image):
    repo_name = "ydshieh/vit-gpt2-coco-en"
    test_image = image
    feature_extractor2 = ViTFeatureExtractor.from_pretrained(repo_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model2 = VisionEncoderDecoderModel.from_pretrained(repo_name)
    pixel_values = feature_extractor2(test_image, return_tensors="pt").pixel_values
    # autoregressively generate text (using beam search or other decoding strategy)
    generated_ids = model2.generate(
        pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True
    )

    # decode into text
    preds = tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    preds = " ".join(preds)
    return {"caption": preds}


def classify_image(image):
    results = image_pipe(image)
    # convert to format Gradio expects
    output = {}
    for prediction in results:
        predicted_label = prediction["label"]
        score = prediction["score"]
        output[predicted_label] = score
    return output
