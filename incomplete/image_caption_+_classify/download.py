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
repo_name = "ydshieh/vit-gpt2-coco-en"
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
feature_extractor2 = ViTFeatureExtractor.from_pretrained(repo_name)
tokenizer = AutoTokenizer.from_pretrained(repo_name)
model2 = VisionEncoderDecoderModel.from_pretrained(repo_name)

print("model loaded")
