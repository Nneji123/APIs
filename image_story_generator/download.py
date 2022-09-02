from transformers import ViTConfig, ViTForImageClassification
from transformers import ViTFeatureExtractor
from transformers import ImageClassificationPipeline, PerceiverForImageClassificationConvProcessing, PerceiverFeatureExtractor
from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore")


config = ViTConfig(num_hidden_layers=12, hidden_size=768)
model = ViTForImageClassification(config)

feature_extractor = ViTFeatureExtractor()

#the following gets called by classify_image() 
feature_extractor = PerceiverFeatureExtractor.from_pretrained("deepmind/vision-perceiver-conv")
model = PerceiverForImageClassificationConvProcessing.from_pretrained("deepmind/vision-perceiver-conv")
image_pipe = ImageClassificationPipeline(model=model, feature_extractor=feature_extractor)

print("model loaded")