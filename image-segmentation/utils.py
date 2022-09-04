from transformers import DetrFeatureExtractor, DetrForSegmentation
from PIL import Image
import numpy as np
import torch

import itertools

import seaborn as sns

#######################################
# get models from hugging face
feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')

def predict_mask(im, confidence=85):
    image = Image.fromarray(im) # im: numpy array 3d: 480, 640, 3: to PIL Image
    image = image.resize((200,200)) #  PIL image # could I upsample output instead? better?

    # encoding is a dict with pixel_values and pixel_mask
    encoding = feature_extractor(images=image, return_tensors="pt") #pt=Pytorch, tf=TensorFlow
    outputs = model(**encoding) # odict with keys: ['logits', 'pred_boxes', 'pred_masks', 'last_hidden_state', 'encoder_last_hidden_state']
    logits = outputs.logits # torch.Size([1, 100, 251]); class logits? but  why 251?
    bboxes = outputs.pred_boxes
    masks = outputs.pred_masks # torch.Size([1, 100, 200, 200]); mask logits? for every pixel, score in each of the 100 classes? there is a mask per class

    # keep only the masks with high confidence?--------------------------------
    # compute the prob per mask (i.e., class), excluding the "no-object" class (the last one)
    prob_per_query = outputs.logits.softmax(-1)[..., :-1].max(-1)[0] # why logits last dim 251?
    # threshold the confidence
    keep = prob_per_query > confidence/100.0

    # postprocess the mask (numpy arrays)
    label_per_pixel = torch.argmax(masks[keep].squeeze(),dim=0).detach().numpy() # from the masks per class, select the highest per pixel
    color_mask = np.zeros(image.size+(3,))
    palette = itertools.cycle(sns.color_palette())
    for lbl in np.unique(label_per_pixel): #enumerate(palette()):
        color_mask[label_per_pixel==lbl,:] = np.asarray(next(palette))*255 #color

    # color_mask = np.zeros(image.size+(3,))
    # for lbl, color in enumerate(ade_palette()):
    #     color_mask[label_per_pixel==lbl,:] = color

    # Show image + mask
    pred_img = np.array(image.convert('RGB'))*0.25 + color_mask*0.75
    pred_img = pred_img.astype(np.uint8) 
    pred_img = Image.fromarray(pred_img)
    pred_img.save('output.jpg')

    return "pred_img"
