from torchvision import transforms
from PIL import Image
from models.ofa import OFAModel
from tasks.mm_tasks.caption import CaptionTask
from utils.eval_utils import eval_step
from fairseq import checkpoint_utils
from fairseq import utils, tasks
import numpy as np
import torch
import os

os.system('cd fairseq;'
          'pip install --use-feature=in-tree-build ./; cd ..')
os.system('ls -l')


# Register caption task
tasks.register_task('caption', CaptionTask)
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

os.system('wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/caption_large_best_clean.pt; '
          'mkdir -p checkpoints; mv caption_large_best_clean.pt checkpoints/caption.pt')

# Load pretrained ckpt & config
overrides = {"bpe_dir": "utils/BPE", "eval_cider": False, "beam": 5,
             "max_len_b": 16, "no_repeat_ngram_size": 3, "seed": 7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths('checkpoints/caption.pt'),
    arg_overrides=overrides
)

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size,
                      cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


# Construct input for caption task
def construct_sample(image: Image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?",
                           append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor(
        [s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


# Function for image captioning
def image_caption(Image):
    sample = construct_sample(Image)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)
    return result[0]['caption']


# title = "OFA-Image_Caption"
# description = "Gradio Demo for OFA-Image_Caption. Upload your own image or click any one of the examples, and click " \
#               "\"Submit\" and then wait for the generated caption.  "
# article = "<p style='text-align: center'><a href='https://github.com/OFA-Sys/OFA' target='_blank'>OFA Github " \
#           "Repo</a></p> "
# examples = [['beatles.jpeg'], ['aurora.jpeg'], [
#     'good_luck.png'], ['pokemons.jpg'], ['donuts.jpg']]
# io = gr.Interface(fn=image_caption, inputs=gr.inputs.Image(type='pil'), outputs=gr.outputs.Textbox(label="Caption"),
#                   title=title, description=description, article=article, examples=examples,
#                   allow_flagging=False, allow_screenshot=False)
# io.launch(cache_examples=True)