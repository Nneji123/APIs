import gradio as gr

from prefix_clip import download_pretrained_model, generate_caption
from gpt2_story_gen import generate_story

coco_weights = 'coco_weights.pt'
conceptual_weights = 'conceptual_weights.pt'
download_pretrained_model('coco', file_to_save=coco_weights)
download_pretrained_model('conceptual', file_to_save=conceptual_weights)


def main(pil_image, genre, model, n_stories, use_beam_search=False):
    if model.lower()=='coco':
        model_file = coco_weights
    elif model.lower()=='conceptual':
        model_file = conceptual_weights

    image_caption = generate_caption(
        model_path=model_file,
        pil_image=pil_image,
        use_beam_search=use_beam_search,
    )
    story = generate_story(image_caption, pil_image, genre.lower(), n_stories)
    return story


if __name__ == "__main__":
    title = "Image to Story"
    article = "Combines the power of [clip prefix captioning](https://github.com/rmokady/CLIP_prefix_caption) with [gpt2 story generator](https://huggingface.co/pranavpsv/genre-story-generator-v2) to create stories of different genres from image"
    description = "Drop an image and generate stories of different genre based on that image"

    interface = gr.Interface(
        main,
        title=title,
        description=description,
        article=article,
        inputs=[
            gr.inputs.Image(type="pil", source="upload", label="Input"),
            gr.inputs.Dropdown(
                type="value",
                label="Story genre",
                choices=[
                    "superhero",
                    "action",
                    "drama",
                    "horror",
                    "thriller",
                    "sci_fi",
                ],
            ),
            gr.inputs.Radio(choices=["coco", "conceptual"], label="Model"),
            gr.inputs.Dropdown(choices=[1, 2, 3], label="No. of stories", type="value"),
        ],
        outputs=gr.outputs.Textbox(label="Generated story"),
        examples=[["car.jpg", "drama", "conceptual"], ["gangster.jpg", "action", "coco"]],
        enable_queue=True,
    )
    interface.launch()
