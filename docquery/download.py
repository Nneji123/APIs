from docquery.pipeline import get_pipeline

PIPELINES = {}
CHECKPOINTS = {
    "LayoutLMv1 🦉": "impira/layoutlm-document-qa"
}


def construct_pipeline(model):
    global PIPELINES
    if model in PIPELINES:
        return PIPELINES[model]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ret = get_pipeline(checkpoint=CHECKPOINTS[model], device=device)
    PIPELINES[model] = ret
    return ret

construct_pipeline(model=list(CHECKPOINTS.keys())[0])