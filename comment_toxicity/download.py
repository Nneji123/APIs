import os
if os.path.exists('comment_tokenizer.pkl'):
    print('comment_tokenizer.pkl already exists')
else:
    os.system("wget https://huggingface.co/spaces/mahidher/comment_toxicity/resolve/main/comment_toxicity_model.h5")
    os.system("wget https://huggingface.co/spaces/mahidher/comment_toxicity/resolve/main/comment_tokenizer.pkl")
    print('comment_tokenizer.pkl downloaded')