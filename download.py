import gdown

def url_to_id(url):
    x = url.split("/")
    return x[5]

id = url_to_id('https://drive.google.com/file/d/1cxUhGB2olQasvBekk-TAhgKzLM6wkOiq/view?usp=sharing')

url = 'https://drive.google.com/uc?id='+id
output = 'Course-Recommender-Sytem__API.zip'
gdown.download(url, output, quiet=False)