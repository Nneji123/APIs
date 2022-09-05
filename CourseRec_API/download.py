import gdown

def url_to_id(url: str) -> str:
    x = url.split("/")
    return x[5]

def main():
    id = url_to_id('https://drive.google.com/file/d/1L_oKp8HMP6hglImJGqVBN_myTg0HD_MV/view?usp=sharing')

    url = 'https://drive.google.com/uc?id='+id
    output = 'similarity.pkl'
    gdown.download(url, output, quiet=False)
    
if __name__ == "__main__":
    main()