import gdown
import os

def url_to_id(url):
    x = url.split("/")
    return x[5]

id = url_to_id('https://drive.google.com/file/d/13oJ_9jeylTmW7ivmuNmadwraWceHoQbK/view?usp=sharing')


def main():

    url = 'https://drive.google.com/uc?id='+id
    output = 'file.zip'
    gdown.download(url, output, quiet=False)
    os.system('unzip file.zip')
    print('unzipped files successfully!')

if __name__=="__main__":
    main()