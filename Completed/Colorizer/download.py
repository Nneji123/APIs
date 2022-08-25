import gdown


def url_to_id(url):
    x = url.split("/")
    return x[5]


id = url_to_id(
    "https://drive.google.com/file/d/1EshSEknFNC0eknpyLk39N1x-PKRi4Vpv/view?usp=sharing"
)


def main():

    url = "https://drive.google.com/uc?id=" + id
    output = "models/colorization_release_v2.caffemodel"
    gdown.download(url, output, quiet=False)


if __name__ == "__main__":
    main()
