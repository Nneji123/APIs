from skimage import io
from pyxelate import Pyx, Pal


def pixel(image, downsample, palette, depth, upscale):
    image = io.imread(image)
    # new image will be 1/14th of the original in size
    downsample_by = int(downsample)
    palette = int(palette)  # find 7 colors
    # 1) Instantiate Pyx transformer
    pyx = Pyx(factor=downsample_by, palette=palette,
              depth=int(depth), upscale=int(upscale))
    # 2) fit an image, allow Pyxelate to learn the color palette
    pyx.fit(image)
    # 3) transform image to pixel art using the learned color palette
    new_image = pyx.transform(image)
    # save new image with 'skimage.io.imsave()'
    io.imsave("output.jpg", new_image)
    return "output.jpg"
