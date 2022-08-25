import os
if not os.path.isfile('weights.pt'):
    weights_url = 'https://archive.org/download/anpr_weights/weights.pt'
    os.system(f'wget {weights_url}')
    print("Downloaded files!")

if not os.path.isdir('examples'):
    examples_url = 'https://archive.org/download/anpr_examples_202208/examples.tar.gz'
    os.system(f'wget {examples_url}')
    os.system('tar -xvf examples.tar.gz')
    os.system('rm -rf examples.tar.gz')
    print("Downloaded files!")