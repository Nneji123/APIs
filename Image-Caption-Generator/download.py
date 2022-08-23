import os
if os.path.exists('models/model.h5'):
    pass
else:
    os.system("wget -O models/model.h5 https://github.com/ABX9801/Image-Caption-Generator/raw/main/model/model.h5")
    print('file downloaded...')