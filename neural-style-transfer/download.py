import os

# Style Transfer Model
import paddlehub as hub

os.system("hub install stylepro_artistic==1.0.1")
stylepro_artistic = hub.Module(name="stylepro_artistic")
print("Downloaded files")
