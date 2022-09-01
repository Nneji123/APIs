import os

os.system('cd fairseq;'
          'pip install --use-feature=2020-resolver ./; cd ..')
os.system('ls -l')

if os.path.exists("caption_large_best_clean.pt") or os.path.exists("checkpoints/caption.pt"):
    print("Model Exists!")
else:
    os.system('wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/caption_large_best_clean.pt; '
          'mkdir -p checkpoints; mv caption_large_best_clean.pt checkpoints/caption.pt')
    os.system("rm -rf caption_large_best_clean.pt")
    print("Model Downloaded")