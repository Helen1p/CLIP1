from utils.tools import load_config
import torch
from clip.model import build_model
from dataset.my_dataset import transform
from PIL import Image
import os
import clip
import torch.nn.functional as F
import matplotlib.pyplot as plt



config_path=r'/data/pxg1/CLIP1/config/train.yaml'

def paint_smi_matrixs(matrix):
    plt.clf()
    w, h = matrix.shape
    plt.imshow(matrix)
    plt.colorbar()
    plt.savefig(fname="/data/pxg1/CLIP1/graph/matrix.png", dpi=400)
    plt.close()

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    config=load_config(config_path)
    transform_=transform(224)
    image_path=r'/data/pxg1/data/q-instruct-images/spaq_koniq/00003.jpg'
    image=transform_(Image.open(image_path)).to(device).unsqueeze(dim=0)
    text_='good photo'
    ckpt_path = config['test']['ckpt']
    checkpoint=torch.load(ckpt_path, map_location=device)
    model=build_model(checkpoint['state_dict'], mode='test', frozen_layers=False, 
            load_from_clip=False).to(device)
    text=clip.tokenize(text_, 248, truncate=True)[0] # [77], torch.int32
    text=text.to(device).unsqueeze(dim=0)
    _, i_=model.encode_image(image) # [1, 256, 768], torch.float16
    text=model.encode_text(text) # [1, 768], torch.float16
    map = i_ @ text.t() # [bs_image=1, 256, bs_text=1], torch.float16
    map = map.reshape([16, 16]).unsqueeze(dim=0).unsqueeze(dim=0)
    map = F.interpolate(map, size=(224, 224), mode='nearest')
    map = map.squeeze().squeeze()
    map = map.detach().cpu().numpy()
    paint_smi_matrixs(map)


if __name__=='__main__':
    main()