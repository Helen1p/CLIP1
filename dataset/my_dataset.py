import torch
from torch.utils.data import Dataset
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import pandas as pd
import json
import os
import numpy as np
import cv2


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# from clip/clip.py
def convert_image_to_rgb(image):
    return image.convert("RGB")

def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class train_set(Dataset):
    def __init__(self, train_json_path, train_image_path, n_px, context_length, train_prior_path=None):
        with open(train_json_path, 'r') as f:
            self.data=json.load(f)
        self.list_image = [x['image'] for x in self.data]
        # 3个
        self.list_caption_local = [x['local'] for x in self.data]
        # 1个 / 3个
        self.list_caption_global = [x['global'] for x in self.data]
        self.preprocess=transform(n_px)
        self.root_path=train_image_path
        self.train_prior_path=train_prior_path
        self.context_length=context_length
        self.n_px=n_px
    
    def __len__(self):
        return len(self.list_image)
    
    def process_prior_sam(self, prior):
        # new1.json
        # q-instruct-images-embed
        prior=cv2.resize(prior, [self.n_px, self.n_px])
        prior=torch.Tensor(prior)
        return prior
    
    def process_prior_dinov2(self, prior):
        # new.json
        # q-instruct-images-dinov2
        prior=torch.Tensor(prior)
        # dinov2 抽特征的时候搞忘降维了。。。
        prior=prior.squeeze(dim=0)
        return prior
    
    def __getitem__(self,idx):
        # caption太长了，truncate = Ture
        images=self.preprocess(Image.open(os.path.join(self.root_path, self.list_image[idx])))
        # sam/dinov2 prior
        # FileNotFoundError: sam
        # prior_image=self.list_image[idx].split('.')[0]+'.npy'
        # prior=np.load(os.path.join(self.train_prior_path, prior_image), allow_pickle=True)
        # prior=self.process_prior_dinov2(prior)

        # esemble
        # if len(self.list_caption_local[idx][0].split('. '))>1:         
        #     prior=self.list_caption_local[idx][0].split('. ')[0]
        #     text='. '.join(self.list_caption_local[idx][0].split('. ')[1:])
        # else:
        #     prior=self.list_caption_local[idx][0].split(', ')[0]
        #     text=', '.join(self.list_caption_local[idx][0].split(', ')[1:])
        # prior=clip.tokenize(self.list_caption_global[idx][0], context_length=self.context_length, truncate=True)[0] # [1, 77] -> [77]
        texts=clip.tokenize(self.list_caption_local[idx][0], context_length=self.context_length, truncate=True)[0] # [1, 77] -> [77]
        # return images, texts, prior
        return images, texts
    

class test_set(Dataset):
    def __init__(self, test_json_path, test_image_path, n_px):
        with open(test_json_path, 'r') as f:
            self.data=json.load(f)
        self.list_image = [x['img_path'] for x in self.data]
        self.list_MOS = [x['gt_score'] for x in self.data]
        self.preprocess=transform(n_px)
        self.test_image_path=test_image_path
    
    def __len__(self):
        return len(self.list_image)
    
    def __getitem__(self,idx):
        images=self.preprocess(Image.open(os.path.join(self.test_image_path, self.list_image[idx])))
        MOS=self.list_MOS[idx]
        return images, MOS


# another test: image + low-level attribute groundtruth

# for webdata: .tar

if __name__=='__main__':
    with open('/data/pxg1/CLIP1/json_data/new_1.json', 'r') as f:
            data=json.load(f)
    list_image = [x['image'] for x in data]
    for idx in range(len(list_image)):
        prior_image=list_image[idx].split('.')[0]+'.npy'
        prior=np.load(os.path.join('/data/pxg1/data/q-instruct-images-embed', prior_image), allow_pickle=True)
        print(list_image[idx], prior.shape, type(prior))
        prior=cv2.resize(prior, [224, 224])
# AVA__427243.jpg