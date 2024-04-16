from torch.utils.data import Dataset
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import pandas as pd
import json
import os


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


class image_title_dataset(Dataset):
    def __init__(self, train_json_path, train_image_path, n_px):
        with open(train_json_path, 'r') as f:
            self.data=json.load(f)
        self.list_image = [x['image'] for x in self.data]
        # 3个
        self.list_caption_local = [x['local'] for x in self.data]
        # 1个 / 3个
        self.list_caption_global = [x['global'] for x in self.data]
        self.preprocess=transform(n_px)
        self.root_path=train_image_path
    
    def __len__(self):
        return len(self.list_image)
    
    def __getitem__(self,idx):
        # caption太长了，truncate = Ture
        images=self.preprocess(Image.open(os.path.join(self.root_path, self.list_image[idx])))
        texts=clip.tokenize(self.list_caption_local[idx][0], truncate=True)[0] # [1, 77] -> [77]
        return images, texts
    

class test_MOS_dataset(Dataset):
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
    df=pd.read_csv('/root/CLIP1/data_test.csv')
    a=df["MOS"].tolist()
    # 就是int啊
    print(type(a[0]))