from torch.utils.data import Dataset
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import pandas as pd

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
# train
class image_title_dataset(Dataset):
    def __init__(self, input_filename, n_px):
        df = pd.read_csv(input_filename)
        self.list_image = df["image"].tolist()
        self.list_caption = df["caption"].tolist()
        self.preprocess=transform(n_px)
    
    def __len__(self):
        return len(self.list_caption)
    
    def __getitem__(self,idx):
        images=self.preprocess(Image.open(self.list_image[idx]))
        texts=clip.tokenize(self.list_caption[idx])[0] # [1, 77] -> [77]
        return images, texts

# test: image + target(MOS)
class test_MOS_dataset(Dataset):
    def __init__(self, input_filename, n_px):
        df = pd.read_csv(input_filename)
        self.list_image = df["image"].tolist()
        self.list_MOS = df["MOS"].tolist()
        self.preprocess=transform(n_px)
    
    def __len__(self):
        return len(self.list_MOS)
    
    def __getitem__(self,idx):
        images=self.preprocess(Image.open(self.list_image[idx]))
        MOSs=self.list_MOS[idx] 
        return images, MOSs
    

# another test: image + low-level attribute groundtruth

# for webdata: .tar

if __name__=='__main__':
    df=pd.read_csv('/root/CLIP1/data_test.csv')
    a=df["MOS"].tolist()
    # 就是int啊
    print(type(a[0]))