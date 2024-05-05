from enum import Enum
from typing import Union

import torch

# from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name

import sys
sys.path.append(r'/data/pxg1/dinov2-main/dinov2/models')

class Weights(Enum):
    LVD142M = "LVD142M"


def make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    # from ..models import vision_transformer as vits
    import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    # model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)
    # model = vit_large(**vit_kwargs)

    if pretrained:
        # model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        # url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        # state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        state_dict = torch.load(r'/data/pxg1/data/dinov2_vitl14_pretrain.pth')
        model.load_state_dict(state_dict, strict=True)

    return model

if __name__=='__main__':
    dinov2_vitl14=make_dinov2_model().cuda().eval()
    for param in dinov2_vitl14.parameters():
        param.requires_grad = False

    from PIL import Image
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    import json
    import os
    import numpy as np
    from tqdm import tqdm

    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    def convert_image_to_rgb(image):
        return image.convert("RGB")

    def transform(n_px=224):
        return Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    t=transform(224)
    
    with open(r'/data/pxg1/CLIP1/json_data/new.json', 'r') as f:
        file=json.load(f)
    for i in tqdm(file):
        path=os.path.join(r'/data/pxg1/data/q-instruct-images/',i['image'])
        img=t(Image.open(path))
        img=img.unsqueeze(dim=0).cuda()
        # print(img.shape) # [1, 3, 224, 224]
        with torch.no_grad():
            img_dict=dinov2_vitl14.forward_features(img)
            image_embeddings = img_dict['x_norm_patchtokens']
        # print(image_embeddings.shape) # [1, 256, 1024] (224/14)^2=256
        image_embeddings= image_embeddings.detach().cpu().numpy()
        np.save(r'/data/pxg1/data/q-instruct-images-dinov2/'+i['image'].split('.')[0]+'.npy',image_embeddings)
        
