from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch import nn
from clip import tokenize


# q align
# 要5个
CLASSNAMES=[
    ['Good photo.', 'Bad photo.'],
]

"""
weights: 
ours: 1,2,3,4,5
q align: [1,0.75,0.5,0.25,0.]->[5, 3.75, 2.5, 1.25, 0.0]
"""

def zero_shot_eval_qalign(model, data_loader, device):

    with torch.no_grad():
        model.eval()
        texts=tokenize(CLASSNAMES[0]).to(device) # [class_num, 77]
        class_embeddings = model.encode_text(texts) # [class_num, 512]
        # class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True) # [class_num, 512]
        class_embeddings = class_embeddings.T # [512, class_num]
        class_num=class_embeddings.shape[-1]
        pred_list=[]
        target_list=[]
        # pbar=tqdm(data_loader, len(data_loader))
        # for image, target in pbar:
        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)
            # target = target.to(device)
            image_features=model.encode_image(image) # [bs, 512]
            image_features = image_features / image_features.norm(dim=1, keepdim=True) # [bs, 512]
            logits = 100. * image_features @ class_embeddings # [bs, class_num]
            logits = F.softmax(logits, dim=1)
            # 6
            score_rank=torch.arange(1, 3).unsqueeze(dim=0).T.to(torch.float16).to(device) # [class_num, 1]
            pred = logits @ score_rank # [bs, 1]
            pred = pred.squeeze().detach().cpu().tolist()
            target = target.detach().cpu().tolist()
            for p in pred:
                pred_list.append(p)
            for t in target:
                target_list.append(t)

    return pred_list, target_list

