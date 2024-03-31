from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch import nn
from clip import tokenize


# test Koniq and liveiwt dataset
CLASSNAMES=[
    ['Good photo.', 'Bad photo.'],
]

# test degradation attributes
# CLASSNAMES=[
#     ['Good photo.', 'Bad photo.'],
#     ['Bright photo.', 'Dark photo.'],
#     ['Sharp photo.', 'Blurry photo.'],
#     ['Noisy photo.', 'Clean photo.'],
#     ['Colorful photo.', 'Dull photo.'],
#     ['High contrast photo.', 'Low contrast photo.'],
# ]

# test AVA attributes
# CLASSNAMES=[
#     ['Aesthetic photo.', 'Not aesthetic photo.'],
#     ['Happy photo.', 'Sad photo.'],
#     ['Natural photo.', 'Synthetic photo.'],
#     ['New photo.', 'Old photo.'],
#     ['Scary photo.', 'Peaceful photo.'],
#     ['Complex photo.', 'Simple photo.'],
# ]


"""
1.PE加不加
2.q bench用/100做logits
3.CLIP—IQA是直接用logits_per_image做logits, *2.6593
4.Open CLIP是100* image_features @ class_embeddings做logits
"""

# q bench, clip-iqa
def zero_shot_eval_clipiqa(model, data_loader, device):

    with torch.no_grad():
        model.eval()
        tokenized_prompts = []
        for i in range(len(CLASSNAMES)):
            tokenized_prompts.append(tokenize(CLASSNAMES[i])) # 里面每个都是[class_num, 77]
        pred_score=[]
        target_list=[]
        for image, target in data_loader:
            image = image.to(device)
            # 假设target只有MOS分数
            target = target.to(device)
            logits_list = []
            for i in range(len(CLASSNAMES)):
                # 此处text没有PE
                logits_per_image, logits_per_text = model(image, tokenized_prompts[i].to(device))
                logits = logits_per_image.softmax(dim=-1) # [bs, class_num]
                logits_list.append(logits[:, 0].unsqueeze(1)) # [bs, 1]
            logits_list = torch.cat(logits_list, dim=1).float() #[bs, n], n组prompt
            pred_score.append(logits_list)
            target = target.detach().cpu().tolist()
            for t in target:
                target_list.append(t) #[all]
        pred_list=torch.cat(pred_score, dim=0).detach().cpu().tolist() #[all, n]
    return pred_list, target_list

