from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch import nn
from clip import tokenize


# CLASSNAMES=[
#     ['Excellent photo.', 'Good photo.', 'Fair photo.', 'Poor photo.', 'Bad photo.'],
# ]

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
logits: 
1.q bench用/100做logits
2.CLIP-IQA是直接用logits_per_image做logits, *100; 
Open CLIP是100* image_features @ class_embeddings做logits, 发现和CLIP-IQA一样的
"""

"""
weights: 
clip iqa: [1, 0]
ours: [5, 4, 3, 2, 1]->[1, 0.8, 0.6, 0.4, 0.1]
q align: [1, 0.75, 0.5, 0.25, 0.]->[5, 3.75, 2.5, 1.25, 0.0]
"""


def zero_shot_eval(model, data_loader, device):

    with torch.no_grad():
        model.eval()
        tokenized_prompts = []
        for i in range(len(CLASSNAMES)):
            tokenized_prompts.append(tokenize(CLASSNAMES[i], context_length=248)) # 里面每个都是[class_num, 77]
        pred_score=[]
        target_list=[]
        for idx, (image, target) in enumerate(data_loader):
            image = image.to(device)
            # 假设target只有MOS分数
            # target = target.to(device)
            logits_list = []
            for i in range(len(CLASSNAMES)):
                # 1.clip-iqa
                logits_per_image, logits_per_text = model(image, tokenized_prompts[i].to(device))

                # logits_per_image, logits_per_text, feature = model(image, tokenized_prompts[i].to(device))
                # fe=[i.detach().cpu().numpy() for i in feature]
                
                # 2.open-clip
                # image_features=model.encode_image(image) # [bs, 512]
                # image_features = image_features / image_features.norm(dim=1, keepdim=True)
                # class_embeddings=model.encode_text(tokenized_prompts[i].to(device))
                # class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True) # [class_num, 512]
                # class_embeddings = class_embeddings.T # [512, class_num]
                # logits_per_image = 100. * image_features @ class_embeddings # [bs, class_num]

                logits = logits_per_image.softmax(dim=-1) # [bs, class_num]
                score_rank=torch.Tensor([1, 0]).unsqueeze(dim=0).T.to(torch.float16).to(device) # [class_num, 1]
                # score_rank=torch.Tensor([1, 0.75, 0.5, 0.25, 0.]).unsqueeze(dim=0).T.to(torch.float16).to(device) # [class_num, 1]
                # score_rank=torch.Tensor([1, 0.8, 0.6, 0.4, 0.1]).unsqueeze(dim=0).T.to(torch.float16).to(device) # [class_num, 1]
                pred = logits @ score_rank # [bs, 1]
                # logits_list.append(logits[:, 0].unsqueeze(1)) # [bs, 1]
                logits_list.append(pred)
            logits_list = torch.cat(logits_list, dim=1).float() #[bs, n], n组prompt
            pred_score.append(logits_list)
            for t in target:
                if type(t)==str:
                    target_list.append(eval(t)) #[all]
                else:  # float
                    target_list.append(t) #[all]
            print('******************',idx,' end******************')
        pred_list=torch.cat(pred_score, dim=0).T.detach().cpu().tolist() #[all, n]->[n, all]->[[all个],..n个..,[]]
    return pred_list, target_list
    # return pred_list, target_list, fe