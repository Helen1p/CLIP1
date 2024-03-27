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
# 要5个
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


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def build_zero_shot_classifier(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names in batches
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm
        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in templates]
        texts = tokenizer(texts).to(device)
        class_embeddings = model.encode_text(texts, normalize=True)
        class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True) # [class_num, dim]
        class_embeddings = class_embeddings.T
        return class_embeddings # [dim, class_num]

    with torch.no_grad():
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
        # [dim, all_class_num]
    return zeroshot_weights

def build_zero_shot_classifier(
        model,
        tokenizer,
        classnames: Sequence[str],
        device: Union[str, torch.device] = 'cpu',
):
    with torch.no_grad():
        texts=tokenizer(classnames).to(device) # [class_num, 77]
        class_embeddings = model.encode_text(texts, normalize=True) # [class_num, 512]
        # class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True) # [class_num, 512]
        class_embeddings = class_embeddings.T # [512, class_num]
        return class_embeddings


def build_zero_shot_classifier_legacy(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names 1 by 1
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    if use_tqdm:
        import tqdm
        iter_wrap = tqdm.tqdm
    else:
        iter_wrap = iter

    use_format = isinstance(templates[0], str)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in iter_wrap(classnames):
            texts = [template.format(classname) if use_format else template(classname) for template in templates]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

# maybe different from the original CLIP zero shot -> no target exist.
def run(model, classifier, dataloader, batch_size, device):

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=batch_size):
            # images = images.to(device=device, dtype=input_dtype)
            images = images.to(device=device)
            target = target.to(device)

            # predict
            output = model(image=images)
            image_features = output['image_features'] if isinstance(output, dict) else output[0]
            logits = 100. * image_features @ classifier
            # [bs, 512]
            # [dim, class_num]

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data_loader, device):
    # if args.distributed and not args.horovod:
    #     model = model.module

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

    # results = {}
    # top1, top5 = run(model, classifier, data_loader, batch_size, device)
    # results['zeroshot-val-top1'] = top1
    # results['zeroshot-val-top5'] = top5

    # return results
