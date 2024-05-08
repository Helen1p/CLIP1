import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from dataset.my_dataset import train_set, test_set
import clip
from clip.model import build_model
from trainer import Trainer
import yaml
import sys
from testing.score_clipiqa import zero_shot_eval
from utils.metrics import srocc, plcc
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def load_config(path):
    path=str(path)
    with open(path, 'r', encoding='utf-8') as f:
        config=yaml.safe_load(f)
    return config

def main(args):
    config=load_config(args.config_path)
    frozen_layers=False
    if args.mode=='fine_tune' and args.frozen_layers:
        frozen_layers=True

    if args.mode == 'train' or args.mode == 'fine_tune':
        # fine tune / train from scratch

        local_rank = int(os.environ["LOCAL_RANK"])
        # local_rank 用os或者args.local_rank传（但torchrun已经把args.local_rank淘汰了），反正是不能dist.get_rank()
        torch.distributed.init_process_group("nccl", world_size=args.n_gpus, rank=local_rank)
        torch.cuda.set_device(local_rank) # CUDA_VISIBLE_DEVICES是哪几张可以看见，这个是在可见的基础上用n_gpus

        if args.load_from_clip==False:
            context_length=248
        else:
            context_length=77
        train_dataset=train_set(config['dataset']['train_json_path'], config['dataset']['train_image_path'], 
        config['dataset']['n_px'], context_length, config['dataset']['train_prior_path'])
        train_sampler=DistributedSampler(train_dataset)

        # valid_dataset=image_title_dataset(config['dataset']['valid_input_filename'], config['dataset']['n_px'])
        train_loader=DataLoader(train_dataset, config['train']['batch_size'], shuffle=(train_sampler is None), num_workers=16, sampler=train_sampler)
        # valid_loader=DataLoader(valid_dataset, config['train']['batch_size'])

        # TODO: change to DDP
        # device = "cuda:0" if torch.cuda.is_available() else "cpu" 
        device = torch.device("cuda", local_rank)

        
        # 1.original CLIP
        # must set jit=False for training
        model, _ = clip.load("ViT-L/14", device=device, jit=False, mode=args.mode, frozen_layers=frozen_layers, 
        load_from_clip=args.load_from_clip)

        # 2.variant CLIP
        # ckpt_path = config['train']['variant_clip']
        # checkpoint=torch.load(ckpt_path, map_location=device)
        # model=build_model(checkpoint, mode=args.mode, frozen_layers=frozen_layers, load_from_clip=True, name='ViT-B/32').to(device)

        # initialize optimizer
        optimizer=getattr(sys.modules['torch.optim'], config['optimizer']['name'])
        optimizer=optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                lr=eval(config['optimizer']['lr']),
                betas=eval(config['optimizer']['betas']),
                eps=eval(config['optimizer']['eps']),
                weight_decay=config['optimizer']['weight_decay']
                )
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, 
        # betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)

        # initialize lr_scheduler
        sc=config.get('lr_scheduler', None)
        if  sc is None:
            lr_scheduler=None
        else:
            # raise error: e.g. milestones=[30,80], because in sc it is '[30,80]'! str!!!
            scheduler=getattr(sys.modules['torch.optim.lr_scheduler'], sc.pop('name'))
            lr_scheduler=scheduler(optimizer, **sc)
        if config['train']['ckpt']=='None':
            ckpt=None
        train_trainer=Trainer.trainer(local_rank=local_rank,
                            model=model, 
                            epoch=config['train']['epoch'], 
                            optimizer=optimizer, 
                            device=device, 
                            ckpt_save_path=config['train']['ckpt_save_path'], 
                            log_dir=config['train']['log_dir'], 
                            train_loader=train_loader, 
                            valid_loader=None, 
                            ckpt=ckpt, 
                            lr_scheduler=lr_scheduler,
                            sampler=train_sampler)
        train_trainer.train()
    
    elif args.mode == 'test':
        # zero shot MOS / zero shot attributes classifer
        test_dataset=test_set(config['dataset']['test_json_path'], config['dataset']['test_image_path'], config['dataset']['n_px'])
        # len(test_dataset)=2985, len(test_loader)=12
        test_loader=DataLoader(test_dataset, config['test']['batch_size'], num_workers=16)
        device = "cuda:0" if torch.cuda.is_available() else "cpu" 

        # 1.test on OG clip
        # model, _ = clip.load("ViT-B/32", device=device, jit=False, mode=args.mode, frozen_layers=frozen_layers)

        # 2.test on our checkpoint
        ckpt_path = config['test']['ckpt']
        checkpoint=torch.load(ckpt_path, map_location=device)
        # model=build_model(checkpoint, mode=args.mode, frozen_layers=frozen_layers, load_from_clip=True).to(device)
        model=build_model(checkpoint['state_dict'], mode=args.mode, frozen_layers=frozen_layers, 
        load_from_clip=args.load_from_clip).to(device)
        # q bench/clip iqa, q align
        pred_list, target_list = zero_shot_eval(model=model, data_loader=test_loader, device=device)
        pred_list1 = pred_list[0]
        # #srcc for feature npz generation
        srocc_, plcc_ = srocc(pred_list1, target_list), plcc(pred_list1, target_list)
        print('srocc: ', srocc_, ' plcc: ', plcc_)

        # save visual encoder features of different layers
        # pred_list, target_list, feature = zero_shot_eval(model=model, data_loader=test_loader, device=device)
        # np.savez('/data/pxg1/CLIP1/npz_data/arrays_ai.npz', l23=feature[23], l22=feature[22], 
        #         l21=feature[21], l20=feature[20], l19=feature[19])
        # np.savez('/data/pxg1/CLIP1/npz_data/arrays_wild.npz', l23=feature[23], l22=feature[22], 
        #         l21=feature[21], l20=feature[20], l19=feature[19])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/data/pxg1/CLIP1/config/train.yaml',
                        help='Config path of models')
    parser.add_argument('--mode', type=str, default='fine_tune',
                        help='train or test or fine_tune')
    parser.add_argument('--frozen_layers', type=bool, default=False,
                        help='frozen_layers')
    parser.add_argument('--load_from_clip', type=bool, default=False,
                        help='context_length = 77 or 248')
    # parser.add_argument("--local_rank", help="local device id on current node",
    #                     type=int)
    parser.add_argument("--n_gpus", type=int, default=2, 
                        help="GPU number")
    args=parser.parse_args()

    main(args)


    # train/fine_tune: torchrun --nnodes 1 --nproc_per_node 2 train.py



    # test: python train.py


    # local rank, clip forward, label在哪个device上