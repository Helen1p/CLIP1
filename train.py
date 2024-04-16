import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from dataset.my_dataset import image_title_dataset, test_MOS_dataset
import clip
from clip.model import build_model
from trainer import Trainer
import yaml
import sys
from testing.score_clipiqa import zero_shot_eval
from utils.metrics import srocc, plcc

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
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
        train_dataset=image_title_dataset(config['dataset']['train_json_path'], config['dataset']['train_image_path'], config['dataset']['n_px'])
        # train_dataset=image_title_dataset('/root/CLIP/data.csv', 224)

        # valid_dataset=image_title_dataset(config['dataset']['valid_input_filename'], config['dataset']['n_px'])
        train_loader=DataLoader(train_dataset, config['train']['batch_size'], shuffle=True, num_workers=8)
        # valid_loader=DataLoader(valid_dataset, config['train']['batch_size'])

        # TODO: change to DDP
        device = "cuda:0" if torch.cuda.is_available() else "cpu" 
        
        # must set jit=False for training
        model, _ = clip.load("ViT-L/14", device=device, jit=False, mode=args.mode, frozen_layers=frozen_layers)
        # initialize optimizer
        optimizer=getattr(sys.modules['torch.optim'],config['optimizer']['name'])
        optimizer=optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                lr=eval(config['optimizer']['lr']),
                betas=eval(config['optimizer']['betas']),
                eps=eval(config['optimizer']['eps']),
                weight_decay=config['optimizer']['weight_decay']
                )
        # initialize lr_scheduler
        sc=config.get('lr_scheduler', None)
        if  sc is None:
            lr_scheduler=None
        else:
            # raise error: e.g. milestones=[30,80], because in sc it is '[30,80]'! str!!!
            scheduler=getattr(sys.modules['torch.optim.lr_scheduler'], config['lr_scheduler']['name'])
            lr_scheduler=scheduler(optimizer, **sc)
        if config['train']['ckpt']=='None':
            ckpt=None
        train_trainer=Trainer.trainer(model=model, 
                            epoch=config['train']['epoch'], 
                            optimizer=optimizer, 
                            device=device, 
                            ckpt_save_path=config['train']['ckpt_save_path'], 
                            log_dir=config['train']['log_dir'], 
                            train_loader=train_loader, 
                            valid_loader=None, 
                            ckpt=ckpt, 
                            lr_scheduler=lr_scheduler)
        train_trainer.train()
    
    if args.mode == 'test':
        # zero shot MOS / zero shot attributes classifer
        test_dataset=test_MOS_dataset(config['dataset']['test_json_path'], config['dataset']['test_image_path'], config['dataset']['n_px'])
        # len(test_dataset)=2985, len(test_loader)=12
        test_loader=DataLoader(test_dataset, config['test']['batch_size'], num_workers=1)
        device = "cuda:0" if torch.cuda.is_available() else "cpu" 

        # 1.test on OG clip
        # model, _ = clip.load("ViT-B/32", device=device, jit=False, mode=args.mode, frozen_layers=frozen_layers)

        # 2.test on our checkpoint
        ckpt_path = config['test']['ckpt']
        checkpoint=torch.load(ckpt_path, map_location=device)
        model=build_model(checkpoint['state_dict'], mode=args.mode, frozen_layers=frozen_layers, PE=True).to(device)
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
    parser.add_argument('--mode', type=str, default='test',
                        help='train or test or fine_tune')
    parser.add_argument('--frozen_layers', type=bool, default=True,
                        help='frozen_layers')
    args=parser.parse_args()

    main(args)