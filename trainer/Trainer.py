import torch
from tqdm import tqdm
import os
from torch import nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import AverageMeter
import torch.distributed as dist


class trainer():
    def __init__(self, local_rank, model, epoch, optimizer, device, ckpt_save_path, log_dir, train_loader, valid_loader=None, ckpt=None, lr_scheduler=None, sampler=None):
        super().__init__()
        self.local_rank=local_rank
        self.model=model
        self.optimizer=optimizer
        self.epoch=epoch
        self.start_epoch = 0
        self.device=device
        self.ckpt_save_path=ckpt_save_path
        self.train_loader=train_loader
        self.valid_loader=valid_loader
        self.lr_scheduler=lr_scheduler
        self.ckpt=ckpt
        self.loss_image=nn.CrossEntropyLoss()
        self.loss_text=nn.CrossEntropyLoss()
        if self.ckpt is not None:
            self.load_ckpt(self.ckpt)
        # ？dir写错了
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_loss = AverageMeter()
        self.val_metric1 = AverageMeter()
        self.sampler=sampler
        self.local_loss=False
        
    # 为了继续训练
    def load_ckpt(self, path):
        path=str(path)
        checkpoint=torch.load(path)
        self.start_epoch=checkpoint['epoch']+1
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])


    def save_ckpt(self, epoch, save_best=False):
        """
        save_best: if True, only best be saved as 'model_best.pth'
        """
        arch=type(self.model).__name__
        state={
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if not os.path.exists(self.ckpt_save_path):
            os.mkdir(self.ckpt_save_path)
        filename= str(self.ckpt_save_path)+'checkpoint_epoch{}.pth'.format(epoch)

        torch.save(state, filename)
        if save_best:
            best_path=str(self.ckpt_save_path)+'model_best.pth'
            torch.save(state, best_path) 
            

    def valid_epoch(self, epoch):
        """
        return: a log that contains information about validation
        """
        self.val_metric1.reset()
        self.model.eval()
        # metrics reset

        with torch.no_grad():
            pbar = tqdm(self.valid_loader, total=len(self.valid_loader), mininterval=10)
            for batch_idx,(image, text) in enumerate(pbar):
                n=image.shape[0]
                image, text=image.to(self.device), text.to(self.device)
                logits_per_image, logits_per_text=self.model(image,text)
                groundtruth=torch.arange(len(image),dtype=torch.long, device=self.device)
                total_loss=(self.loss_image(logits_per_image, groundtruth)+self.loss_text(logits_per_text, groundtruth))/2
                # 这个要不要.item()和n？
                self.val_metric1.update(total_loss.item(), n)

                pbar.set_postfix({'Epoch': epoch,
                                    'loss': self.val_metric1.avg,})
            self.writer.add_scalar('val_metric1', self.val_metric1.avg, epoch)
        
        return


    def train(self):
        scaler = GradScaler()
        gather_with_grad=False

        self.model = nn.parallel.DistributedDataParallel(self.model.cuda(self.local_rank), device_ids=[self.local_rank])

        for epoch in range(self.start_epoch,self.epoch):
            # result=self.train_epoch(epoch, scaler)
            self.train_loss.reset()
            self.model.train()
            self.sampler.set_epoch(epoch)

            
            pbar = tqdm(self.train_loader, total=len(self.train_loader), mininterval=10)
            # for batch_idx, (image, text, prior) in enumerate(pbar):
            for batch_idx, (image, text) in enumerate(pbar):
                # model的todevice放到train.py里面
                # image, text, prior = image.to(self.device), text.to(self.device), prior.to(self.device)
                image, text, prior = image.cuda(self.local_rank), text.cuda(self.local_rank), prior.cuda(self.local_rank)
                image, text = image.cuda(self.local_rank), text.cuda(self.local_rank)
                # image, text = image.to(self.device), text.to(self.device)
                # n=image.shape[0]
                # groundtruth=torch.arange(len(image),dtype=torch.long, device=self.device)
                self.optimizer.zero_grad()
                # with autocast():
                #     logits_per_image, logits_per_text=self.model(image, text, prior)
                    # logits_per_image, logits_per_text=self.model(image, text)
                
                image_features=self.model.module.encode_image(image) # [bs, 512]
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = self.model.module.encode_text(text)
                text_features = text_features / text_features.norm(dim=1, keepdim=True) # [class_num, 512]
                logit_scale = self.model.module.logit_scale.exp() 

                # open_clip: src/open_clip/loss.py
                # gather with grad retained
                if gather_with_grad:
                    all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
                    all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
                else:
                    gathered_image_features = [torch.zeros_like(image_features) for _ in range(dist.get_world_size())]
                    gathered_text_features = [torch.zeros_like(text_features) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_image_features, image_features)
                    dist.all_gather(gathered_text_features, text_features)
                    # print(gathered_image_features)
                    if not self.local_loss:
                        # ensure grads for local rank when all_* features don't have a gradient
                        gathered_image_features[dist.get_rank()] = image_features
                        gathered_text_features[dist.get_rank()] = text_features
                    # print(gathered_image_features)
                    all_image_features = torch.cat(gathered_image_features, dim=0)
                    all_text_features = torch.cat(gathered_text_features, dim=0)

                if self.local_loss:
                    logits_per_image = logit_scale * image_features @ all_text_features.T
                    logits_per_text = logit_scale * text_features @ all_image_features.T
                else:
                    logits_per_image = logit_scale * all_image_features @ all_text_features.t() # [bs, class_num]
                    logits_per_text = logits_per_image.t()

                device=image_features.device
                labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
                if self.local_loss:
                    labels = labels + logits_per_image.shape[0] * self.local_rank
                n=logits_per_image.shape[0]


                total_loss=(self.loss_image(logits_per_image, labels)+self.loss_text(logits_per_text, labels))/2
                # scaler.scale(total_loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()
                total_loss.backward()
                self.optimizer.step()

                if self.valid_loader is not None:
                    self.valid_epoch(epoch)
                
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    
                self.train_loss.update(total_loss.item(), n)
                pbar.set_postfix({'Epoch': epoch,
                                'loss': self.train_loss.avg})
            self.writer.add_scalar('loss', self.train_loss.avg, epoch)
            # if epoch+1 %10==0:
            if self.local_rank==0:
                self.save_ckpt(epoch, save_best=False)
            
        return
    