import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
# from pytorch_lightning.logging import TestTubeLogger

# dataset.py
from hw4_dataset import KlevrDataset
import wandb

"""
Citation:
@misc{queianchen_nerf,
  author={Quei-An, Chen},
  title={Nerf_pl: a pytorch-lightning implementation of NeRF},
  url={https://github.com/kwea123/nerf_pl/},
  year={2020},
}
"""

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)
        # self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number    
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]


    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        self.train_dataset = KlevrDataset(root_dir=self.hparams.root_dir, split='train')
        self.val_dataset = KlevrDataset(root_dir=self.hparams.root_dir, split='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        """return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }"""
    
        self.log('train_loss', log['train/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        self.log('train_psnr', log['train/psnr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        loss = self.loss(results, rgbs)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)

            # Log images to wandb
            wandb.log({"val/GT_pred_depth": [wandb.Image(i) for i in stack]}, step=self.global_step)

        psnr_value = psnr(results[f'rgb_{typ}'], rgbs)
        self.log('val/psnr', psnr_value, on_step=False, on_epoch=True, prog_bar=True)


    # def validation_epoch_end(self, outputs):
    #     mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

    #     return {'progress_bar': {'val_loss': mean_loss,
    #                              'val_psnr': mean_psnr},
    #             'log': {'val/loss': mean_loss,
    #                     'val/psnr': mean_psnr}
    #            }
    def on_validation_epoch_end(self):
        # outputs = self.trainer.callback_metrics
        # mean_loss = outputs.get('val_loss')
        # mean_psnr = outputs.get('val_psnr')
        
        # if mean_loss is not None:
        #     self.log('val/loss', mean_loss, on_epoch=True, prog_bar=True)
        # if mean_psnr is not None:
        #     self.log('val/psnr', mean_psnr, on_epoch=True, prog_bar=True)
        pass

    def configure_optimizers(self):
        # # optimizer
        # if self.hparams.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # elif self.hparams.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        
        # # scheduler
        # if self.hparams.lr_scheduler == 'steplr':
        #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.decay_step[0], gamma=self.hparams.decay_gamma)
        # elif self.hparams.lr_scheduler == 'cosine':
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.num_epochs)
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)

        return [self.optimizer], [scheduler]
    
if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    # init wandb
    wandb.init(project='nerf', name=hparams.exp_name, config=hparams)


    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('ckpts',hparams.exp_name),
                                          filename='{epoch:d}',
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5,)

    logger = pl.loggers.WandbLogger(name=hparams.exp_name, project='nerf')
    

    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,  
        enable_model_summary=True,
        devices=hparams.num_gpus if torch.cuda.is_available() else None,  
        accelerator='gpu' if hparams.num_gpus > 0 else None, 
        num_sanity_val_steps=1,
        benchmark=True,
        # enable_checkpointing=True,
        profiler = 'simple'if hparams.num_gpus > 0 else None,
    )

    trainer.fit(system)