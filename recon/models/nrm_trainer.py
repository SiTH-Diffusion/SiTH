"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import os
import logging as log
import torch
import torch.nn as nn
import numpy as np  
import wandb

from .networks.normal_predictor import define_G
from diffusers.optimization import get_scheduler


class NrmTrainer(nn.Module):

    def __init__(self, config, log_dir):
        # Set device to use
        super().__init__()

        self.device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.cfg = config

        self.log_dir = log_dir
        self.log_dict = {}

        self.epoch = 0
        self.global_step = 0
        #self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)

        self._init_model()
        self._init_optimizer()
        self._init_log_dict()


    def _init_model(self):
        """Initialize model.
        """
        self.nrm_predictor = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.nrm_predictor.to(self.device)

    def _init_optimizer(self):
        """Initialize optimizer.
        """

        self.nrm_params = list(self.nrm_predictor.parameters())
        params = []
        params.append({'params': self.nrm_params,
                            'lr': self.cfg.lr_encoder})
        
        self.optimizer = torch.optim.AdamW(params,
                                    betas=(self.cfg.beta1, self.cfg.beta2),
                                    weight_decay=self.cfg.weight_decay)


        self.lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps,
            num_training_steps=self.cfg.epochs * 100,
            num_cycles=self.cfg.lr_num_cycles,
            power=self.cfg.lr_power,
        )


    def _init_log_dict(self):
        """Custom logging dictionary.
        """
        self.log_dict['total_iter_count'] = 0
        self.log_dict['Loss_2D/total_loss'] = 0


    def log(self, step, epoch):
        """Log the training information.
        """
        log_text = 'STEP {} - EPOCH {}/{}'.format(step, epoch, self.cfg.epochs)

        self.log_dict['Loss_2D/total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['Loss_2D/total_loss'])
        log_text += ' | lr: {:>.3E}'.format(self.lr_scheduler.get_last_lr()[0])
        log.info(log_text)

        for key, value in self.log_dict.items():
            wandb.log({key: value}, step=step)

        self._init_log_dict()

        front_img = self.front_rgb_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy()
        wandb.log({"Input Front Images": [wandb.Image(front_img[i]) for i in range(self.b_size)]}, step=step)
        back_img = self.back_rgb_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy()
        wandb.log({"Input Back Images": [wandb.Image(back_img[i]) for i in range(self.b_size)]}, step=step)
        front_nrm = self.pred_front_nrm_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        front_nrm = np.clip(front_nrm, 0, 1)
        wandb.log({"Input Front Nrm": [wandb.Image(front_nrm[i]) for i in range(self.b_size)]}, step=step)
        back_nrm = self.pred_back_nrm_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        back_nrm = np.clip(back_nrm, 0, 1)
        wandb.log({"Input Back Nrm": [wandb.Image(back_nrm[i]) for i in range(self.b_size)]}, step=step)
        gt_front_nrm = self.gt_front_nrm_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        gt_front_nrm = np.clip(gt_front_nrm, 0, 1)
        wandb.log({"GT Front Nrm": [wandb.Image(gt_front_nrm[i]) for i in range(self.b_size)]}, step=step)
        gt_back_nrm = self.gt_back_nrm_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        gt_back_nrm = np.clip(gt_back_nrm, 0, 1)
        wandb.log({"GT Back Nrm": [wandb.Image(gt_back_nrm[i]) for i in range(self.b_size)]}, step=step)



    def step(self, epoch, n_iter, data):
        self.epoch = epoch
        self.global_step = n_iter

        self.optimizer.zero_grad()

        self.b_size = data['front_nrm_img'].shape[0]

        self.gt_front_nrm_img = data['front_nrm_img'].to(self.device)
        self.gt_back_nrm_img = data['back_nrm_img'].to(self.device)

        # We need to flip z-axis of normal map so that it is always front facing
        self.gt_back_nrm_img[:,[2], ...] *= -1

        self.front_rgb_img = data['front_rgb_img'].to(self.device)
        self.back_rgb_img = data['back_rgb_img'].to(self.device)

        self.front_mask = data['front_mask'].to(self.device)
        self.back_mask = data['back_mask'].to(self.device)


        self.pred_front_nrm_img = self.nrm_predictor(self.front_rgb_img) 
        self.pred_back_nrm_img = self.nrm_predictor(self.back_rgb_img) 

        if self.cfg.use_mask:
            nrm_2D_loss = torch.abs(self.gt_front_nrm_img * self.front_mask  - \
                                    self.pred_front_nrm_img * self.front_mask).mean() + \
                          torch.abs(self.gt_back_nrm_img * self.back_mask - \
                                    self.pred_back_nrm_img * self.back_mask).mean()
        else:
            nrm_2D_loss = torch.abs(self.gt_front_nrm_img - self.pred_front_nrm_img).mean() + \
                          torch.abs(self.gt_back_nrm_img - self.pred_back_nrm_img).mean()
        
        self.log_dict['Loss_2D/total_loss'] += nrm_2D_loss.item()
        nrm_2D_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.nrm_params, self.cfg.max_grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()
        self.log_dict['total_iter_count'] += 1



    def save_checkpoint(self, full=True, replace=False):
        """Save the model checkpoint.
        """

        if replace:
            model_fname = os.path.join(self.log_dir, f'normal.pth')
        else:
            model_fname = os.path.join(self.log_dir, f'normal-{self.epoch:04d}.pth')

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
    
        state['nrm_encoder'] = self.nrm_predictor.state_dict()

        log.info(f'Saving model checkpoint to: {model_fname}')
        torch.save(state, model_fname)


    def load_checkpoint(self, fname):
        """Load checkpoint.
        """
        try:
            checkpoint = torch.load(fname, map_location=self.device)
            log.info(f'Loading model checkpoint from: {fname}')
        except FileNotFoundError:
            log.warning(f'No checkpoint found at: {fname}, model randomly initialized.')
            return

        # update meta info
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        self.nrm_predictor.load_state_dict(checkpoint['nrm_encoder'])

        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        log.info(f'Loaded checkpoint at epoch {self.epoch} with global step {self.global_step}.')


