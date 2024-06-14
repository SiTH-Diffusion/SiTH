"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import os
import logging as log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

from .geo_model import GeoModel
from .tex_model import TexModel
from .SMPL_query import SMPL_query
from .networks.normal_predictor import define_G
from diffusers.optimization import get_scheduler


class Trainer(nn.Module):

    def __init__(self, config, log_dir, watertight, can_V):
        # Set device to use
        super().__init__()

        self.device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.cfg = config
        self.use_pred_nrm = config.use_pred_nrm
        self.log_dir = log_dir
        self.log_dict = {}

        self.epoch = 0
        self.global_step = 0
        self.smpl_query = SMPL_query(watertight['smpl_F'], can_V)

        self._init_model()
        self._init_optimizer()
        self._init_log_dict()


    def _init_model(self):
        """Initialize model.
        """

        self.geo_model = GeoModel(self.cfg, self.smpl_query)
        self.tex_model = TexModel(self.cfg, self.smpl_query)

        self.geo_model.to(self.device)
        self.tex_model.to(self.device)
        
        self.nrm_predictor = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

        if self.use_pred_nrm:
            self.nrm_predictor.to(self.device)

    def _init_optimizer(self):
        """Initialize optimizer.
        """


        self.decoder_params = []
        self.decoder_params.extend(list(self.geo_model.sdf_decoder.parameters()))
        self.decoder_params.extend(list(self.tex_model.rgb_decoder.parameters()))

        self.encoder_params = []
        self.encoder_params.extend(list(self.geo_model.image_encoder.parameters()))
        self.decoder_params.extend(list(self.tex_model.image_encoder.parameters()))
        
        params = []
        params.append({'params': self.decoder_params,
                          'lr': self.cfg.lr_decoder})
        params.append({'params': self.encoder_params,
                          'lr': self.cfg.lr_encoder})

        if self.use_pred_nrm:
            self.nrm_params = list(self.nrm_predictor.parameters())
            params.append({'params': self.nrm_params,
                           'lr': self.cfg.lr_encoder * 0.01})
            
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
        self.log_dict['Loss_3D/reco_loss'] = 0
        self.log_dict['Loss_3D/rgb_loss'] = 0
        self.log_dict['Loss_3D/nrm_loss'] = 0
        self.log_dict['Loss_3D/total_loss'] = 0

        self.log_dict['Loss_2D/nrm_2D_loss'] = 0
        self.log_dict['Loss_2D/total_loss'] = 0
        self.log_dict['total_loss'] = 0
        self.log_dict['total_iter_count'] = 0

    def _set_inputs(self, data):
        """Set inputs for training.
        """
        self.b_size, self.n_vertice, _ = data['pts'].shape
        self.pts = data['pts'].to(self.device)
        self.gts = data['d'].to(self.device)
        self.rgb = data['rgb'].to(self.device) 
        self.nrm = data['nrm'].to(self.device)

        self.smpl_v = data['smpl_v'].to(self.device)
        self.vis_class = data['vis_class'].to(self.device)

        self.front_mask = data['front_mask'].to(self.device)
        self.back_mask = data['back_mask'].to(self.device)

        self.front_rgb_img = data['front_rgb_img'].to(self.device)
        self.back_rgb_img = data['back_rgb_img'].to(self.device)

        self.gt_front_nrm_img = data['front_nrm_img'].to(self.device)
        self.gt_back_nrm_img = data['back_nrm_img'].to(self.device)
        self.gt_back_nrm_img[:,[2], ...] *= -1

        if self.use_pred_nrm:
            self.front_nrm_img = self.nrm_predictor(self.front_rgb_img)
            self.back_nrm_img = self.nrm_predictor(self.back_rgb_img)
        else: 
            self.front_nrm_img = self.gt_front_nrm_img
            self.back_nrm_img = self.gt_back_nrm_img

        self.front_nrm_feat, self.back_nrm_feat = \
                self.geo_model.compute_feat_map(self.front_rgb_img, self.back_rgb_img, self.front_nrm_img, self.back_nrm_img)

        self.front_img_feat, self.back_img_feat = self.tex_model.compute_feat_map(self.front_rgb_img, self.back_rgb_img)

    def _forward_3D(self):
        """Forward pass for 3D.
            predict sdf, rgb, nrm
        """

        self.pred_sdf, self.pred_nrm = self.geo_model.forward(
                       self.front_nrm_feat, self.back_nrm_feat, self.pts, self.smpl_v, self.vis_class)
        self.pred_rgb = self.tex_model.forward(
                        self.front_img_feat, self.back_img_feat, self.pts, self.smpl_v, self.vis_class)

    def _backward_3D(self):

        total_loss = 0.0
        reco_loss = 0.0
        rgb_loss = 0.0
        nrm_loss = 0.0

        # Compute 3D losses
        reco_loss += torch.abs(self.pred_sdf - self.gts).mean()

        rgb_loss += torch.abs(self.pred_rgb - self.rgb).mean()

        nrm_loss += torch.abs(1 - F.cosine_similarity(self.pred_nrm, self.nrm, dim=-1)).mean()

        # Compute 2D losses
        nrm_2D_loss = torch.tensor(0.0).to(self.device)
        if self.use_pred_nrm:
            nrm_2D_loss += torch.abs(self.gt_front_nrm_img - self.front_nrm_img).mean() + \
                            torch.abs(self.gt_back_nrm_img - self.back_nrm_img).mean()
        
        # Compute total loss
        loss_3D = reco_loss * self.cfg.lambda_sdf + \
                      rgb_loss * self.cfg.lambda_rgb + \
                      nrm_loss * self.cfg.lambda_nrm
        
        loss_2D =   nrm_2D_loss * self.cfg.lambda_2D

        total_loss = loss_3D + loss_2D

        # Update logs
        self.log_dict['Loss_3D/reco_loss'] += reco_loss.item()
        self.log_dict['Loss_3D/rgb_loss'] += rgb_loss.item()
        self.log_dict['Loss_3D/nrm_loss'] += nrm_loss.item()
        self.log_dict['Loss_3D/total_loss'] += loss_3D.item()

        self.log_dict['Loss_2D/nrm_2D_loss'] += nrm_2D_loss.item()
        self.log_dict['Loss_2D/total_loss'] += loss_2D.item()
        self.log_dict['total_loss'] += total_loss.item()

        total_loss.backward()

    def step(self, epoch, n_iter, data):
        """Training step.
            1. 3D forward
            2. 3D backward
            3. 2D forward
            4. 2D backward
        """
        # record stats
        self.epoch = epoch
        self.global_step = n_iter

        # Set inputs to device
        self._set_inputs(data)

        # Train
        self.optimizer.zero_grad()
        self._forward_3D()
        self._backward_3D()

        torch.nn.utils.clip_grad_norm_(self.decoder_params, self.cfg.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.encoder_params, self.cfg.max_grad_norm)
        if self.use_pred_nrm:
            torch.nn.utils.clip_grad_norm_(self.nrm_params, self.cfg.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.log_dict['total_iter_count'] += 1


    def log(self, step, epoch):
        """Log the training information.
        """
        log_text = 'STEP {} - EPOCH {}/{}'.format(step, epoch, self.cfg.epochs)
        self.log_dict['Loss_3D/total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | 3D loss: {:>.3E}'.format(self.log_dict['Loss_3D/total_loss'])
        self.log_dict['Loss_3D/reco_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | reco loss: {:>.3E}'.format(self.log_dict['Loss_3D/reco_loss'])
        self.log_dict['Loss_3D/rgb_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['Loss_3D/rgb_loss'])
        self.log_dict['Loss_3D/nrm_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | nrm loss: {:>.3E}'.format(self.log_dict['Loss_3D/nrm_loss'])
        self.log_dict['Loss_2D/total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | 2D loss: {:>.3E}'.format(self.log_dict['Loss_2D/total_loss'])
        self.log_dict['Loss_2D/nrm_2D_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | nrm 2D loss: {:>.3E}'.format(self.log_dict['Loss_2D/nrm_2D_loss'])
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        log_text += ' | lr: {:>.3E}'.format(self.lr_scheduler.get_last_lr()[0])
        log.info(log_text)

        for key, value in self.log_dict.items():
            wandb.log({key: value}, step=step)

        self._init_log_dict()

        front_img = self.front_rgb_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy()
        wandb.log({"Input Front Images": [wandb.Image(front_img[i]) for i in range(self.b_size)]}, step=step)
        back_img = self.back_rgb_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy()
        wandb.log({"Input Back Images": [wandb.Image(back_img[i]) for i in range(self.b_size)]}, step=step)
        front_nrm = self.front_nrm_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        front_nrm = np.clip(front_nrm, 0, 1)
        wandb.log({"Input Front Nrm": [wandb.Image(front_nrm[i]) for i in range(self.b_size)]}, step=step)
        back_nrm = self.back_nrm_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        back_nrm = np.clip(back_nrm, 0, 1)
        wandb.log({"Input Back Nrm": [wandb.Image(back_nrm[i]) for i in range(self.b_size)]}, step=step)
        gt_front_nrm = self.gt_front_nrm_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        gt_front_nrm = np.clip(gt_front_nrm, 0, 1)
        wandb.log({"GT Front Nrm": [wandb.Image(gt_front_nrm[i]) for i in range(self.b_size)]}, step=step)
        gt_back_nrm = self.gt_back_nrm_img.clone().detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        gt_back_nrm = np.clip(gt_back_nrm, 0, 1)
        wandb.log({"GT Back Nrm": [wandb.Image(gt_back_nrm[i]) for i in range(self.b_size)]}, step=step)


    def save_checkpoint(self, full=True, replace=False):
        """Save the model checkpoint.
        """

        if replace:
            model_fname = os.path.join(self.log_dir, f'model.pth')
        else:
            model_fname = os.path.join(self.log_dir, f'model-{self.epoch:04d}.pth')

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
    
        state['nrm_encoder'] = self.nrm_predictor.state_dict()
        
        state['image_encoder'] = self.geo_model.image_encoder.state_dict()
        state['sdf_decoder'] = self.geo_model.sdf_decoder.state_dict()

        state['high_res_encoder'] = self.tex_model.image_encoder.state_dict()
        state['rgb_decoder'] = self.tex_model.rgb_decoder.state_dict()

        if self.nrm_predictor is not None:
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

        basename = os.path.basename(fname)
        if 'normal' in basename:
            self.nrm_predictor.load_state_dict(checkpoint['nrm_encoder'])
        else:
            self.nrm_predictor.load_state_dict(checkpoint['nrm_encoder'])

            self.geo_model.image_encoder.load_state_dict(checkpoint['image_encoder'])
            self.geo_model.sdf_decoder.load_state_dict(checkpoint['sdf_decoder'])

            self.tex_model.image_encoder.load_state_dict(checkpoint['high_res_encoder'])
            self.tex_model.rgb_decoder.load_state_dict(checkpoint['rgb_decoder'])


        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        log.info(f'Loaded checkpoint at epoch {self.epoch} with global step {self.global_step}.')


