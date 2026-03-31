"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
import pytorch_lightning as pl
import argparse

# Don't generate pyc codes
sys.dont_write_bytecode = True

class HomographyModel(pl.LightningModule):
    def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.net = Net()
        
        self.loss_fn_sup = nn.MSELoss()
        self.loss_fn_unsup = nn.L1Loss() # Photometric loss

    def forward(self, a, b):
        return self.net(a, b)
    
    def tensor_dlt(self, src_pts, dst_pts):
        """
        Docstring for tensor_dlt
        
        :param self: Description
        :param src_pts: (Batch, 4, 2) - original corners
        :param dst_pts: (Batch, 4, 2) - predicted corners
        """
        batch_size = src_pts.shape[0]

        A = torch.zeros((batch_size, 8, 8), device=self.device)
        b = torch.zeros((batch_size, 8, 1), device=self.device)

        for i in range(4):
            x, y = src_pts[:, i, 0], src_pts[:, i, 1]
            xp, yp = dst_pts[:, i, 0], dst_pts[:, i, 1]

            A[:, 2*i, 0] = x
            A[:, 2*i, 1] = y
            A[:, 2*i, 2] = 1
            A[:, 2*i, 6] = -x * xp
            A[:, 2*i, 7] = -y * xp
            b[:, 2*i, 0] = xp

            A[:, 2*i + 1, 3] = x
            A[:, 2*i + 1, 4] = y
            A[:, 2*i + 1, 5] = 1
            A[:, 2*i + 1, 6] = -x * yp
            A[:, 2*i + 1, 7] = -y * yp
            b[:, 2*i + 1, 0] = yp
        
        h_8 = torch.linalg.solve(A, b) # we get (b, 8, 1)
        h_one = torch.ones((batch_size, 1, 1), device=self.device)

        h_9 = torch.cat((h_8, h_one), dim=1) # get (b, 9, 1)

        return h_9.view(-1,3,3)

    
    def spatial_transformer(self, img, H, patch_w = 128, patch_h = 128):
        batch_size = img.shape[0]

        x_base = torch.linspace(-1, 1, patch_w).repeat(patch_h, 1)
        y_base = torch.linspace(-1, 1, patch_h).unsqueeze(1).repeat(1, patch_w)

        grid = torch.stack((x_base, y_base, torch.ones_like(x_base))).view(3, -1).to(self.device)
        # (3, h * w) where for each pixel in (h,w) we have 3 dim vector of (x, y, 1) on grid
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)

        Hinv = torch.inverse(H)
        # Hinv (b, 3, 3) and grid (b, 3, h*w) should give (b, 3, h*w)
        warped_grid = torch.bmm(Hinv, grid)

        # normalize warped x and y with z
        xw = warped_grid[:, 0, :]
        yw = warped_grid[:, 1, :]
        zw = warped_grid[:, 2, :] + 1e-8

        xnorm = xw / zw
        ynorm = yw / zw
        # is it dim=2 or -1
        sample_grid = torch.stack((xnorm, ynorm), dim=2).view(batch_size, patch_h, patch_w, 2) # convert to (B, H, W, 2) shape
        # print("norm", sample_grid.shape)
        img_warped = F.grid_sample(img, sample_grid, align_corners=True, mode='bilinear')

        return img_warped

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt_h4pt = batch
        batch_size = patch_a.shape[0]
        pred_h4pt = self.net(patch_a, patch_b)

        if self.hparams.ModelType == "Sup":
            loss = self.loss_fn_sup(pred_h4pt.view(batch_size, -1), gt_h4pt.view(batch_size, -1))
            logs = {"loss": loss}

        elif self.hparams.ModelType == "Unsup":
            # predicted corners and Ca
            # corners_a = corners #Do not use absolute corners
            corners_a = torch.tensor([
                [0.0, 0.0],
                [128.0, 0.0],
                [128.0, 128.0],
                [0.0, 128.0]
            ], device=self.device).repeat(batch_size, 1, 1) # [B, 4, 2]
            corners_b_pred = corners_a + pred_h4pt
            # DLT
            H_matrix = self.tensor_dlt(corners_a, corners_b_pred)
            # H_matrix = kornia.geometry.get_perspective_transform(corners_a, corners_b_pred)
            
            # Spatial transformer Network
            warped_patch_a = self.spatial_transformer(patch_a, H_matrix, 128, 128)
            # warped_patch_a = kornia.geometry.transform.warp_perspective(patch_a, H_matrix, (128,128), mode="bilinear",align_corners=True)
            # Photometric loss (with or without crop??)
            crop = 10 # to remove the black border artefacts
            loss = self.loss_fn_unsup(warped_patch_a[:, :, crop:-crop, crop:-crop], patch_b[:, :, crop:-crop, crop:-crop])
            logs = {"loss": loss}
        return loss, pred_h4pt

    def validation_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt_h4pt = batch
        batch_size = patch_a.shape[0]
        pred_h4pt = self.net(patch_a, patch_b)

        if self.hparams.ModelType == "Sup":
            loss = self.loss_fn_sup(pred_h4pt.view(batch_size, -1), gt_h4pt.view(batch_size, -1))
        elif self.hparams.ModelType == "Unsup":
            # predicted corners and Ca
            # corners_a = corners
            corners_a = torch.tensor([
                [0.0, 0.0],
                [128.0, 0.0],
                [128.0, 128.0],
                [0.0, 128.0]
            ], device=self.device).repeat(batch_size, 1, 1) # [B, 4, 2]
            corners_b_pred = corners_a + pred_h4pt
            # DLT
            H_matrix = self.tensor_dlt(corners_a, corners_b_pred)
            # H_matrix = kornia.geometry.get_perspective_transform(corners_a, corners_b_pred)
            
            # Spatial transformer Network
            warped_patch_a = self.spatial_transformer(patch_a, H_matrix, 128, 128)
            # warped_patch_a = kornia.geometry.transform.warp_perspective(patch_a, H_matrix, (128,128), mode="bilinear",align_corners=True)

            # Photometric loss (with or without crop??)
            loss = self.loss_fn_unsup(warped_patch_a, patch_b)
        return {"val_loss": loss}




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, max_pool = True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
        else:
            self.layers.append(nn.ReLU(inplace=True))
        
        self.layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
        else:
            self.layers.append(nn.ReLU(inplace=True))
        
        if max_pool:
            self.layers.append(nn.MaxPool2d(2, 2))
        

    def forward(self, x):
        return self.layers(x)

class Net(nn.Module):
    def __init__(self, InputSize=2, OutputSize=8, batch_norm=True):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        ############################# 
        self.conv_net = nn.Sequential(
            ConvBlock(InputSize, 64, batch_norm=batch_norm),
            ConvBlock(64, 64, batch_norm=batch_norm),
            ConvBlock(64, 128, batch_norm=batch_norm),
            ConvBlock(128, 128,batch_norm=batch_norm, max_pool=False),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, OutputSize),
        )
        nn.init.zeros_(self.fc_layers[-1].weight)
        nn.init.zeros_(self.fc_layers[-1].bias)

    def forward(self, xa, xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        x = torch.cat([xa, xb], dim=1)
        out = self.conv_net(x)
        out = self.fc_layers(out)
        return out.view(-1, 4, 2)

