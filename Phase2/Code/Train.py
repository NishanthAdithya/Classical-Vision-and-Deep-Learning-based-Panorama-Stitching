#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm


def cal_epe(pred, gt):
    """
    calculate the Average Endpoint Error (EPE) (Mean Corner Error).
    Input:
        pred: Predicted H4pt (Batch, 8) or (Batch, 4, 2)
        gt: Ground Truth H4pt (Batch, 4, 2)
    Output:
        scalar EPE value
    """
    # If network outputs (B, 8), reshape it.
    if pred.dim() == 2 and pred.shape[1] == 8:
        pred = pred.view(-1, 4, 2)
    
    # diff = pred - gt
    # dist = sqrt(dx^2 + dy^2)
    error = torch.sqrt(torch.sum((pred - gt)**2, dim=2)) # shape (B, 4)
    
    # mean over all 4 corners and all images in batch
    epe = torch.mean(error)
    return epe

def GenerateBatch(BasePath, DirNames, MiniBatchSize, Mode="Train"):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    b_ia, b_pa, b_pb, b_c, b_gt = [], [], [], [], []

    if len(DirNames) == 0:
        return None

    while len(b_pa) < MiniBatchSize:
        name = random.choice(DirNames)
        img_path = os.path.join(BasePath, name + ".jpg")
        img = cv2.imread(img_path, 0)
        if img is None: continue

        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        pa, pb, gt, corners = GenerateData(img)

        b_ia.append(torch.from_numpy(img).float())
        b_pa.append(torch.from_numpy(pa).float().unsqueeze(0) / 255.) # standardize patches
        b_pb.append(torch.from_numpy(pb).float().unsqueeze(0) / 255.)
        b_c.append(torch.from_numpy(corners).float())
        b_gt.append(torch.from_numpy(gt).float())

    return torch.stack(b_ia), torch.stack(b_pa), torch.stack(b_pb), torch.stack(b_c), torch.stack(b_gt)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    DirNamesVal,
    NumTrainSamples,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
    Args,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel(hparams=Args).float()
    
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=20, gamma=0.5)

    # check for cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")
    
    
    for Epochs in tqdm(range(StartEpoch, NumEpochs), desc="Epochs"):
        ##### Training Phase ######
        model.train()
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)

        epoch_train_loss = 0
        train_epe = 0
        
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch), desc="Train", leave=False):
            Batch = GenerateBatch(BasePath, DirNamesTrain, MiniBatchSize, Mode="Train")
            if Batch is None: continue
            # Batch order: [Img, Pa, Pb, Corners, GT]
            Batch = [item.to(device) for item in Batch]
            Pa, Pb, GT = Batch[1], Batch[2], Batch[4]
            # Batch = [item.to(device) if torch.is_tensor(item) else item for item in Batch]
            # Predict output with forward pass
            LossThisBatch, pred = model.training_step(Batch, PerEpochCounter)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            epoch_train_loss += LossThisBatch.item()

            train_epe += cal_epe(pred, GT).item()

            # Tensorboard
            Writer.add_scalar("Train/LossEveryIter",LossThisBatch.item(),Epochs * NumIterationsPerEpoch + PerEpochCounter,)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()
        train_loss = epoch_train_loss/NumIterationsPerEpoch
        avg_train_epe = train_epe/NumIterationsPerEpoch
        Writer.add_scalar("Train/Loss_Epoch", train_loss, Epochs)
        Writer.add_scalar("Train/EPE", avg_train_epe, Epochs)
        scheduler.step()

        ####Validation Phase##########
        model.eval()
        val_loss = 0
        val_epe = 0
        NumIterationsPerEpoch = int(len(DirNamesVal) / MiniBatchSize)
        # val_results_per_epoch = []
        with torch.no_grad():
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch), desc="Val", leave=False):
                Batch = GenerateBatch(BasePath, DirNamesVal, MiniBatchSize, Mode="Val")
                if Batch is None: break
                Batch = [item.to(device) for item in Batch]
                Pa, Pb, GT = Batch[1], Batch[2], Batch[4]

                result = model.validation_step(Batch, PerEpochCounter)

                if isinstance(result, dict):
                    curr_loss = result['val_loss']
                    pred = model(Pa, Pb)
                else:
                    curr_loss = result
                    pred = model(Pa, Pb)
                
                val_loss += curr_loss.item() if torch.is_tensor(curr_loss) else curr_loss
                val_epe += cal_epe(pred, GT).item()
        avg_val_loss = val_loss / NumIterationsPerEpoch
        avg_val_epe = val_epe / NumIterationsPerEpoch
        
        print(f"Epoch {Epochs} | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val EPE: {avg_val_epe:.4f}")
        Writer.add_scalar("Val/Loss", avg_val_loss, Epochs)
        Writer.add_scalar("Val/EPE", avg_val_epe, Epochs)


        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--BasePath", default="./Phase2/Data", help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",)
    Parser.add_argument("--CheckPointPath", default="./Checkpoints/", help="Path to save Checkpoints, Default: ../Checkpoints/",)
    Parser.add_argument("--ModelType", default="Unsup", help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",)
    Parser.add_argument("--NumEpochs", type=int, default=50, help="Number of Epochs to Train for, Default:50",)
    Parser.add_argument("--DivTrain", type=int, default=1, help="Factor to reduce Train data by per epoch, Default:1",)
    Parser.add_argument("--MiniBatchSize", type=int, default=10, help="Size of the MiniBatch to use, Default:1",)
    Parser.add_argument("--LoadCheckPoint", type=int, default=0, help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",)
    Parser.add_argument("--LogsPath", default="Logs/", help="Path to save Logs for Tensorboard, Default=Logs/",)

    Args = Parser.parse_args()

    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        DirNamesVal,
        SaveCheckPoint,
        NumTrainSamples,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)
    
    TrainOperation(
        DirNamesTrain,
        DirNamesVal,
        NumTrainSamples,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
        Args,
    )


if __name__ == "__main__":
    main()
