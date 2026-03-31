#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
import os
import argparse
import torch
from Network.Network import HomographyModel
from Test import estimate_homography, StitchRealPano

# Add any python libraries here


def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser() # 
    Parser.add_argument("--ModelPath", dest="ModelPath", default="./Checkpoints/49model.ckpt", help="Path to load latest model from, Default:ModelPath",)
    # Parser.add_argument("--ModelPath", dest="ModelPath", default="/home/chaitanya/git/final_sup_results/Checkpoints/49model.ckpt", help="Path to load latest model from, Default:ModelPath",)
    Parser.add_argument("--BasePath", dest="BasePath", default="./Phase2/Data", help="Path to load images from, Default:BasePath",)
    Parser.add_argument("--ModelType", default="Unsup", help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",)

    
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    ModelType = Args.ModelType

    """
    Read a set of images for Panorama stitching
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HomographyModel(hparams=Args)
    model = model.to(device)
    # Load Weights
    if os.path.exists(Args.ModelPath):
        checkpoint = torch.load(Args.ModelPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {Args.ModelPath}")
    else:
        print(f"Error: Model not found at {Args.ModelPath}")
        return
    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
    
    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    StitchRealPano(Args, model, device)


if __name__ == "__main__":
    main()
