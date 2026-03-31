"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
from pathlib import Path

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll(BasePath, CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    """
    # Setup DirNames
    DirNamesTrain, DirNamesVal = SetupDirNames(BasePath)
    
    # If CheckPointPath doesn't exist make the path
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 2500

    NumTrainSamples = len(DirNamesTrain)

    return (
        DirNamesTrain,
        DirNamesVal,
        SaveCheckPoint,
        NumTrainSamples,
    )

def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    CodePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(CodePath)
    PathToTxt = os.path.join(CodePath, "TxtFiles")
    DirNamesTrain = ReadDirNames(os.path.join(PathToTxt, "DirNamesTrain.txt"))
    DirNamesVal = ReadDirNames(os.path.join(PathToTxt, "DirNamesVal.txt"))

    return DirNamesTrain, DirNamesVal


def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames


def GenerateData(img, patch_size=128, rho=32):#rho=32
    """
    Generate data from an image
    """
    h, w = img.shape[:2]
    t = 10
    # ensure patch + perturbation stays within image boundaries
    p = rho + t

    # if h < (patch_size + 2*p) or w < (patch_size + 2*p):
    #     # Resize if too small, or return None to handle in GenerateBatch
    #     img = cv2.resize(img, (patch_size + 2*p + 10, patch_size + 2*p + 10))
    #     h, w = img.shape[:2]

    x = np.random.randint(p, w - patch_size - p)
    y = np.random.randint(p, h - patch_size - p)

    #corners of Patch A
    corners = np.array([[x, y], [x+patch_size, y], [x+patch_size, y+patch_size], [x, y+patch_size]], dtype=np.float32)
    
    # perturb corners for Patch B
    perturbation = np.random.randint(-rho, rho, size=(4, 2)).astype(np.float32)
    warped_corners = corners + perturbation
    
    # get transform and extract patches
    H = cv2.getPerspectiveTransform(corners, warped_corners)
    warped_img = cv2.warpPerspective(img, np.linalg.inv(H), (w, h))
    
    patch_a = img[y:y+patch_size, x:x+patch_size]
    patch_b = warped_img[y:y+patch_size, x:x+patch_size]
    
    return patch_a, patch_b, perturbation, corners




