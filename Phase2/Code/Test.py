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

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network import HomographyModel
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from Misc.DataUtils import SetupDirNames, ReadDirNames, GenerateData
from Train import GenerateBatch


# Don't generate pyc codes
sys.dont_write_bytecode = True

def cal_epe(pred, gt):
    """
    calculates the Average Endpoint Error (EPE) (Mean Corner Error).
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


def TestOperation(Args, DirNamesTest, model, device):
    
    model.eval()
    total_epe = 0
    total_time = 0
    count = 0
    NumTestSamples = len(DirNamesTest)
    NumBatches = int(NumTestSamples / Args.MiniBatchSize)

    with torch.no_grad():
        for i in tqdm(range(NumBatches), desc="Test"):
            Batch = GenerateBatch(Args.BasePath, DirNamesTest, Args.MiniBatchSize, Mode="Test")
            if Batch is None: break

            Batch = [item.to(device) for item in Batch]
            Pa, Pb, GT = Batch[1], Batch[2], Batch[4]

            if i > 5: # Warmup first 5 batches
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
            
            # Inference
            pred_h4pt = model(Pa, Pb)

            if i > 5:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end = time.time()
                total_time += (end - start)
                count += 1
            
            if pred_h4pt.shape[1] == 8:
                pred_h4pt = pred_h4pt.view(-1, 4, 2)
            
            batch_epe = cal_epe(pred_h4pt, GT)
            total_epe += batch_epe.item()

    avg_epe = total_epe / NumBatches
    avg_time = (total_time / count) * 1000 if count > 0 else 0 # ms per batch
    avg_time_per_image = avg_time / Args.MiniBatchSize

    print(f"Average EPE: {avg_epe:.4f}")
    print(f"Average Runtime: {avg_time:.2f} ms/batch ({avg_time_per_image:.4f} ms/image)")


def VisualizeResults(model, base_path, dir_names, device, modeltype, save_path="./Phase2/Code/Results"):
    """
    Generate comparison visualizations for model results vs Ground Truth.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    # select 4 random samples from the directory names
    samples = np.random.choice(dir_names, 4, replace=False)

    for idx, name in enumerate(samples):
        img_path = os.path.join(base_path, name + ".jpg")
        img = cv2.imread(img_path)
        if img is None: continue
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        patch_a, patch_b, gt_perturbation, corners = GenerateData(img_gray)
        pa_t = torch.from_numpy(patch_a).float().unsqueeze(0).unsqueeze(0) / 255.0
        pb_t = torch.from_numpy(patch_b).float().unsqueeze(0).unsqueeze(0) / 255.0
        pa_t = pa_t.to(device)
        pb_t = pb_t.to(device)

        with torch.no_grad():
            pred = model(pa_t, pb_t).cpu().numpy()
 
        vis_img = img.copy()
        # Ground Truth Corners for B (Green)
        gt_corners_b = corners + gt_perturbation
        # Predicted Corners for B (Supervised = Blue, Unsupervised = Red)
        pred_corners_b = corners + pred

        def draw_poly(pts, color, thickness=2):
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=thickness)

        # draw ground truth
        draw_poly(gt_corners_b, (0, 255, 0), 2)  # Green
        # draw sup or unusp
        if modeltype == "Sup":
            draw_poly(pred_corners_b, (255, 0, 0), 2) # Blue
            cv2.putText(vis_img, "Green: GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_img, "Blue: Supervised", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif modeltype == "Unsup":
            draw_poly(pred_corners_b, (0, 0, 255), 2) # Red
            cv2.putText(vis_img, "Green: GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_img, "Red: Unsupervised", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(save_path, f"comparison_{idx}.jpg"), vis_img)
        print(f"Saved visualization to {save_path}/comparison_{idx}.jpg")

def estimate_homography(img1, img2, model, device, patch_size=128):
    i1_small = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (patch_size, patch_size))
    i2_small = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (patch_size, patch_size))

    t1 = torch.from_numpy(i1_small).float().unsqueeze(0).unsqueeze(0) / 255.0
    t2 = torch.from_numpy(i2_small).float().unsqueeze(0).unsqueeze(0) / 255.0
    t1 = t1.to(device)
    t2 = t2.to(device)
    with torch.no_grad():
        # predict 4-point displacement (H4pt)
        pred_h4pt = model(t1,t2)
    pred_h4pt = pred_h4pt.cpu().numpy()

    # scale the predicted h4pt to orig img size
    h_orig, w_orig = img1.shape[:2]
    scale_x = w_orig / patch_size
    scale_y = h_orig / patch_size

    pred_h4pt_orig = pred_h4pt * np.array([scale_x, scale_y])
    corners_a = np.array([
        [0, 0], 
        [w_orig, 0], 
        [w_orig, h_orig], 
        [0, h_orig]
    ], dtype=np.float32)

    corners_b = corners_a + pred_h4pt_orig
    corners_b = corners_b.astype(np.float32)
    H_matrix = cv2.getPerspectiveTransform(corners_a, corners_b)
    
    
    return H_matrix


def StitchRealPano(Args, model, device):

    PanoPath = os.path.join(Args.BasePath, "Phase2Pano/unity_hall/")
    OutputPath = "./Phase2/Code/Results/unity_hall/"
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    image_paths = sorted(glob.glob(os.path.join(PanoPath, "*.jpg")))
    image_paths = image_paths[::5]
    images = [cv2.imread(p) for p in image_paths]
    num_images = len(images)

    center_idx = num_images // 2 
    
    print(f"stitching {num_images} images. referemce is image {center_idx}")

    H_global = [np.eye(3) for _ in range(num_images)]

    
    def is_valid_homography(H, h_img, w_img):
        # check if determinant is too small (singular)
        if abs(np.linalg.det(H)) < 1e-5: return False
        
        corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H)
        
        if np.max(np.abs(warped)) > 20000: 
            return False
        return True

    # left side to center reference
    for i in range(center_idx - 1, -1, -1):
        H_local = estimate_homography(images[i], images[i+1], model, device)
         
        H_candidate = np.matmul(H_global[i+1], H_local)
        H_candidate /= H_candidate[2, 2]

        if is_valid_homography(H_candidate, images[i].shape[0], images[i].shape[1]):
            H_global[i] = H_candidate
        else:
            print(f"dropping bad homography for Image {i}")
            H_global[i] = H_global[i+1] # fallback to previous one

    # right side to center reference
    for i in range(center_idx + 1, num_images):
        H_local = estimate_homography(images[i], images[i-1], model, device)
  
        H_candidate = np.matmul(H_global[i-1], H_local)
        H_candidate /= H_candidate[2, 2]

        if is_valid_homography(H_candidate, images[i].shape[0], images[i].shape[1]):
            H_global[i] = H_candidate
        else:
            print(f"dropping bad homography for Image {i}")
            H_global[i] = H_global[i-1]

    # panorama total size calculation
    all_corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H_global[i])
        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    
    [xmin, ymin] = all_corners.min(axis=0).ravel()
    [xmax, ymax] = all_corners.max(axis=0).ravel()
    
    if (xmax - xmin) > 30000 or (ymax - ymin) > 30000:
        print("panorama is too large. memory issues")
        return

    xmin, ymin = int(xmin - 0.5), int(ymin - 0.5)
    xmax, ymax = int(xmax + 0.5), int(ymax + 0.5)

    width_pano = xmax - xmin
    height_pano = ymax - ymin
    
    print(f"final canvas Size: {width_pano} x {height_pano}")

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    panorama = np.zeros((height_pano, width_pano, 3), dtype=np.uint8)

    for i in range(num_images):
        H_final = np.matmul(Ht, H_global[i])
        warped_img = cv2.warpPerspective(images[i], H_final, (width_pano, height_pano))
        mask = (warped_img > 0)
        panorama[mask] = warped_img[mask]

    cv2.imwrite(os.path.join(OutputPath, "unity_hall.jpg"), panorama)
    print("Saved panorama")


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser() # 
    Parser.add_argument("--ModelPath", dest="ModelPath", default="./Checkpoints/49model.ckpt", help="Path to load latest model from, Default:ModelPath",)
    # Parser.add_argument("--ModelPath", dest="ModelPath", default="/home/chaitanya/git/final_sup_results/Checkpoints/49model.ckpt", help="Path to load latest model from, Default:ModelPath",)
    Parser.add_argument("--BasePath", dest="BasePath", default="./Phase2/Data", help="Path to load images from, Default:BasePath",)
    Parser.add_argument("--MiniBatchSize", type=int, default=10, help="Size of the MiniBatch to use, Default:10",)
    Parser.add_argument("--ModelType", default="Unsup", help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",)

    
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    MiniBatchSize = Args.MiniBatchSize
    ModelType = Args.ModelType

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
    
    # Get Test set names
    CodePath = os.path.dirname(os.path.abspath(__file__))
    TestTxtPath = os.path.join(CodePath, "TxtFiles", "DirNamesTest.txt")
    if os.path.exists(TestTxtPath):
        DirNamesTest = ReadDirNames(TestTxtPath)
        TestOperation(Args, DirNamesTest, model, device)
    else:
        print("DirNamesTest.txt not found")
    

    VisualizeResults(model, BasePath, DirNamesTest, device, ModelType)


if __name__ == "__main__":
    main()
