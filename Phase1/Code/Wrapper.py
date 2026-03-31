#!/usr/bin/env python3

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt

import networkx as nx # to find shortest path on the connectivity graph




# Helper Funcs

def maximum_wind(img, ksize=3):
    try:
        from scipy.ndimage import maximum_filter
        return maximum_filter(img, size=ksize, mode='constant', cval=-np.inf)
    except ImportError:
        # Use manual implementation if scipy not available
        pad = ksize // 2
        padded = np.pad(img, pad_width=pad, mode='constant', constant_values=-np.inf)
        output = np.zeros_like(img)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                window = padded[y:y+ksize, x:x+ksize]
                output[y, x] = np.max(window)
        return output


def normalize_points(points):
    
    points = np.array(points, dtype=np.float64)
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    mean_abs_dev = np.mean(np.abs(centered), axis=0)
    scale = 1.0 / (mean_abs_dev + 1e-8)
    
    # Build normalization matrix
    T = np.array([
        [scale[0], 0, -scale[0] * centroid[0]],
        [0, scale[1], -scale[1] * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return T

# visualization

def visualize_corners(img, corners, output_path, threshold=0.01):

    img_vis = img.copy()
    
    corners_norm = corners.copy()
    corners_norm[corners_norm < 0] = 0
    
    if corners_norm.max() > 0:
        corners_norm = corners_norm / corners_norm.max()
    
    corner_coords = np.where(corners_norm > threshold)
    
    for y, x in zip(corner_coords[0], corner_coords[1]):
        color = (0, 0, 255) # red corners
        cv2.circle(img_vis, (x, y), 3, color, -1)
    
    cv2.imwrite(output_path, img_vis)
    print("Saved corner visualization")


def visualize_anms(img, keypoints, output_path):
    img_vis = img.copy()
    
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        color = (0, 255, 0)
        cv2.circle(img_vis, (x, y), 5, color, 2)
        cv2.circle(img_vis, (x, y), 1, (255, 255, 255), -1)
    
    cv2.putText(img_vis, f"features: {len(keypoints)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, img_vis)
    print("Saved ANMS visualization")


def visualize_descriptors(descriptors, output_path, title="Feature Descriptors"):
    # Creates a 2D heatmap of feature descriptors.
    # each row is one descriptor (64 dimensions)
    if len(descriptors) == 0:
        print("No descriptors to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(descriptors, aspect='auto', cmap='viridis', interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('normalized descriptor val', rotation=270, labelpad=20)
    
    ax.set_xlabel('descriptor dim (0-63)')
    ax.set_ylabel('Feature Index')
    ax.set_title(f'{title}\n{descriptors.shape[0]} features x {descriptors.shape[1]} dimensions')
    
    # ax.set_xticks(np.arange(0, 64, 8))
    # ax.set_yticks(np.arange(0, min(descriptors.shape[0], 50), 5))
    # ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"saved descriptor visualization")


def visualize_matches(img1, img2, pts1, pts2, matches, output_path, title="Feature Matches"):
    
    # convert points to cv2.KeyPoint objects
    if len(pts1) > 0 and not isinstance(pts1[0], cv2.KeyPoint):
        kp1 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in pts1]
        kp2 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in pts2]
    else:
        kp1 = pts1
        kp2 = pts2
    
    pts1_tuples = [(int(kp.pt[0]), int(kp.pt[1])) if isinstance(kp, cv2.KeyPoint) else kp for kp in pts1]
    pts2_tuples = [(int(kp.pt[0]), int(kp.pt[1])) if isinstance(kp, cv2.KeyPoint) else kp for kp in pts2]
    
    id1 = {p: i for i, p in enumerate(pts1_tuples)}
    id2 = {p: i for i, p in enumerate(pts2_tuples)}
    
    dm = []
    for match in matches:
        if isinstance(match, cv2.DMatch):
            dm.append(match)
        elif isinstance(match, tuple) and len(match) == 2:
            p1, p2 = match
            p1_tuple = (int(p1[0]), int(p1[1])) if not isinstance(p1, tuple) else p1
            p2_tuple = (int(p2[0]), int(p2[1])) if not isinstance(p2, tuple) else p2
            
            if p1_tuple in id1 and p2_tuple in id2:
                dm.append(cv2.DMatch(id1[p1_tuple], id2[p2_tuple], 0))
    
    # draw matches
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, dm, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    cv2.putText(vis, f"{title}: {len(dm)} matches", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, vis)
    print("Saved match visualization")

# ANMS and feature descriptors and matching

def non_max_sup(corners, n_best=50):

    local_max = (corners == maximum_wind(corners, ksize=3))
    ys, xs = np.where(local_max)
    
    strs = corners[ys, xs]
    top_1000_strs = np.argsort(-strs)[:1000]
    
    ys = ys[top_1000_strs]
    xs = xs[top_1000_strs]
    strs = strs[top_1000_strs]
    
    ns = len(xs)
    radii = np.full(ns, np.inf)
    
    # Compute suppression radius for each point
    for i in range(ns):
        for j in range(ns):
            if strs[j] > strs[i]:
                dist = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                if dist < radii[i]:
                    radii[i] = dist
    
    
    sorted_indices = np.argsort(-radii)
    selected = sorted_indices[:n_best]
    
    # Convert to cv2.KeyPoint objects
    selected_ones = [(xs[i], ys[i]) for i in selected]
    features = cv2.KeyPoint.convert(np.array([[x, y] for x, y in selected_ones], dtype=np.float32))
    
    return features # List of cv2.KeyPoint objects


def feature_vec(points, gray_img):

    patch_size = 40
    k_size = 41 # 5,5 - tune this
    half_patch = patch_size // 2
    
    desc_vec = []
    good_pts = []

    h, w = gray_img.shape
    
    # Blur entire image first (more efficient)
    blur = cv2.GaussianBlur(gray_img, (k_size, k_size), 0, borderType=cv2.BORDER_CONSTANT)
    
    # Pad the blurred image
    pad_width = (k_size - 1) // 2
    img_padded = np.pad(blur, pad_width, mode='constant', constant_values=0)

    
    if len(points) > 0 and isinstance(points[0], cv2.KeyPoint):
        pts = cv2.KeyPoint.convert(points)
    else:
        pts = np.array(points, dtype=np.float32)
    

    for pt in pts:
        if isinstance(pt, cv2.KeyPoint):
            x_c, y_c = int(pt.pt[0]), int(pt.pt[1])
        else:
            x_c, y_c = int(pt[0]), int(pt[1])
        
        
        patch = img_padded[y_c+pad_width-half_patch:y_c+pad_width+half_patch, 
                          x_c+pad_width-half_patch:x_c+pad_width+half_patch]
        
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue
        
        patch = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
        vec = patch.flatten().astype(np.float32)

        # Normalize
        mu = np.mean(vec)
        sigma = np.std(vec)
        vec = (vec - mu) / (sigma + 1e-6)

        desc_vec.append(vec)
        good_pts.append((x_c, y_c))

    return np.array(desc_vec, dtype=np.float32), good_pts


def match_features(desc1, desc2, pts1, pts2, ratio=0.75):
    matches = []
    if len(desc1) < 2 or len(desc2) < 2:
        return matches

    
    vk = np.full(len(desc1), -1)  #match from desc1 to desc2
    vl = np.full(len(desc2), -1)  #match from desc2 to desc1

    for i in range(len(desc1)):
        distances = np.linalg.norm(desc2 - desc1[i], axis=1)

        indices = np.argsort(distances)
        best = distances[indices[0]]
        second_b = distances[indices[1]]

        if second_b < 1e-8:
            continue

        best_idx = indices[0]
        if best / (second_b + 1e-8) < ratio and vl[best_idx] < 0:
            vk[i] = best_idx
            vl[best_idx] = i
            
            matches.append((pts1[i], pts2[best_idx]))

    return matches


# RANSAC and Homography estimation
def homo(pairs):
    pts1 = np.array([p1 for p1, _ in pairs], dtype=np.float64)
    pts2 = np.array([p2 for _, p2 in pairs], dtype=np.float64)
    
    # Normalize coordinates
    T1 = normalize_points(pts1)
    T2 = normalize_points(pts2)
    
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
    
    pts1_norm = (T1 @ pts1_h.T).T[:, :2]
    pts2_norm = (T2 @ pts2_h.T).T[:, :2]
    
    
    A = []
    for i in range(len(pairs)):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A.append([-x1, -y1, -1,   0,   0,  0,  x2*x1, x2*y1, x2])
        A.append([  0,   0,  0, -x1, -y1, -1,  y2*x1, y2*y1, y2])

    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h_norm = Vt[-1].reshape(3, 3)
    
    # convert back
    h = np.linalg.inv(T2) @ h_norm @ T1
    
    # check if denominator is small
    if abs(h[2, 2]) < 1e-10:
        return None

    return h / h[2, 2]


def reproj_errors(h, pairs):

    p1 = np.array([[x1, y1, 1.0] for (x1, y1), _ in pairs], dtype=np.float64)  
    p2 = np.array([[x2, y2]      for _, (x2, y2) in pairs], dtype=np.float64)  
    
    Hp1 = (h @ p1.T).T  
    Hp1 = Hp1[:, :2] / Hp1[:, 2:3]
    
    err = np.linalg.norm(p2 - Hp1, axis=1)
    return err


def ransac(matches, eps=4.0, confidence=0.995, n_iters=2000):

    if len(matches) < 4:
        return None, []
    
    matches = list(matches)
    best_inliers = []
    best_h = None

    n = len(matches)
    rng = np.random.default_rng()
    
    iter_count = 0
    adaptive_iters = n_iters # adaptive iterations

    while iter_count < adaptive_iters:
        iter_count += 1
        
        indices = rng.choice(n, size=4, replace=False)
        sample = [matches[i] for i in indices]

        h = homo(sample)
        if h is None:
            continue

        err = reproj_errors(h, matches)
        inlier_id = np.where(err < eps)[0]
        inliers = [matches[i] for i in inlier_id]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_h = h.copy()

            # adaptive updates
            inlier_ratio = len(best_inliers) / n
            if inlier_ratio > 0.1:
                prob_all_inliers = inlier_ratio ** 4
                
                if prob_all_inliers > 1e-10:
                    adaptive_iters = int(
                        np.log(1 - confidence) / np.log(1 - prob_all_inliers)
                    )
                    adaptive_iters = min(adaptive_iters, n_iters)

    if best_h is None or len(best_inliers) < 4:
        return None, []

    best_h = homo(best_inliers)
    if best_h is None:
        return None, []
    
    return best_h, best_inliers


# Warping and blending

def warp_and_blend_all(image_set, ordered, H_to_ref, ref_idx): # warp all images to reference frame and blend them together.
    
    # get bounding box for all images including reference
    all_corners = []
    for idx in ordered:
        img = image_set[idx]
        h, w = img.shape[:2]
        
        # image corners
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        
        
        if idx == ref_idx: # keep ref corners same
            warped_corners = corners
        else: # transform image corners to ref frame
            H = H_to_ref[idx]
            corners_h = np.hstack([corners, np.ones((4, 1))])
            warped_corners = (H @ corners_h.T).T
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:3]
        
        all_corners.append(warped_corners)
    
    # global bounding box
    all_corners = np.vstack(all_corners)
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    
    print(f"Bounding box: min={min_xy}, max={max_xy}")
    
    tx = -min_xy[0]
    ty = -min_xy[1]
    T = np.array([[1, 0, tx], 
                  [0, 1, ty], 
                  [0, 0, 1]], dtype=np.float64)
    
    pano_w = int(max_xy[0] - min_xy[0]) # + 1
    pano_h = int(max_xy[1] - min_xy[1]) # + 1
    
    print(f"panorama size: {pano_w} x {pano_h}")
    
    # blend the images and make the panorama
    pano = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
    alpha_sum = np.zeros((pano_h, pano_w), dtype=np.float32)

    feather_width = 10  # narrow feather width in pixels
    
    for idx in ordered:
        img = image_set[idx]
        
        if idx == ref_idx:
            # ref frame just translates
            warped = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            warped[ty:ty+h, tx:tx+w] = img
        else:
            H_final = T @ H_to_ref[idx]
            warped = cv2.warpPerspective(img, H_final, (pano_w, pano_h))
        
        
        mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
        
        # alpha with narrow feathering
        kernel = np.ones((feather_width, feather_width), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        border = mask - eroded
        
        alpha = mask.astype(np.float32) / 255.0
        
        if np.any(border > 0):
            border_float = border.astype(np.float32) / 255.0
            border_blurred = cv2.GaussianBlur(border_float, (21, 21), 0)
            alpha = np.where(border > 0, border_blurred, alpha)
        
        pano += warped.astype(np.float32) * alpha[..., None]
        alpha_sum += alpha
        
        print(f"added image {idx} to panorama")
    
    alpha_sum = np.maximum(alpha_sum, 1e-6)
    pano = pano / alpha_sum[..., None]
    pano = np.clip(pano, 0, 255).astype(np.uint8)
    
    return pano


# Order images for stitching in an optimal manner using connectivity graph

def find_optimal_ordering(match_graph, n_images):
    
    # build graph
    G = nx.Graph()
    frame_count = np.zeros(n_images)
    
    for (i, j), data in match_graph.items():
        # higher the inliers better the edge connection and lower the weight
        weight = 1.0 / (data['count'] + 1)  # 1 in denom to avoid / by 0
        G.add_edge(i, j, weight=weight)
        frame_count[i] += 1
        frame_count[j] += 1
    
    # image with most connections is chosen as ref frame
    ref_idx = int(np.argmax(frame_count))
    print(f"ref frame is {ref_idx}")
    
    # Find shortest paths from all nodes to reference frame
    try:
        paths = nx.shortest_path(G, target=ref_idx, weight='weight')
    except nx.NetworkXNoPath:
        print("not all images are connected then use largest connected component")

        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        
        component_counts = {i: frame_count[i] for i in largest_component}
        ref_idx = max(component_counts, key=component_counts.get)
        
        subgraph = G.subgraph(largest_component)
        paths = nx.shortest_path(subgraph, target=ref_idx, weight='weight')
    
    # Order images by path length (images closer to reference come first)
    ordered_with_dist = []
    for node, path in paths.items():
        ordered_with_dist.append((node, len(path)))
    
    ordered_with_dist.sort(key=lambda x: x[1])
    ordered = [node for node, _ in ordered_with_dist]
    
    return ordered, ref_idx, paths



# Create the whole panorama

def stitch_images(image_set, save_intermediate=True, output_dir="Phase1/Code/output_visualizations"):
    
    n_images = len(image_set)

    # Create output directory for visualizations
    if save_intermediate:
        os.makedirs(output_dir, exist_ok=True)
    
    
    match_graph = {}  # make connectivity graph
    
    print("Make connectivyt graph")
    for i in range(n_images):
        for j in range(i + 1, n_images):
            im1 = image_set[i]
            im2 = image_set[j]
            
            gray_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.float32)

            corners_im1 = cv2.cornerHarris(gray_im1, 2, 3, 0.04)
            corners_im2 = cv2.cornerHarris(gray_im2, 2, 3, 0.04)

            # Visualize first corners for first pair
            if save_intermediate and i == 0 and j == 1:
                visualize_corners(im1, corners_im1, 
                                f"{output_dir}/corners_img{i}.png", threshold=0.01)
                visualize_corners(im2, corners_im2, 
                                f"{output_dir}/corners_img{j}.png", threshold=0.01)

            pts_im1 = non_max_sup(corners_im1, n_best=300)
            pts_im2 = non_max_sup(corners_im2, n_best=300)

            # Visualize ANMS for first pair
            if save_intermediate and i == 0 and j == 1:
                visualize_anms(im1, pts_im1, 
                             f"{output_dir}/anms_img{i}.png")
                visualize_anms(im2, pts_im2, 
                             f"{output_dir}/anms_img{j}.png")

            desc1, good_pts1 = feature_vec(pts_im1, gray_im1)
            desc2, good_pts2 = feature_vec(pts_im2, gray_im2)

            # Visualize feature descriptor for first pair
            if save_intermediate and i == 0 and j == 1:
                visualize_descriptors(desc1, 
                                    f"{output_dir}/descriptors_img{i}.png",
                                    title=f"Feature Descriptors - Image {i}")
                visualize_descriptors(desc2, 
                                    f"{output_dir}/descriptors_img{j}.png",
                                    title=f"Feature Descriptors - Image {j}")

            matches = match_features(desc1, desc2, good_pts1, good_pts2, ratio=0.7)
            
            if len(matches) < 10:
                continue

            # Visualize feature matches before RANSAC for fst pair
            if save_intermediate and i == 0 and j == 1:
                visualize_matches(im1, im2, good_pts1, good_pts2, matches,
                                f"{output_dir}/matches_before_ransac_{i}_{j}.png",
                                title=f"Matches Before RANSAC")
                
            h, inliers = ransac(matches, eps=4.0, confidence=0.995, n_iters=2000)
            # Visualize feature matches after RANSAC for first pair
            if save_intermediate and i == 0 and j == 1:
                visualize_matches(im1, im2, good_pts1, good_pts2, inliers,
                                f"{output_dir}/matches_after_ransac_{i}_{j}.png",
                                title=f"Inlier Matches After RANSAC")
            
            if h is not None and len(inliers) >= 30:  # Min inlier thresh (tune this) (10 for customset2)
                match_graph[(i, j)] = {
                    'homography': h,
                    'inliers': inliers,
                    'count': len(inliers)
                }
                print(f"  Images {i}-{j}: {len(inliers)} inliers") #print image correspondance
                
    
    if not match_graph:
        print("No valid matches found!")
        return None
    
    
    
    ordered, ref_idx, paths = find_optimal_ordering(match_graph, n_images)
    
    print(f"\nfinal optimal image order for the set: {ordered}")
    
    
    H_to_ref = {ref_idx: np.eye(3)}
    
    #Get homography relative to ref image for other images (with accumulation)
    
    if paths is not None:
        for idx in ordered:
            if idx == ref_idx:
                continue
            
            path = paths[idx]
            print(f"image {idx} -> ref: path = {path}") #check path from img to ref img
            
            # multiply homographies to accumulate along the path (is it pre or post multiply)
            H = np.eye(3)
            for k in range(len(path) - 1):
                curr = path[k]
                next_node = path[k + 1]
                
                
                pair_key = tuple(sorted([curr, next_node]))
                if pair_key not in match_graph:
                    print(f"  missing edge {curr}-{next_node}")
                    break
                
                h = match_graph[pair_key]['homography']
                
                if pair_key[0] == curr:
                    h_curr_to_next = h
                else:
                    h_curr_to_next = np.linalg.inv(h)
                
                H = h_curr_to_next @ H
            
            H_to_ref[idx] = H
    
    
    
    pano = warp_and_blend_all(image_set, ordered, H_to_ref, ref_idx)
    
    if pano is not None:
        cv2.imwrite(f"{output_dir}/final_panorama.png", pano)
        print("\nPanorama saved")
        cv2.imshow("Final Panorama", pano)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return pano


def main():
    
    curr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    image_set = []
    # image_set_path = Path(os.path.join(curr_dir, "Data/Train/Set1/"))
    image_set_path = Path(os.path.join(curr_dir, "../Phase2/Data/Phase2Pano/trees"))
    # image_set_path = Path(os.path.join(curr_dir, "Data/Train/CustomSet1/"))
    # image_set_path = Path(os.path.join(curr_dir, "Data/Test/TestSet4/"))
    
    # Load images
    for file_path in sorted(image_set_path.iterdir()):
        if file_path.is_file():
            img = cv2.imread(str(file_path))
            if img is not None:
                image_set.append(img)
                print(f"Loaded: {file_path.name}")
    
    if len(image_set) < 2:
        print("Need at least 2 images!")
        return
    
    
    # Create panorama
    pano = stitch_images(image_set)
    
    if pano is None:
        print("Failed to create panorama")
    else:
        print("Panorama creation successful")


if __name__ == "__main__":
    main()
