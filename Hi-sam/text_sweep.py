import json
import sys
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import skimage
import os
import argparse

from hi_sam.modeling.build import model_registry
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
from hi_sam.modeling.predictor import SamPredictor
import gc


import glob
from tqdm import tqdm
from PIL import Image
import random
from utils import utilities
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree

import numpy as np
from scipy.spatial import distance
from shapely.geometry import Polygon
import pyclipper
import datetime
import warnings
from plantcv import plantcv as pcv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from scipy.interpolate import BSpline, make_interp_spline
from collections import defaultdict, deque
from scipy.interpolate import CubicSpline
from fil_finder import FilFinder2D
import astropy.units as u
import heapq
import pandas as pd
from math import atan2, degrees



import time
from tqdm import tqdm
warnings.filterwarnings("ignore")\







def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", default=True)
    parser.add_argument("--dataset", type=str, required=True, default='totaltext',
                        help="'totaltext' or 'ctw1500', or 'ic15'.")
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--zero_shot", action='store_true')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')
    parser.add_argument('--layout_thresh', type=float, default=0.5)
    return parser.parse_args()



def get_default_args():

    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, nargs="+", default=["./sample_image.jpg"],
                        help="Path to the input image")  # Default to a sample image
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_h",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default="Hi-sam/pretrained_checkpoint/word_detection_totaltext.pth",
                        help="The path to the SAM checkpoint to use for mask generation.")  # Default checkpoint
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", default=True, type=bool)
    parser.add_argument("--dataset", type=str, default="totaltext",
                        help="'totaltext' or 'ctw1500', or 'ic15'.")
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--zero_shot", action='store_true')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)

    # Self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image-to-token cross-attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int,
                        help='The number of prompt tokens')
    parser.add_argument('--layout_thresh', type=float, default=0.5)

    return parser.parse_args([])  # Pass an empty list to use defaults


def get_default_args_word_model():

    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, nargs="+", default=["./sample_image.jpg"],
                        help="Path to the input image")  # Default to a sample image
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_h",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default="/home/jazz/Harish_ws/Demo/Hi-sam/pretrained_checkpoint/line_detection_ctw1500.pth",
                        help="The path to the SAM checkpoint to use for mask generation.")  # Default checkpoint
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", default=True, type=bool)
    parser.add_argument("--dataset", type=str, default="ctw1500",
                        help="'totaltext' or 'ctw1500', or 'ic15'.")
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--zero_shot", action='store_true')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)

    # Self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image-to-token cross-attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int,
                        help='The number of prompt tokens')
    parser.add_argument('--layout_thresh', type=float, default=0.5)

    return parser.parse_args([])  # Pass an empty list to use defaults


def get_args_parser_letter_model(args=None,file_path="smile.png"):
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, nargs="+", default=[file_path],
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default="Hi-sam/pretrained_checkpoint/sam_tss_l_textseg.pth",
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", action='store_true',
                        help="If False, only text stroke segmentation.")

    parser.add_argument('--input_size', type=int, nargs=2, default=[1024, 1024],
                        help="Input size as two integers: width height.")
    parser.add_argument('--patch_mode', action='store_true',
                        help="Enable patch mode.")

    # Self-prompting parameters
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image-to-token cross-attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int,
                        help='The number of prompt tokens')

    # Parse an empty list to return defaults
    return parser.parse_args([])



def show_hi_masks(masks, word_masks, input_points, filename, image, scores):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    for i, (line_para_masks, word_mask, hi_score, point) in enumerate(zip(masks, word_masks, scores, input_points)):
        line_mask = line_para_masks[0]
        para_mask = line_para_masks[1]
        show_mask(para_mask, plt.gca(), color=np.array([255 / 255, 144 / 255, 30 / 255, 0.5]))
        show_mask(line_mask, plt.gca())
        word_mask = word_mask[0].astype(np.uint8)
        contours, _ = cv2.findContours(word_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        select_word = None
        for cont in contours:
            epsilon = 0.002 * cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            pts = unclip(points)
            if len(pts) != 1:
                continue
            pts = pts[0].astype(np.int32)
            if cv2.pointPolygonTest(pts, (int(point[0]), int(point[1])), False) >= 0:
                select_word = pts
                break
        if select_word is not None:
            word_mask = cv2.fillPoly(np.zeros(word_mask.shape), [select_word], 1)
            show_mask(word_mask, plt.gca(), color=np.array([30 / 255, 255 / 255, 144 / 255, 0.5]))
        show_points(point, plt.gca())
        # print(f'point {i}: line {hi_score[1]}, para {hi_score[2]}')

    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()




def patchify(image: np.array, patch_size: int=256):
    h, w = image.shape[:2]
    patch_list = []
    h_num, w_num = h//patch_size, w//patch_size
    h_remain, w_remain = h%patch_size, w%patch_size
    row, col = h_num + int(h_remain>0), w_num + int(w_remain>0)
    h_slices = [[r * patch_size, (r + 1) * patch_size] for r in range(h_num)]
    if h_remain:
        h_slices = h_slices + [[h - h_remain, h]]
    h_slices = np.tile(h_slices, (1, col)).reshape(-1, 2).tolist()
    w_slices = [[i * patch_size, (i + 1) * patch_size] for i in range(w_num)]
    if w_remain:
        w_slices = w_slices + [[w-w_remain, w]]
    w_slices = w_slices * row
    assert len(w_slices) == len(h_slices)
    for idx in range(0, len(w_slices)):
        # from left to right, then from top to bottom
        patch_list.append(image[h_slices[idx][0]:h_slices[idx][1], w_slices[idx][0]:w_slices[idx][1], :])
    return patch_list, row, col


def unpatchify(patches, row, col):
    # return np.array
    whole = [np.concatenate(patches[r*col : (r+1)*col], axis=1) for r in range(row)]
    whole = np.concatenate(whole, axis=0)
    return whole



def save_binary_mask(mask: np.array, filename):
    if len(mask.shape) == 3:
        assert mask.shape[0] == 1
        mask = mask[0].astype(np.uint8)*255
    elif len(mask.shape) == 2:
        mask = mask.astype(np.uint8)*255
    else:
        raise NotImplementedError
    mask = Image.fromarray(mask)
    mask.save(filename)

def patchify_sliding(image: np.array, patch_size: int=512, stride: int=256):
    h, w = image.shape[:2]
    patch_list = []
    h_slice_list = []
    w_slice_list = []
    for j in range(0, h, stride):
        start_h, end_h = j, j+patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i+patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            h_slice_list.append(h_slice)
            w_slice = slice(start_w, end_w)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])

    return patch_list, h_slice_list, w_slice_list


def unpatchify_sliding(patch_list, h_slice_list, w_slice_list, ori_size):
    assert len(ori_size) == 2  # (h, w)
    whole_logits = np.zeros(ori_size)
    assert len(patch_list) == len(h_slice_list)
    assert len(h_slice_list) == len(w_slice_list)
    for idx in range(len(patch_list)):
        h_slice = h_slice_list[idx]
        w_slice = w_slice_list[idx]
        whole_logits[h_slice, w_slice] += patch_list[idx]

    return whole_logits


def show_points(coords, ax, marker_size=200):
    ax.scatter(coords[0], coords[1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = color if color is not None else np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def polygon2rbox(polygon, image_height, image_width):
    rect = cv2.minAreaRect(polygon)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1, 2)
    return pts


def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = color if color is not None else np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def skeletonize_mask(mask):
    """
    Tasks a mask of an image and returns a mask for the skeleton.
    
    """

    binary_mask = (mask > 0).astype(np.uint8) * 255
    skeleton = pcv.morphology.skeletonize(mask=binary_mask)

    return skeleton




def extract_skeleton_points(skeleton_mask):
    """
    Extract points from the skeleton mask.

    Returns:
        list: List of (x, y) coordinates of skeleton points.
    """
    points = np.argwhere(skeleton > 0)  # Get non-zero points
    return [(x, y) for y, x in points]  # Swap to (x, y) format





    
def find_endpoints(skeleton_points):
    """
    Extract end points from skeleton points.
    Uses a search radius to find points with only one neighbour
    """
    endpoints = []
    for point in skeleton_points:
            x, y = point
            # Define an 8-connected neighborhood
            neighborhood = skeleton_points[
                (skeleton_points[:, 0] >= x - 1) & (skeleton_points[:, 0] <= x + 1) &
                (skeleton_points[:, 1] >= y - 1) & (skeleton_points[:, 1] <= y + 1)
            ]
            # Endpoint has only one neighbor
            if len(neighborhood) == 2:
                endpoints.append((x, y))

    return np.array(endpoints)



def calculate_bend_angle(left_closest, left_center, right_closest):
    dx1 = left_center[0] - left_closest[0]
    dy1 = left_center[1] - left_closest[1]
    dx2 = right_closest[0] - left_closest[0]
    dy2 = right_closest[1] - left_closest[1]
    mag1 = math.sqrt(dx1**2 + dy1**2)
    mag2 = math.sqrt(dx2**2 + dy2**2)
    if mag1 == 0 or mag2 == 0:
        return 180.0
    dot = dx1 * dx2 + dy1 * dy2
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))



def calculate_angle(point1, point2):
    """Calculate the angle of the vector between two points."""
    vector = point2 - point1
    angle = np.arctan2(vector[1], vector[0])
    return angle
    

def find_pairs(image,masks,dir):
    
    """
    Finds best matching pairs from the image masks, using angle and distance.


            right = skeleton_points_all[pair[0]]
            left = skeleton_points_all[pair[1]]
            left_endpoints = find_endpoints(left)
            right_endpoints = find_endpoints(right)
    """
    
    skeleton_points_all = []
    angle_weight=8
    max_pairs=2
    # Extract skeleton points from masks

    save_path  = os.path.join(dir,"segement_angles.jpg")

    height, width, _ = image.shape
    combined_mask = np.zeros((height, width), dtype=bool)



    overlay = image.copy()
    green = (0, 255, 0)
    for mask in masks:
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        skeleton = pcv.morphology.skeletonize(mask=mask)
        pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skeleton, size=70)
        labeled_img = pcv.morphology.segment_tangent_angle(segmented_img=segmented_img, objects=segment_objects,size =15)
        gray = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
        overlay[gray > 0] = green
    cv2.imwrite(os.path.join(dir, "segment_angles.jpg"), overlay)





    save_path  = os.path.join(dir,"endpoints.jpg")


    skeleton_points_all = []
    # print(len(masks))
    for mask in masks:
        skeleton = skeletonize_mask(mask[0].astype(np.uint8))
        skeleton_points = np.argwhere(skeleton > 0)  # Get non-zero points in the skeleton
        skeleton_points_all.append(skeleton_points)
    
    # Extract endpoints for each skeleton
    if len(image.shape) == 2:  
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image.copy()



    endpoints_list = []
    z = 0
    for skeleton in skeleton_points_all:
        endpoints = find_endpoints(skeleton)
        z+=1
        # print(endpoints,z)
        if endpoints.size > 0:
            endpoints_list.append(endpoints)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()

    # Plot the skeleton endpoints on the image
    idx = 0
    for endpoints in endpoints_list:
        for (y, x) in endpoints:  # Swap x and y for correct plotting
            cv2.circle(image_rgb, (x, y), radius=3, color=(255, 0, 0), thickness=-1)  # Red dot
            cv2.putText(image_rgb, str(idx), (x + 5, y - 5), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
                        color=(0, 255, 0), thickness=1)  # Green text
        idx+=1

    # Save the image with endpoints overlay
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image_rgb)
    # print(f"Overlay image with endpoints saved at {save_path}")



    num_skeletons = len(endpoints_list)
    if num_skeletons < 2:
        pairs = [(0,0,None,None)]
        return pairs
        # print(num_skeletons)
        
        raise ValueError("Not enough skeletons with endpoints to form pairs.")
    # Initialize distance and angle matrices
    distance_matrix = np.full((num_skeletons, num_skeletons), np.inf)
    angle_matrix = np.full((num_skeletons, num_skeletons), np.inf)
    angle_threshold = 2  # Convert 270 degrees to radians
    angle_bend_matrix = np.full((num_skeletons, num_skeletons), np.inf)

    # Compute distance and angle matrices
    endpoint_pair_indices = {}  # To store endpoint indices for each skeleton pair
    for i in range(num_skeletons):
        for j in range(i + 1, num_skeletons):
            dists = cdist(endpoints_list[i], endpoints_list[j])
            dist = np.min(dists)
            angle_i = calculate_angle(*endpoints_list[i][:2])  # First two endpoints
            angle_j = calculate_angle(*endpoints_list[j][:2])
            angle_diff = abs(angle_i - angle_j)
            
            le = find_endpoints(skeleton_points_all[i])
            re = find_endpoints(skeleton_points_all[j])
            r_center = find_center(skeleton_points_all[i])
            l_center = find_center(skeleton_points_all[j])
            distances = cdist(le, re)
            min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
            max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
            left_closest = le[min_distance_idx[0]]
            right_closest = re[min_distance_idx[1]]
                
            left_furthest = le[max_distance_idx[0]]
            right_furthest = re[max_distance_idx[1]]


            angle_bend_matrix[i,j] = calculate_bend_angle(left_center=l_center,left_closest=left_closest,right_closest=right_closest)


            distance_matrix[i, j] = dist
            angle_matrix[i, j] = angle_diff
            # Store the closest endpoints for this pair
            closest_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            point_i = endpoints_list[i][closest_pair_idx[0]]
            point_j = endpoints_list[j][closest_pair_idx[1]]

            # Validate the transition angle
            transition_vector = np.array(point_j) - np.array(point_i)
            transition_angle = np.arctan2(transition_vector[1], transition_vector[0])

            deviation_i = abs(transition_angle - angle_i)
            deviation_j = abs(transition_angle - angle_j)

            # Normalize deviations
            deviation_i = min(deviation_i, 2 * np.pi - deviation_i)
            deviation_j = min(deviation_j, 2 * np.pi - deviation_j)
       
            # Only update the matrices if the transition is valid
            
            distance_matrix[i, j] = dist
            angle_matrix[i, j] = abs(transition_angle)

                # Store the closest endpoints for valid transitions
            endpoint_pair_indices[(i, j)] = closest_pair_idx
            

    # Combine metrics
    combined_metric = distance_matrix + angle_weight * angle_matrix

    # Priority queue for pairing
    pair_counts = [0] * num_skeletons
    endpoint_used = [np.zeros(len(endpoints)) for endpoints in endpoints_list]  # Track endpoint usage
    pairs = []
    pq = []
    for i in range(num_skeletons):
        for j in range(i + 1, num_skeletons):
            heapq.heappush(pq, (combined_metric[i, j], i, j))
    
    # Pair skeletons based on priority queue
    while pq:
        metric, i, j = heapq.heappop(pq)

        if pair_counts[i] < max_pairs and pair_counts[j] < max_pairs:
            if (i, j) in endpoint_pair_indices:
                endpoint_i_idx, endpoint_j_idx = endpoint_pair_indices[(i, j)]
            else:
                # print(f"Invalid pair: {(i, j)} was not added due to validation failure.")
                continue  # Skip this pair

            if endpoint_used[i][endpoint_i_idx] or endpoint_used[j][endpoint_j_idx]:
                continue
            

            if angle_bend_matrix[i,j] >= 40 :
                endpoint_used[i][endpoint_i_idx] = 1
                endpoint_used[j][endpoint_j_idx] = 1
                continue

            endpoint_used[i][endpoint_i_idx] = 1
            endpoint_used[j][endpoint_j_idx] = 1

            # Pair the skeletons
            # print(f"pair :{i},{j} , angle : {angle_bend_matrix[i,j] } ")
            pairs.append((i, j, endpoint_i_idx, endpoint_j_idx))
            pair_counts[i] += 1
            pair_counts[j] += 1


    if pairs is None or len(pairs) == 0:
        for idx , mask in enumerate(masks) :
            pairs.append((idx,idx,None , None))
    return pairs
    
from collections import defaultdict

def link_pairs(pairs):
    graph = defaultdict(list)
    edges_count = 0
    
    for a, b, c, d in pairs:
        graph[a].append(b)
        graph[b].append(a)
        edges_count += 1

    visited_edges = set()
    linked_nodes = set()
    linked_pairs_list = []  # Store all linked paths (nested list)

    start_node = None
    for node, neighbors in graph.items():
        if len(neighbors) == 1:  # Degree 1 node
            start_node = node
            break
    if start_node is None:
        # If no degree 1 node, pick any node
        start_node = next(iter(graph))

    def dfs(node, current_path):
        for neighbor in graph[node]:
            if (node, neighbor) not in visited_edges and (neighbor, node) not in visited_edges:
                visited_edges.add((node, neighbor))
                visited_edges.add((neighbor, node))
                current_path.append((node, neighbor))
                linked_nodes.add(node)
                linked_nodes.add(neighbor)
                dfs(neighbor, current_path)

    current_path = []
    dfs(start_node, current_path)
    if current_path:  # Add the first path (starting from start_node)
        linked_pairs_list.append(current_path)

    for node in graph:
        if node not in linked_nodes:
            current_path = []
            dfs(node, current_path)
            if current_path:  # Add the path for the next component
                linked_pairs_list.append(current_path)

    # Debug: Check remaining edges

    return linked_pairs_list


def generate_curve(control_points, num_points=1000):
    control_points = np.array(control_points)
    x = control_points[:, 0]
    y = control_points[:, 1]

    # Fit a B-spline to the control points
    tck, u = splprep([x, y], s=0)  # s=0 ensures the curve passes through all points

    # Generate a dense set of points along the curve
    unew = np.linspace(0, 1, num_points)  # Parameter values for interpolation
    x_new, y_new = splev(unew, tck)



    return np.vstack((x_new, y_new)).T



def trim_far_points(y, x, trim_fraction=0.1):
    """
    Trims a fraction of the skeleton points that are farthest from the center.
    """
    points = np.column_stack((y, x))
    center = points.mean(axis=0)  # [center_y, center_x]
    distances = np.linalg.norm(points - center, axis=1)

    num_points_to_trim = int(len(points) * trim_fraction)
    keep_indices = np.argsort(distances)[:-num_points_to_trim]  # Indices of points to keep

    trimmed_points = points[keep_indices]
    trimmed_y, trimmed_x = trimmed_points[:, 0], trimmed_points[:, 1]

    return trimmed_y, trimmed_x


# https://fil-finder.readthedocs.io/en/latest/tutorial.html#skeletonization
def skeleton_remove_branch(skeleton):
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=400 * u.pix, skel_thresh=10 * u.pix, prune_criteria='intensity')
    skeleton_points = []
    longest_path_skeleton = fil.skeleton_longpath
    for idx, filament in enumerate(fil.filaments): 

        data = filament.branch_properties.copy()
        data_df = pd.DataFrame(data)
        data_df['offset_pixels'] = data_df['pixels'].apply(lambda x: x+filament.pixel_extents[0])


        longest_branch_idx = data_df.length.idxmax()
        longest_branch_pix = data_df.offset_pixels.iloc[longest_branch_idx]

        y,x = longest_branch_pix[:,0],longest_branch_pix[:,1]
        y,x = trim_far_points(y, x, trim_fraction=0.1)

        skeleton_points.extend(np.column_stack((y, x)))
        
    skeleton_points = np.array(skeleton_points, dtype=int)
    return skeleton_points



def find_center(points):
    """
    Given an array of 2D points (shape: (n_points, 2)), 
    compute the centroid of those points.
    """
    if points is None or len(points) == 0:
        return None
    return np.mean(points, axis=0)

def skeletonize_merge(image, masks,dir):
    skeleton_points_all = []
    
    for i, mask in enumerate(masks):
        skeleton = skeletonize_mask(mask[0].astype(np.uint8))
        skeleton_points = skeleton_remove_branch(skeleton)
        skeleton_points_all.append(skeleton_points)

    pairs = find_pairs(image, masks,dir)
    


    linked_pairs = link_pairs(pairs)



    unlinked_pairs = []
    linked_nodes = set()

    # Collect all linked nodes
    for path in linked_pairs:  # Each path is a list of (a, b) tuples
        for a, b in path:
            linked_nodes.add(a)
            linked_nodes.add(b)

    # Find masks that are not in linked_nodes
    for i in range(len(masks)):
        if i not in linked_nodes:
            unlinked_pairs.append(i)

    curves = []
    ultimate_all_control_points = []

    for idx in unlinked_pairs:
        all_control_points = []
        skele = skeleton_points_all[idx]
        endpoints = find_endpoints(skele)
        distances = cdist(endpoints, endpoints)
        max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
        center = find_center(skele)
        mid_point= (endpoints[max_distance_idx[0]] + center)/2
        all= [endpoints[max_distance_idx[0]],mid_point,center,endpoints[max_distance_idx[1]]]
        all_control_points.extend(all)
        unique_control_points = []
        seen = set()
        for point in all_control_points:
            tuple_point = tuple(point)
            if tuple_point not in seen:
                seen.add(tuple_point)
                unique_control_points.append(point)

        all_control_points = np.array(all_control_points)

        curve = generate_curve(all_control_points)
        curves.append(curve)

    for group in linked_pairs:
        all_control_points = []
        n = len(group)
        for i, pair in enumerate(group[::-1]):
            right = skeleton_points_all[pair[0]]
            left = skeleton_points_all[pair[1]]
            left_endpoints = find_endpoints(left)
            right_endpoints = find_endpoints(right)
            r_center = find_center(right)
            l_center = find_center(left)
            if left_endpoints.size > 0 and right_endpoints.size > 0:
                distances = cdist(left_endpoints, right_endpoints)
                min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
                max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
                left_closest = left_endpoints[min_distance_idx[0]]
                right_closest = right_endpoints[min_distance_idx[1]]
                
                left_furthest = left_endpoints[max_distance_idx[0]]
                right_furthest = right_endpoints[max_distance_idx[1]]

                
                

                points_to_add = [
                    left_furthest,
                    l_center,
                    left_closest,
                    right_closest,
                    r_center,
                    right_furthest,
                ]
                all_control_points.extend(points_to_add)
                ultimate_all_control_points.append(points_to_add)
        all_control_points = np.array(all_control_points)
        
        seen = set()
        unique_control_points = []
        for point in all_control_points:
            tuple_point = tuple(point)
            if tuple_point not in seen:
                seen.add(tuple_point)
                unique_control_points.append(point)
        all_control_points = np.array(unique_control_points)
        if len(all_control_points) < 4:
            if len(all_control_points) == 1:
                p0 = all_control_points[0]
                v = np.array([1, 0], dtype=float)
                p1 = p0 + v
                p2 = p1 + v
                p3 = p2 + v
                all_control_points = np.array([p0, p1, p2, p3])
            elif len(all_control_points) == 2:
                p0, p1 = all_control_points
                v = p1 - p0
                p2 = p1 + v
                p3 = p0 - v
                all_control_points = np.array([p0, p1, p2, p3])
            elif len(all_control_points) == 3:
                p0, p1, p2 = all_control_points
                v = p2 - p1
                p3 = p2 + v  # Extend forward
                p_1 = p0 - v  # Extend backward
                all_control_points = np.array([p_1, p0, p1, p2, p3])  # Include the new backward point

        curve = generate_curve(all_control_points)
        curves.append(curve)
    output_filename = os.path.join(dir,"control_points.png")
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')

    for idx, points in enumerate(ultimate_all_control_points):
        points = np.array(points)
        plt.scatter(points[:, 1], points[:, 0], color='red', s=20, label=f'Control Points Group {idx+1}' if idx == 0 else None)
    plt.title("Skeletonized Curves with Control Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(output_filename)
    plt.close()



    output_filename = os.path.join(dir,"curves2.png")

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    for i, curve in enumerate(curves):
        if curve is not None and curve.shape[0] > 0:
            plt.plot(curve[:, 1], curve[:, 0], marker='o', label=f"Curve {i+1}")
 
    plt.title("Skeletonized Curves with Control Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(output_filename)
    plt.close()

    return curves, ultimate_all_control_points
            

def skeletonize_merge2(image, masks,dir):
    skeleton_points_all = []
    
    for i, mask in enumerate(masks):
        skeleton = skeletonize_mask(mask[0].astype(np.uint8))
        skeleton_points = skeleton_remove_branch(skeleton)
        skeleton_points_all.append(skeleton_points)

    # Plot the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Overlay skeleton points
    for skeleton_points in skeleton_points_all:
        if len(skeleton_points) > 0:
            skeleton_points = np.array(skeleton_points)
            plt.scatter(skeleton_points[:, 1], skeleton_points[:, 0], color='red', s=5)  # Small red dots

    plt.axis("off")

    # Ensure output directory exists
    os.makedirs(dir, exist_ok=True)

    # Save the result
    output_path = os.path.join(dir, "skeletonized_image.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # print(f"Saved skeletonized image to {output_path}")

    
    linked_pairs = []

    print(linked_pairs)

    unlinked_pairs = []
    linked_nodes = set()


    for i in range(len(masks)):
        if i not in linked_nodes:
            unlinked_pairs.append(i)

    curves = []
    ultimate_all_control_points = []

    for idx in unlinked_pairs:
        all_control_points = []
        skele = skeleton_points_all[idx]
        endpoints = find_endpoints(skele)
        distances = cdist(endpoints, endpoints)
        max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
        center = find_center(skele)
        mid_point= (endpoints[max_distance_idx[0]] + center)/2
        all= [endpoints[max_distance_idx[0]],mid_point,center,endpoints[max_distance_idx[1]]]
        all_control_points.extend(all)
        unique_control_points = []
        seen = set()
        for point in all_control_points:
            tuple_point = tuple(point)
            if tuple_point not in seen:
                seen.add(tuple_point)
                unique_control_points.append(point)

        all_control_points = np.array(all_control_points)
        print(all_control_points)
        curve = generate_curve(all_control_points)
        curves.append(curve)

    for group in linked_pairs:
        all_control_points = []
        n = len(group)
        for i, pair in enumerate(group[::-1]):
            print(pair)
            right = skeleton_points_all[pair[0]]
            left = skeleton_points_all[pair[1]]
            left_endpoints = find_endpoints(left)
            right_endpoints = find_endpoints(right)
            r_center = find_center(right)
            l_center = find_center(left)
            print(left_endpoints,right_endpoints)
            if left_endpoints.size > 0 and right_endpoints.size > 0:
                distances = cdist(left_endpoints, right_endpoints)
                min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
                max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
                left_closest = left_endpoints[min_distance_idx[0]]
                right_closest = right_endpoints[min_distance_idx[1]]
                
                left_furthest = left_endpoints[max_distance_idx[0]]
                right_furthest = right_endpoints[max_distance_idx[1]]

                
                

                points_to_add = [
                    left_furthest,
                    l_center,
                    left_closest,
                    right_closest,
                    r_center,
                    right_furthest,
                ]
                print(points_to_add)
                all_control_points.extend(points_to_add)
                ultimate_all_control_points.append(points_to_add)
        all_control_points = np.array(all_control_points)
        
        seen = set()
        unique_control_points = []
        for point in all_control_points:
            tuple_point = tuple(point)
            if tuple_point not in seen:
                seen.add(tuple_point)
                unique_control_points.append(point)
        all_control_points = np.array(unique_control_points)
        if len(all_control_points) < 4:
            if len(all_control_points) == 1:
                p0 = all_control_points[0]
                v = np.array([1, 0], dtype=float)
                p1 = p0 + v
                p2 = p1 + v
                p3 = p2 + v
                all_control_points = np.array([p0, p1, p2, p3])
            elif len(all_control_points) == 2:
                p0, p1 = all_control_points
                v = p1 - p0
                p2 = p1 + v
                p3 = p0 - v
                all_control_points = np.array([p0, p1, p2, p3])
            elif len(all_control_points) == 3:
                p0, p1, p2 = all_control_points
                v = p2 - p1
                p3 = p2 + v
                all_control_points = np.array([p0, p1, p2, p3])
        print(all_control_points)
        curve = generate_curve(all_control_points)
        curves.append(curve)
    print("curves found!")
    output_filename = os.path.join(dir,"control_points.png")
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')

    for idx, points in enumerate(ultimate_all_control_points):
        points = np.array(points)
        plt.scatter(points[:, 1], points[:, 0], color='red', s=20, label=f'Control Points Group {idx+1}' if idx == 0 else None)
    plt.title("Skeletonized Curves with Control Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(output_filename)
    plt.close()



    output_filename = os.path.join(dir,"curves2.png")

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    for i, curve in enumerate(curves):
        if curve is not None and curve.shape[0] > 0:
            plt.plot(curve[:, 1], curve[:, 0], marker='o', label=f"Curve {i+1}")
 
    plt.title("Skeletonized Curves with Control Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(output_filename)
    plt.close()
    print("curves : " ,output_filename)
    print("Plot finished!")
    return curves, ultimate_all_control_points
            

    
def downsample_points(points, step=5):
    """
    Downsample points to reduce their number while keeping the overall shape.
    """
    return points[::step]  # Select every `step`th point



def show_masks(masks, filename, image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    for i, mask in enumerate(masks):
        mask = mask[0].astype(np.uint8)
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

import cv2
import numpy as np
import cv2
import numpy as np

def get_rotated_rectangle_info(mask):

    mask = mask.astype(np.uint8)  # Ensure binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None  

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the minimum area bounding rectangle
    rotated_rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rotated_rect)  # Get rectangle corners
    box = np.int0(box)  # Convert to integers for pixel coordinates

    # Extract rectangle properties
    center, size, angle = rotated_rect
    width, height = size

    # Compute side lengths and angles
    side_info = []
    for i in range(4):
        pt1, pt2 = box[i], box[(i + 1) % 4]  # Get consecutive points
        length = np.linalg.norm(pt2 - pt1)  # Compute Euclidean distance
        dx, dy = pt2 - pt1
        side_angle = np.degrees(np.arctan2(dy, dx))  # Compute angle in degrees
        side_info.append((length, side_angle))

    # Identify the longest side and its angle
    longest_side, longest_angle = max(side_info, key=lambda x: x[0])

    # Ensure angle is positive (convert negative angles)
    if longest_angle < 0:
        longest_angle += 180  # Convert to positive equivalent



    return {
        "center": center,
        "width": width,
        "height": height,
        "angle": angle,  # The positive angle along the longest side
        "sides": side_info,  # List of (length, angle) tuples for each side
        "longest_angle":longest_angle,
    }




def split_masks(image,masks):
    height, width, _ = image.shape

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the image
    ax.imshow(image, interpolation='nearest')

    for mask in masks:
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        center_x, center_y = width / 2, height / 2
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            contour_points = contour.squeeze(1)  # Remove unnecessary dimension
            
            if len(contour_points.shape) < 2:
                continue  # Skip contours with insufficient points

            # Calculate the x-coordinate center
            x_center = np.mean(contour_points[:, 0])

            # Separate points into left and right based on x_center
            left_points = contour_points[contour_points[:, 0] < x_center]
            right_points = contour_points[contour_points[:, 0] >= x_center]

            # Plot the left points in blue
            ax.scatter(left_points[:, 0], left_points[:, 1], color='blue', s=10, label="Left Points")

            # Plot the right points in red
            ax.scatter(right_points[:, 0], right_points[:, 1], color='red', s=10, label="Right Points")


            if len(left_points) > 0:
                distances_left = np.sqrt((left_points[:, 0] - center_x) ** 2 + (left_points[:, 1] - center_y) ** 2)
                top_left = left_points[np.argmax(distances_left)]  # Closest point in the left group
                ax.scatter(top_left[0], top_left[1], color='green', s=50, marker='o', label='Top Left')

            if len(right_points) > 0:
                distances_right = np.sqrt((right_points[:, 0] - center_x) ** 2 + (right_points[:, 1] - center_y) ** 2)
                top_right = right_points[np.argmax(distances_right)]  # Closest point in the right group
                ax.scatter(top_right[0], top_right[1], color='yellow', s=50, marker='o', label='Top Right')


    plt.savefig("experiment_images/spliting_masks/john_dickens_lr.png")


#Works fine but is not perfect. Has flaws witht he spiral image.
def calculate_local_angle(mask, point, window_size=20):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0  # Default angle if no contours are found

    contour_points = np.vstack(contours).squeeze()

    if len(contour_points.shape) < 2:
        return 0  # Not enough points to calculate an angle

    x_center  = np.mean(contour_points[:, 0])

    left_points = contour_points[contour_points[:, 0] < x_center]
    right_points = contour_points[contour_points[:, 0] >= x_center]

    if len(left_points) == 0 or len(right_points) == 0:
        return 0  # Default angle if we cannot divide the mask

    top_left = left_points[np.argmin(left_points[:, 1])]
    top_right = right_points[np.argmin(right_points[:, 1])]

    x_diff = abs(top_right[0] - top_left[0])
    y_diff = abs(top_right[1] - top_left[1])

    if x_diff < 20: 
        if top_right[1] > top_left[1]:  
            print("90")
            return 90
        else:  # Bottom-to-top orientation
            return 90

    # For non-vertical cases, calculate the angle between the two topmost points
    vector = top_right - top_left
    angle = degrees(atan2(vector[1], vector[0]))  # Convert to degrees
    if angle <10:
        # print(x_diff,y_diff)
        pass
    angle = (angle + 360) % 360
    return angle


def trim_far_points(y, x, trim_fraction=0.1):
    """
    Trims a fraction of the skeleton points that are farthest from the center.
    """
    points = np.column_stack((y, x))
    center = points.mean(axis=0)  # [center_y, center_x]
    distances = np.linalg.norm(points - center, axis=1)

    num_points_to_trim = int(len(points) * trim_fraction)
    keep_indices = np.argsort(distances)[:-num_points_to_trim]  # Indices of points to keep

    trimmed_points = points[keep_indices]
    trimmed_y, trimmed_x = trimmed_points[:, 0], trimmed_points[:, 1]

    return trimmed_y, trimmed_x


# https://fil-finder.readthedocs.io/en/latest/tutorial.html#skeletonization
def skeleton_remove_branch(skeleton):
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=400 * u.pix, skel_thresh=10 * u.pix, prune_criteria='intensity')
    skeleton_points = []
    longest_path_skeleton = fil.skeleton_longpath
    for idx, filament in enumerate(fil.filaments): 

        data = filament.branch_properties.copy()
        data_df = pd.DataFrame(data)
        data_df['offset_pixels'] = data_df['pixels'].apply(lambda x: x+filament.pixel_extents[0])


        longest_branch_idx = data_df.length.idxmax()
        longest_branch_pix = data_df.offset_pixels.iloc[longest_branch_idx]

        y,x = longest_branch_pix[:,0],longest_branch_pix[:,1]
        y,x = trim_far_points(y, x, trim_fraction=0.1)

        skeleton_points.extend(np.column_stack((y, x)))
        
    skeleton_points = np.array(skeleton_points, dtype=int)
    return skeleton_points






def smooth_angles(angles):
    smoothed_angles = [angles[0]]  # Start with the first angle
    for i in range(1, len(angles)):
        diff = angles[i] - smoothed_angles[-1]
        # Adjust to take the shortest path
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        # Append the adjusted angle
        smoothed_angles.append(smoothed_angles[-1] + diff)
    return smoothed_angles


def calculate_local_angle_curve(mask, center, window_size=10):
    """
    Calculate the angle between the current point and the next point along the curve.

    Parameters:
        mask (np.ndarray): Binary mask of the region.
        center (tuple): The (y, x) coordinates of the current point.
        window_size (int): The window size for calculating the angle. Default is 10.

    Returns:
        float: The calculated angle in degrees.
    """
    height, width = mask.shape

    # Create a window around the center point
    y, x = center
    y_start = max(0, y - window_size)
    y_end = min(height, y + window_size)
    x_start = max(0, x - window_size)
    x_end = min(width, x + window_size)

    # Extract the points within the window
    window_points = np.argwhere(mask[y_start:y_end, x_start:x_end])
    if len(window_points) < 2:
        return 0  

    window_points[:, 0] += y_start
    window_points[:, 1] += x_start

    distances = np.linalg.norm(window_points - np.array(center), axis=1)
    sorted_indices = np.argsort(distances)

    current_point = window_points[sorted_indices[0]]
    next_point = window_points[sorted_indices[1]]

    delta_y = next_point[0] - current_point[0]
    delta_x = next_point[1] - current_point[1]
    angle = degrees(atan2(delta_y, delta_x))

    return angle





def smooth_and_sort_skeleton(skeleton_points):
    """Sort skeleton points along the main path and smooth them."""
    if len(skeleton_points) < 2:
        return skeleton_points  # Not enough points to sort

    # Sort points based on the minimum spanning path (approximation)
    sorted_points = skeleton_points[np.argsort(skeleton_points[:, 1])]  # Sorting by x-coordinate

    # Smooth using Gaussian filter to remove small fluctuations
    x_smooth = gaussian_filter1d(sorted_points[:, 1], sigma=2)
    y_smooth = gaussian_filter1d(sorted_points[:, 0], sigma=2)

    return np.column_stack((y_smooth, x_smooth))  # Return as (y, x) format

def compute_skeleton_length(sorted_points):
    """Compute the cumulative length of the skeleton."""
    total_length = 0
    for i in range(1, len(sorted_points)):
        total_length += distance.euclidean(sorted_points[i-1], sorted_points[i])
    return total_length

def find_cut_positions(sorted_points, total_length,num_cuts):

    segment_length = total_length / num_cuts+1
    current_length = 0
    cut_positions = []
    
    for i in range(1, len(sorted_points)):
        step_length = distance.euclidean(sorted_points[i-1], sorted_points[i])
        current_length += step_length
        if len(cut_positions) < num_cuts and current_length >= (len(cut_positions) + 1) * segment_length:
            cut_positions.append(sorted_points[i])  # Store the point for cutting
    
    return np.array(cut_positions)  # Return the 3 cutting points



def segment_mask(mask, sorted_points, cut_positions):
    """
    
    Assigns each pixel in the mask to the closest partition and colors it accordingly.
    
    """
    height, width = mask.shape
    segmented_mask = np.zeros((height, width), dtype=np.uint8)

    # Create a KDTree for fast nearest-neighbor search
    cut_tree = KDTree(cut_positions)

    # Assign each skeleton point to the nearest cut position
    labels = cut_tree.query(sorted_points)[1]  # Find closest cut point index

    unique_labels = np.unique(labels)
    mask_points = np.column_stack(np.where(mask > 0))  # Get all mask points
    mask_labels = cut_tree.query(mask_points)[1]  # Find closest cut point index for each mask pixel

    for label, point in zip(mask_labels, mask_points):
        segmented_mask[int(point[0]), int(point[1])] = label + 1  # Avoid zero for better coloring

    return segmented_mask


def remove_close_points(cut_positions):
    """"

    Removes cut points that are too close together. If the cutpoints are so close as to not make a difference in the cutting, we just remove thoose.

    """

    if len(cut_positions) < 2:
        return cut_positions  # Return as is if less than 2 points

    pairwise_distances = [distance.euclidean(cut_positions[i], cut_positions[i + 1]) 
                          for i in range(len(cut_positions) - 1)]
    
    avg_distance = np.mean(pairwise_distances)  # Compute the average spacing

    filtered_positions = [cut_positions[0]]  # Keep the first cut position

    for i in range(1, len(cut_positions)):
        if distance.euclidean(cut_positions[i], filtered_positions[-1]) >= avg_distance * 0.7:
            filtered_positions.append(cut_positions[i])

    return np.array(filtered_positions)


from skimage.morphology import thin, skeletonize

def divide_masks(image, masks , dir):

    """"

    This function attempts to divide the masks into various sub segments. 

    Step 1 : We find the skeletons of all the masks, we also identify masks which are smaller than the rest so we dont divide these too much (<25th percentile)

    Step 2 : We identify sparse points (AKA cutpoints), these points will be where we cut the masks to segments. using find_cut_positions()

    Step 3 : We segment the mask based on the cut point, so points are distributed based on which cut point they are closed to. using segment_mask()
    """


    save_dir = dir
    os.makedirs(save_dir, exist_ok=True)

    all_skeleton_lengths = []
    all_cut_positions = []

    # Step 1: Compute skeleton lengths for all masks
    for mask in masks:
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask.astype(np.uint8)

        skeleton = skeletonize_mask(mask)
        
        skeleton_points = thin(skeleton)
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        
        if len(skeleton_points) < 4:
            all_skeleton_lengths.append(0)
            continue

        sorted_points = smooth_and_sort_skeleton(skeleton_points)

        total_length = compute_skeleton_length(sorted_points)
        all_skeleton_lengths.append(total_length)

    skeleton_lengths_array = np.array(all_skeleton_lengths)
    threshold_25th = np.percentile(skeleton_lengths_array[skeleton_lengths_array > 0], 25)
    
    all_segmented_masks = []
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")  # Show the original image as the background
    for idx, mask in enumerate(masks):
        # Ensure mask is correctly formatted
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask.astype(np.uint8)

 
        skeleton = skeletonize_mask(mask)
        skeleton_points = np.column_stack(np.where(skeleton > 0))


        if len(skeleton_points) < 4:
            print(f"Mask {idx}: Skeleton too small for segmentation.")
            all_cut_positions.append(None)
            continue

        # Step 2 
        sorted_points = smooth_and_sort_skeleton(skeleton_points)
        total_length = compute_skeleton_length(sorted_points)
        num_cuts = 2 if total_length < threshold_25th else 5

        # Find cut positions
        cut_positions = find_cut_positions(sorted_points, total_length, num_cuts)
        cut_positions = remove_close_points(cut_positions)

        # Step 3
        segmented_mask = segment_mask(mask, sorted_points, cut_positions)
        masked_segmented = np.where(mask > 0, segmented_mask, np.nan) 
        plt.imshow(masked_segmented, cmap="tab10", alpha=0.6)  


        for (y, x) in cut_positions:
            plt.scatter(x, y, s=100, color='green', edgecolors='black', linewidth=1.5, label="Cut Position")

        num_segments = np.max(segmented_mask)  # Get the number of segments
        segment_masks = [(segmented_mask == i).astype(bool) for i in range(1, num_segments + 1)]  # Boolean masks

        all_segmented_masks.extend(segment_masks)
        
        all_cut_positions.append(cut_positions)
    vis_save_path = f"{save_dir}/skeleton_with_cut_positions_all.png"
    plt.axis('off')
    plt.savefig(vis_save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # print(f"Mask {idx}: Skeletonized image with cut positions saved at: {vis_save_path}")
    return all_segmented_masks

def angle_offset(angle,rect_info_original):
    """"

    https://theailearner.com/tag/cv2-minarearect/ 

    cv2.minAreaRect() works in a peculiar way where the angle is based on the points closes to the x axis or something similar , so the orientation of the rectangle is still jumbled when we get the rectangle.

    Here we compare the original masks' angle and the angle with the segmented mask. If the offset it too high , we correct it. This works well as the segmented mask gives us more 
    accuracy while the orignal mask gives us better orientation.

    """
    width = rect_info_original['width']
    height = rect_info_original['height']

    if width < height:
        angle+=90
    if min(190- (rect_info_original['longest_angle']-angle) % 180, (rect_info_original['longest_angle']-angle) % 180) > 45:
        angle+=90
    
    return angle

import cv2
import numpy as np



def text_animation(curves, image, masks ,letter_mask, ultimate_all_control_points, output_file="experiment_images/text_sweep/animation.mp4",square_width = 10 , brightness = 1 , speed = 1 , mode = "animate" , transform = False , matrix = None,whole_image = None):


    """"

    Step by step explanation for this function. 

    Step 1 : We divide our mask into various segments , this is done using divide_masks() , here we segment them using skeletonization. After skeletonisation we split them to points and then segment them accordingly.

    Step 2 : We enumerate and reshape all the masks. Using these masks , we can find the minimum area rectangle around it using get_rotated_rectangle_info(). https://theailearner.com/tag/cv2-minarearect/

    Step 3 : Now we iterate through each curve point for each frame. For each point in our curve , we make a temporary mask around the curved point. We use this mask to identify which mask contains our curve point. 

    Step 4 : Using the best mask , we are able to find the appropriate rotated rectangle info , this gives us the angle of the rectangle. This angle is then used to rotate the mask. The maximum length of the rectangle is also used to identify the size of our mask window.

    Step 5 : Once the angle and the length of window is known , we just create an overlay and animate this text sweep.

    """


    


    height, width, _ = image.shape

    fps = 30
    duration = 25  # seconds
    total_frames = fps * duration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    dir = os.path.dirname(output_file)
    # Step 1
    original_masks = masks.copy()
    masks = divide_masks(image,masks,dir)
    persistent_mask = np.zeros((height, width), dtype=np.uint8)
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Step 2
    for idx, mask in enumerate(masks):
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        combined_mask |= mask

    for idx, mask in enumerate(original_masks):
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        combined_mask |= mask

    original_masks_rectangles = [get_rotated_rectangle_info(mask.squeeze(0).astype(np.uint8)) for mask in original_masks]
    mask_rectangles = [get_rotated_rectangle_info(mask) for mask in masks]


    curve_points = [point for curve in curves for point in curve]  
    total_curve_points = len(curve_points)
    # Ensure the first point is the leftmost (smallest x-coordinate)
    print(curve_points[0][0] ,curve_points[-1][0])
    if curve_points[0][1] > curve_points[-1][1]:  
        curve_points = curve_points[::-1]

    prev_angle = 0
    angles=[]
    speed = max(speed , 1)
    for frame_idx in range(total_frames):
        progress = (frame_idx * speed) / (total_frames - 1)  # Ensure it scales properly
        progress = min(progress, 1.0)  # Clamp to 1.0
        curve_point_idx = int(progress * (total_curve_points - 1))  # Use total_curve_points - 1 to avoid overflow
        curve_point = tuple(map(int, curve_points[curve_point_idx]))
        swapped_curve_point = (curve_point[1], curve_point[0])  # Swap x and y for mask logic


        #Step 3 
        square_size = 10  
        x_start = max(0, swapped_curve_point[1] - square_size // 2)  
        x_end = min(height, swapped_curve_point[1] + square_size // 2)
        y_start = max(0, swapped_curve_point[0] - square_size // 2)  
        y_end = min(width, swapped_curve_point[0] + square_size // 2)

        # Create the center mask
        center_mask = np.zeros_like(combined_mask, dtype=bool)
        center_mask[x_start:x_end, y_start:y_end] = True
        

        best_mask_idx = None
        best_orginal_mask_idx = None
        max_overlap = 0
        max_orginal_mask_overlap = 0

        for i, mask in enumerate(masks):
            overlap = np.sum(mask & center_mask)
            if overlap > max_overlap:
                max_overlap = overlap
                best_mask_idx = i

        for i, mask in enumerate(original_masks):
            overlap = np.sum(mask & center_mask)
            if overlap > max_orginal_mask_overlap:
                max_orginal_mask_overlap = overlap
                best_orginal_mask_idx = i

       
        #Step 4
        if best_mask_idx is not None:
            rect_info = mask_rectangles[best_mask_idx]
            rect_info_original = original_masks_rectangles[best_orginal_mask_idx]
            if rect_info:
                square_size = min(int(rect_info_original["height"]), int(rect_info_original["width"])) 
                angle = rect_info["angle"]
                angle = angle_offset(angle,rect_info_original)
                best_mask = masks[best_mask_idx].astype(np.uint8)
                # angle = calculate_local_angle_curve(best_mask, (int(curve_point[1]), int(curve_point[0])), window_size=square_size)
            else:
                angle = 0  # Default angle if no rectangle info is available
                best_mask = combined_mask
        else:

            angle = 0  # Default angle if no mask matches
            best_mask = combined_mask

        
        # Step 5 
        if square_size < square_width :
            square_size , square_width = square_width , square_size
        seconds = frame_idx / fps

        angles.append(angle)
        rect_center = (int(curve_point[1]), int(curve_point[0]))  # (y, x) because OpenCV uses this order
        rect_size = (square_width, square_size)  # (width, height)
        rotation_matrix = cv2.getRotationMatrix2D(rect_center, angle, 1.0)


        temp_mask = np.zeros((height, width), dtype=np.uint8)
        box = cv2.boxPoints(((rect_center[0], rect_center[1]), rect_size, angle))  # Get rotated rectangle vertices
        box = np.int0(box)  # Convert to integer
        cv2.fillPoly(temp_mask, [box], 255)

        # final_mask = (temp_mask > 0) & (best_mask > 0)  # Use the best matching mask directly
        # mask_overlay = np.zeros_like(image)
        # mask_overlay[final_mask] = image[final_mask]

        # mask_overlay = cv2.convertScaleAbs(mask_overlay, alpha=brightness, beta=0)
        # video_writer.write(mask_overlay.astype(np.uint8))
        letter_mask = letter_mask.astype(np.uint8)
        letter_mask = letter_mask.squeeze() 
        final_mask = (temp_mask > 0) & (letter_mask > 0)
        final_output = np.zeros((height, width), dtype=np.uint8)
        intensity = np.clip(brightness * 25.5, 0, 255).astype(np.uint8)

        final_output[final_mask] = intensity
        persistent_mask[final_mask] = intensity
        
        if mode == "reveal":
            frame_bgr = cv2.cvtColor(persistent_mask, cv2.COLOR_GRAY2BGR)
        else :
            frame_bgr = cv2.cvtColor(final_output, cv2.COLOR_GRAY2BGR)
        

        # if transform:
        #     # Apply inverse perspective transformation
        #     transformed_frame = cv2.warpPerspective(frame_bgr, matrix, (width_transform,height_transform),flags = cv2.WARP_INVERSE_MAP)
        #     video_writer.write(transformed_frame)
        # else:
        # No transformation needed, write frame as is
        video_writer.write(frame_bgr)


        if progress >= 1.0:
            break

    video_writer.release()
    # print(f"video is saved at {output_file}")
    return output_file




import base64

def get_mp4_base64(file_path):
    """
    Reads an MP4 file and returns its Base64 encoded string.

    Args:
        file_path (str): Path to the MP4 file.

    Returns:
        str: Base64-encoded MP4 data.
    """
    try:
        with open(file_path, 'rb') as mp4_file:
            return base64.b64encode(mp4_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def check_mem(state):
    print(state)
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))




def text_reveal(img_path, dir ,speed=1,brightness=1,window_width=10, device = "cuda", ):

    start_time = time.time()  # Start time
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = get_default_args()



    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)

    amg = AutoMaskGenerator(hisam)

    if args.dataset == 'totaltext':
        if args.zero_shot:
            fg_points_num = 50  # assemble text kernel
            score_thresh = 0.3
            unclip_ratio = 1.5
        else:
            fg_points_num = 500
            score_thresh = 0.95
    elif args.dataset == 'ctw1500':
        if args.zero_shot:
            fg_points_num = 100
            score_thresh = 0.6
        else:
            fg_points_num = 300
            score_thresh = 0.7
    else:
        raise ValueError


    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Error: Image file '{img_path}' not found!")

  
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # h, w, 3
    img_h, img_w = image.shape[:2]

    amg.set_image(image)

    masks, scores = amg.predict_text_detection(
            from_low_res=False,
            fg_points_num=fg_points_num,
            batch_points_num=min(fg_points_num, 100),
            score_thresh=score_thresh,
            nms_thresh=score_thresh,
            zero_shot=args.zero_shot,
            dataset=args.dataset
    )

    del amg
    del hisam
    if len(masks) <= 0 :
        print("Not enough masks.")
        return
    curves,ultimate_all_control_points = skeletonize_merge(image,masks,dir)
    end_time = time.time()  # End time
    execution_time = end_time - start_time


    vid_path = os.path.join(dir,"video.mp4")
    letter_mask_output = letter_mask(img_path)

    text_animation(curves,image,masks,letter_mask_output,ultimate_all_control_points,output_file=vid_path,speed=speed,brightness=brightness,square_width=window_width,mode = "reveal" )
    return vid_path



def text_sweep_sentence(img_path, dir ,speed=1,brightness=1,window_width=10, device = "cuda", ):
    import os 
    os.chdir('/home/jazz/Harish_ws/Demo/Hi-sam')
    start_time = time.time()  # Start time
    seed = 42
    check_mem("Begin")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = get_default_args_word_model()

    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    print("Loaded model")

    amg = AutoMaskGenerator(hisam)

    if args.dataset == 'totaltext':
        if args.zero_shot:
            fg_points_num = 50  # assemble text kernel
            score_thresh = 0.3
            unclip_ratio = 1.5
        else:
            fg_points_num = 500
            score_thresh = 0.95
    elif args.dataset == 'ctw1500':
        if args.zero_shot:
            fg_points_num = 100
            score_thresh = 0.6
        else:
            fg_points_num = 300
            score_thresh = 0.7
    else:
        raise ValueError


    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Error: Image file '{img_path}' not found!")

  
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # h, w, 3
    img_h, img_w = image.shape[:2]

    amg.set_image(image)

    masks, scores = amg.predict_text_detection(
            from_low_res=False,
            fg_points_num=fg_points_num,
            batch_points_num=min(fg_points_num, 100),
            score_thresh=score_thresh,
            nms_thresh=score_thresh,
            zero_shot=args.zero_shot,
            dataset=args.dataset
    )

    del amg
    del hisam
    if len(masks) <= 0 :
        print("Not enough masks.")
        return

    if len(masks) == 0:
        print("Not enough masks.")
        return

    # Plot the image with detected masks
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Overlay masks
    for mask in masks:
        plt.contour(mask.squeeze(0), colors='r', linewidths=2)

    plt.axis("off")

    # Save the image
    output_path = os.path.join(dir, os.path.basename(img_path).replace(".jpg", "_masks.jpg"))
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved image with masks to {output_path}")

    plt.show()
    curves,ultimate_all_control_points = skeletonize_merge2(image,masks,dir)
    return 






def text_sweep(img_path, dir ,speed=1,brightness=1,window_width=10, device = "cuda", ):
    
    import os 
    
    
    start_time = time.time()  # Start time
    seed = 42
    # check_mem("Begin")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = get_default_args()



    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)

    amg = AutoMaskGenerator(hisam)

    if args.dataset == 'totaltext':
        if args.zero_shot:
            fg_points_num = 50  # assemble text kernel
            score_thresh = 0.3
            unclip_ratio = 1.5
        else:
            fg_points_num = 500
            score_thresh = 0.95
    elif args.dataset == 'ctw1500':
        if args.zero_shot:
            fg_points_num = 100
            score_thresh = 0.6
        else:
            fg_points_num = 300
            score_thresh = 0.7
    else:
        raise ValueError


    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Error: Image file '{img_path}' not found!")

  
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # h, w, 3
    img_h, img_w = image.shape[:2]

    amg.set_image(image)

    masks, scores = amg.predict_text_detection(
            from_low_res=False,
            fg_points_num=fg_points_num,
            batch_points_num=min(fg_points_num, 100),
            score_thresh=score_thresh,
            nms_thresh=score_thresh,
            zero_shot=args.zero_shot,
            dataset=args.dataset
    )

    del amg
    del hisam
    if len(masks) <= 0 :
        print("Not enough masks.")
        return
    curves,ultimate_all_control_points = skeletonize_merge(image,masks,dir)
    end_time = time.time()  # End time
    execution_time = end_time - start_time


    vid_path = os.path.join(dir,"video.mp4")
    letter_mask_output = letter_mask(img_path)
    
    
    text_animation(curves,image,masks,letter_mask_output,ultimate_all_control_points,output_file=vid_path,speed=speed,brightness=brightness,square_width=window_width)
    return vid_path



def letter_mask(img_path):
    args = get_args_parser_letter_model(file_path=img_path)
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    predictor = SamPredictor(hisam)
    output_mask = None
    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm(args.input, disable=not args.output):
        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            img_name = os.path.basename(path).split('.')[0] + '.png'
            out_filename = os.path.join(args.output, img_name)
        else:
            assert len(args.input) == 1
            out_filename = args.output

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if args.patch_mode:
            ori_size = image.shape[:2]
            patch_list, h_slice_list, w_slice_list = patchify_sliding(image, 512, 384)  # sliding window config
            mask_512 = []
            for patch in tqdm(patch_list):
               predictor.set_image(patch)
               m, hr_m, score, hr_score = predictor.predict(multimask_output=False, return_logits=True)
               assert hr_m.shape[0] == 1  # high-res mask
               mask_512.append(hr_m[0])
            mask_512 = unpatchify_sliding(mask_512, h_slice_list, w_slice_list, ori_size)
            assert mask_512.shape[-2:] == ori_size
            mask = mask_512
            mask = mask > predictor.model.mask_threshold
            save_binary_mask(mask, out_filename)
            output_mask = mask
        else:
            predictor.set_image(image)
            if args.hier_det:
                input_point = np.array([[125, 275]])  # for demo/img293.jpg
                input_label = np.ones(input_point.shape[0])
                mask, hr_mask, score, hr_score, hi_mask, hi_iou, word_mask = predictor.predict(
                    multimask_output=False,
                    hier_det=True,
                    point_coords=input_point,
                    point_labels=input_label,
                )
                show_hi_masks(hi_mask, word_mask, input_point, out_filename, image, hi_iou)
                output_mask = hi_mask
            else:
                mask, hr_mask, score, hr_score = predictor.predict(multimask_output=False)
                # save_binary_mask(hr_mask, out_filename)
                output_mask = hr_mask
    del predictor
    del hisam
    torch.cuda.empty_cache()
    return mask


