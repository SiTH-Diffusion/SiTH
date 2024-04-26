"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import os 
import torch
import numpy as np
import argparse
import PIL.Image as Image
from tqdm import tqdm
import cv2
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import torchvision.transforms as transforms

import mediapipe as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=2.0).to(device)
kid = KernelInceptionDistance(feature=2048, normalize=True).to(device)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.0)

body_pose = [11,12,13,14,15,16,23,24,25,26,27,28]
def get_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    else:
        return []

def compute_errors(landmarks1, landmarks2):
    errors = []
    for i, (l1, l2) in enumerate(zip(landmarks1, landmarks2)):
        if i not in body_pose:
            continue
        error = ((l1.x - l2.x)**2 + (l1.y - l2.y)**2)**0.5
        errors.append(error)
    mean_error = np.mean(errors)

    return mean_error


image_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
to_tensor = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)

def compute_lpips(gt_path, src_path, mask=False):

    lpips_list = []
    ssim_list = []
    for gt_img, gen_folder in tqdm(zip(gt_path, src_path)):
        pil_gt = Image.open(gt_img)
        gt_tensor = image_transforms(pil_gt).unsqueeze(0)[:, :3,...]
        mask_tensor = to_tensor(pil_gt)[3:,...]
    
        #mask_tensor = image_transforms(pil_gt).unsqueeze(0)[:, 3:,...]
        gt_tensor[mask_tensor.expand_as(gt_tensor) < 0.5] = 1.0

        img_list= [x for x in sorted(os.listdir(gen_folder)) if x.endswith('png')]
        total_l = 0.0
        total_s = 0.0
        for gen_img in img_list:
            pil_gen = Image.open(os.path.join(gen_folder, gen_img))
            gen_tensor = image_transforms(pil_gen).unsqueeze(0)[:, :3,...]

            if mask:
                gen_tensor[mask_tensor.expand_as(gen_tensor) < 0.5] = 1.0

            _lpips = lpips(gen_tensor.to(device), gt_tensor.to(device))
            _ssim = ms_ssim(gen_tensor.to(device), gt_tensor.to(device))
            total_l += _lpips.item()
            total_s += _ssim.item()

        lpips_list.append(total_l/len(img_list))
        ssim_list.append(total_s/len(img_list))


    mean_lpips = np.mean(lpips_list)
    print('mean lpips: ', mean_lpips)
    std = np.std(lpips_list)
    print('std: ', std)

    mean_ssim = np.mean(ssim_list)
    print('mean ssim: ', mean_ssim)
    std = np.std(ssim_list)
    print('std: ', std)


def compute_kid(gt_path, src_path, run=32, mask=False):

    for _ in tqdm(range(run)):
        gt_list = []
        img_list = []
        for idx in range(len(gt_path)):
            pil_gt = Image.open(gt_path[idx])
            gt_tensor = to_tensor(pil_gt).unsqueeze(0)[:, :3,...]
            #mask_tensor = image_transforms(pil_gt).unsqueeze(0)[:, 3:,...]
            mask_tensor = to_tensor(pil_gt)[3:,...]


            gt_tensor[mask_tensor.expand_as(gt_tensor) < 0.5] = 1.0
            gt_list.append(gt_tensor)
            
            gen_folder = src_path[idx]
            gen_list= [x for x in sorted(os.listdir(gen_folder)) if x.endswith('png')]
            gen_idx = torch.randint(0, len(gen_list), (1,))
            pil_gen = Image.open(os.path.join(gen_folder, gen_list[gen_idx]))
            gen_tensor = to_tensor(pil_gen).unsqueeze(0)[:, :3,...]
            if mask:
                gen_tensor[mask_tensor.expand_as(gen_tensor) < 0.5] = 1.0

            img_list.append(gen_tensor)
        
        gt_tensor = torch.cat(gt_list, 0)
        img_tensor = torch.cat(img_list, 0)

        kid.update(gt_tensor.to(device), real=True)
        kid.update(img_tensor.to(device), real=False)

    _kid= kid.compute()
    print(_kid)

def compute_pose_errors(gt_path, src_path, mask=False):
    error_list = []

    for gt_img, gen_folder in tqdm(zip(gt_path, src_path)):

        gt = cv2.imread(gt_img)
        gt_pose = get_pose_landmarks(gt)
        
        img_list= [x for x in sorted(os.listdir(gen_folder)) if x.endswith('png')]
        err = 0.0
        for gen_img in img_list:
            gen = cv2.imread(os.path.join(gen_folder, gen_img))
            gen_pose = get_pose_landmarks(gen)
            _err =  compute_errors(gt_pose, gen_pose)
            if np.isnan(_err):
                continue
            err += _err
        mean_err = err/len(img_list)
        error_list.append(mean_err)

    error_list = np.array(error_list)
    error_list = error_list[error_list != 0.0]
    mean_error = np.mean(error_list)
    print('mean error: ', mean_error * 512)
    std = np.std(error_list)
    print('std: ', std * 512)


def main(args):
    # The input folder contains 60 subfolders, each subfolder contains 16 generated images
    input_subfolder =  [os.path.join(args.input_path, x) for x in sorted(os.listdir(args.input_path)) if os.path.isdir(os.path.join(args.input_path, x))]
    # The ground truth folder contains 60 images
    gt_images = [os.path.join(args.gt_path, x) for x in sorted(os.listdir(args.gt_path)) if x.endswith('png')]
   
    compute_kid(gt_images, input_subfolder, mask = args.mask)

    compute_lpips(gt_images, input_subfolder, args.mask)

    compute_pose_errors(gt_images, input_subfolder, args.mask)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', required=True ,type=str)
    parser.add_argument('-g', '--gt_path', required=True ,type=str)
    parser.add_argument('--mask', action='store_true', help='Use GT mask to avoid background noise')
    main(parser.parse_args())
