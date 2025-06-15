import torch
import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.baselineUtils import load_baseline, prediction, generate_patch
from utils.prompt_tuning import prompt_tuning

def get_arguments():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--baseline', type=str, default='DNA')

    # data
    parser.add_argument('--base_path', type=str, default='IRDST/real')
    parser.add_argument('--dataset', type=str, default='1')
    parser.add_argument('--ref_idx', type=str, default='1(1)')

    # method
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--save_path', type=str, default='one-shot')

    args = parser.parse_args()
    return args

def main():

    args = get_arguments()

    # path
    os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)

    # load training image
    ref_image_path = os.path.join(args.base_path, 'images', args.dataset, args.ref_idx + '.png')
    ref_mask_path = os.path.join(args.base_path, 'masks', args.dataset, args.ref_idx + '.png')
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)

    # load baseline model and extract feature map
    with torch.no_grad():
        baseline_model, input_transform = load_baseline(args)
        pred = prediction(args, baseline_model, input_transform, ref_image)
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred = np.where(pred > 0.4, 255, 0).astype(np.uint8)

    # crop T&F patch
    generate_patch(args, ref_image, ref_mask, pred)

    # prompt tuning
    prompt_tuning(args.dataset)


if __name__ == "__main__":
        main()