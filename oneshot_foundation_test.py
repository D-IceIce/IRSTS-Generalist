import argparse
import os
import cv2
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from utils.foundationUtils import load_model, extract_ref_local_features, extract_test_local_features, ensure_even, Patches2Image, extract_attention
from utils.cgcUtils import classification
from utils.irstd import IRSTD
import warnings
warnings.filterwarnings('ignore')

def get_arguments():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model', type=str, default='SAM')
    parser.add_argument('--nctx', type=int, default=8)
    parser.add_argument('--token_position', default='end', type=str)

    # data
    parser.add_argument('--base_path', type=str, default='IRDST/real')
    parser.add_argument('--dataset', type=str, default='1')
    parser.add_argument('--ref_idx', type=str, default='1(1)')
    parser.add_argument('--save_path', type=str, default='results')

    # method
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--win_size', type=int, default=3)
    parser.add_argument('--overlap_ratio', type=int, default=0.1)

    args = parser.parse_args()
    return args

def main():

    args = get_arguments()

    os.makedirs(os.path.join(args.save_path, args.model, args.dataset), exist_ok=True)

    # load baseline
    foundation_model, input_transform = load_model(args)

    # Extract Target Feature
    ref_list = args.ref_idx.split(',')
    target_features = []
    for ref_name in ref_list:
        ref_image_path = os.path.join(args.base_path, 'images', args.dataset, ref_name + '.png')
        ref_mask_path = os.path.join(args.base_path, 'masks', args.dataset, ref_name + '.png')

        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

        ref_feature, mask_feature = extract_ref_local_features(args, foundation_model, input_transform, ref_image, ref_mask)
        target_feature = ref_feature[mask_feature > 0]
        target_embedding = target_feature.mean(0).unsqueeze(0)
        target_feature = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_features.append(target_feature)

    target_feature = torch.mean(torch.stack(target_features), dim=0)

    test_path = os.path.join(args.base_path, 'images', args.dataset)
    path_list = [os.path.join(test_path, img) for img in os.listdir(test_path)]
    for test_image_path in tqdm(path_list):
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        patch_size = ensure_even(test_image.shape[0] // args.win_size)

        # extract local features of test_image
        sim_list = []
        local_features = extract_test_local_features(args, foundation_model, input_transform, test_image)
        for local_feature in local_features:
            local_feature = local_feature.permute(1, 0).unsqueeze(0) # 1*384*4096
            sim = target_feature @ local_feature
            sim = sim.reshape(sim.shape[0], sim.shape[1], int(sim.shape[2] ** 0.5), int(sim.shape[2] ** 0.5))
            sim = F.interpolate(sim, [patch_size, patch_size], mode="bilinear").squeeze()
            sim_list.append(sim)
        sim = Patches2Image(args, sim_list, test_image)
        sim = (sim - sim.min()) / (sim.max() - sim.min())
        plt.imshow(sim.cpu(), cmap='viridis')
        plt.show()

        if 0:
            attention = extract_attention(args, foundation_model, input_transform, test_image)
            sim = sim * attention
            sim = (sim - sim.min()) / (sim.max() - sim.min())

        mask = classification(args, sim, torch.from_numpy(test_image))
        mask.save(os.path.join(args.save_path, args.model, args.dataset, os.path.basename(test_image_path)))



if __name__ == "__main__":
    with torch.no_grad():
        main()