# 输入图像
# feature matching
# classification

import argparse
import os
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils.baselineUtils import load_baseline, extract_ref_features, extract_test_features, transform_mask, prediction
from utils.cgcUtils import classification
from utils.irstd import IRSTD

def get_arguments():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--baseline', type=str, default='MSH')
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

    args = parser.parse_args()
    return args

def main():

    args = get_arguments()

    os.makedirs(os.path.join(args.save_path, args.baseline, args.dataset), exist_ok=True)

    # load baseline
    baseline_model, input_transform = load_baseline(args)

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

        ref_feature = extract_ref_features(args, baseline_model, input_transform, ref_image, ref_mask)
        mask_feature = transform_mask(ref_feature, ref_mask)
        target_feature = ref_feature[mask_feature > 0]
        target_feature = target_feature.mean(0).unsqueeze(0)
        # target_feature = target_feature / target_feature.norm(dim=-1, keepdim=True)
        target_features.append(target_feature)

    target_feature = torch.mean(torch.stack(target_features), dim=0)

    # load test seq
    test_dataset = IRSTD(args, input_transform)
    DatasetLoad = DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=False)
    for it, [ori_image, image, image_name, original_size] in enumerate(tqdm(DatasetLoad)):

        test_feature = extract_test_features(args, baseline_model, image.cuda())
        _, c, w, h = test_feature.shape
        test_feature = test_feature.reshape(args.batch_size, c, w*h)

        sim = target_feature @ test_feature
        sim = sim.reshape(args.batch_size, 1, w, h)
        sim = F.interpolate(sim, [original_size[0][0], original_size[1][0]], mode="bilinear").squeeze()
        sim = (sim - sim.min()) / (sim.max() - sim.min())
        sim = sim.cpu()

        if 0:
            pred = prediction(args, baseline_model, input_transform, ori_image.squeeze(0))
            pred = (pred - pred.min()) / (pred.max() - pred.min())

            sim = (sim.cpu() + to_tensor(pred).squeeze())
            sim = (sim - sim.min()) / (sim.max() - sim.min())
            sim = sim.cpu()

        mask = classification(args, sim, ori_image.squeeze(0))
        mask.save(os.path.join(args.save_path, args.baseline, args.dataset, image_name[0]))

if __name__ == "__main__":
    with torch.no_grad():
        main()