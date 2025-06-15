from torchvision import transforms
import cv2
import torch
import numpy as np
from PIL import Image
import os

from baseline.DNANet.model.model_DNANet import DNANet, Res_CBAM_block
from baseline.DNANet.model.utils import *
from baseline.MSHNet.model.MSHNet import *

def load_baseline(args):
    if args.baseline == 'DNA':
        baseline_model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2],
                                nb_filter=[16, 32, 64, 128, 256], deep_supervision='True')
        baseline_model = baseline_model.cuda()
        baseline_model.apply(weights_init_xavier)
        checkpoint = torch.load('baseline/DNANet/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar')
        baseline_model.load_state_dict(checkpoint['state_dict'])
        baseline_model.eval()

        input_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    elif args.baseline == 'MSH':
        baseline_model = MSHNet(3)
        weight = torch.load('baseline/MSHNet/IRSTD-1k_weight.tar')
        baseline_model.load_state_dict(weight['state_dict'])
        baseline_model = baseline_model.cuda()
        baseline_model.eval()
        input_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    return baseline_model, input_transform

def load_FT_baseline(args):
    if args.baseline == 'DNA':
        baseline_model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2],
                                nb_filter=[16, 32, 64, 128, 256], deep_supervision='True')
        baseline_model = baseline_model.cuda()
        baseline_model.apply(weights_init_xavier)
        checkpoint = torch.load(f'finetuning/{args.dataset}/DNA.pth.tar')
        baseline_model.load_state_dict(checkpoint['state_dict'])
        baseline_model.eval()

        input_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    elif args.baseline == 'MSH':
        baseline_model = MSHNet(3)
        weight = torch.load(f'finetuning/{args.dataset}/MSH.pth.tar')
        baseline_model.load_state_dict(weight['state_dict'])
        baseline_model = baseline_model.cuda()
        baseline_model.eval()
        input_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    return baseline_model, input_transform


def load_LP_baseline(args):
    if args.baseline == 'DNA':
        baseline_model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2],
                                nb_filter=[16, 32, 64, 128, 256], deep_supervision='True')
        baseline_model = baseline_model.cuda()
        baseline_model.apply(weights_init_xavier)
        checkpoint = torch.load(f'linearprobing/{args.dataset}/DNA.pth.tar')
        baseline_model.load_state_dict(checkpoint['state_dict'])
        baseline_model.eval()

        input_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    elif args.baseline == 'MSH':
        baseline_model = MSHNet(3)
        weight = torch.load(f'linearprobing/{args.dataset}/MSH.pth.tar')
        baseline_model.load_state_dict(weight['state_dict'])
        baseline_model = baseline_model.cuda()
        baseline_model.eval()
        input_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    return baseline_model, input_transform

def prediction(args, baseline_model, input_transform, ref_image):
    original_size = [ref_image.shape[1], ref_image.shape[0]]

    if args.baseline == 'DNA':
        TImage = Image.fromarray(np.uint8(ref_image))
        TImage = input_transform(TImage).unsqueeze(0).cuda()
        pred = baseline_model(TImage)
        pred = pred[-1].squeeze()
        pred = pred.cpu().numpy()
        pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)

    elif args.baseline == 'MSH':
        TImage = Image.fromarray(np.uint8(ref_image))
        TImage = input_transform(TImage).unsqueeze(0).cuda()
        _, pred = baseline_model(TImage, False)
        pred = pred[0].squeeze()
        pred = pred.cpu().numpy()
        pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)

    return pred

def extract_ref_features(args, baseline_model, input_transform, image, mask):

    image = Image.fromarray(np.uint8(image))
    image = input_transform(image).unsqueeze(0).cuda()

    if args.baseline == 'DNA':
        feature = baseline_model.featureExtract(image).squeeze()  # 16*512*512
        feature = feature.permute(1, 2, 0)  # 32*32*256

    elif args.baseline == 'MSH':
        feature = baseline_model.featureExtract(image).squeeze()  # 256*32*32
        feature = feature.permute(1, 2, 0)  # 32*32*256

    return feature

def extract_test_features(args, baseline_model, image):

    if args.baseline == 'DNA':
        feature = baseline_model.featureExtract(image)

    elif args.baseline == 'MSH':
        feature = baseline_model.featureExtract(image)

    return feature


def transform_mask(ref_feature, ref_mask):
    w, h, _ = ref_feature.shape

    mask_transform = transforms.Compose([
        transforms.Resize([w, h]),
        transforms.ToTensor()])
    ref_mask = Image.fromarray(np.uint8(ref_mask))
    mask_feature = mask_transform(ref_mask).permute(1, 2, 0)
    mask_feature = mask_feature[:, :, 0]

    return mask_feature

def generate_patch(args, ref_image, ref_mask, pred):

    train_file_path = os.path.join(args.save_path, args.dataset, 'train.txt')
    with open(train_file_path, 'w') as file:
        pass

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ref_mask, connectivity=8)
    for i in range(1, num_labels):
        centroid = centroids[i]
        left = int(centroid[0] - args.patch_size / 2)
        upper = int(centroid[1] - args.patch_size / 2)
        right = left + args.patch_size
        lower = upper + args.patch_size
        left, upper, right, lower = max(0, left), max(0, upper), min(ref_image.shape[1], right), min(ref_image.shape[0],
                                                                                                     lower)
        target_patch = ref_image[upper:lower, left:right]
        cv2.imwrite(os.path.join(args.save_path, args.dataset, 'T.png'), target_patch)
        with open(train_file_path, 'a') as file:
            file.write('T a photo of a small object\n')

    num_labels_pred, labels_pred, stats_pred, centroids_pred = cv2.connectedComponentsWithStats(pred, connectivity=8)
    fa_index = 1
    for i in range(1, num_labels_pred):
        centroid = centroids_pred[i]
        left = int(centroid[0] - args.patch_size / 2)
        upper = int(centroid[1] - args.patch_size / 2)
        right = left + args.patch_size
        lower = upper + args.patch_size
        left, upper, right, lower = max(0, left), max(0, upper), min(ref_image.shape[1], right), min(ref_image.shape[0],
                                                                                                     lower)
        component_mask = (labels_pred == i).astype(np.uint8)
        overlap = np.sum(component_mask & ref_mask)
        if overlap == 0:
            fa_patch = ref_image[upper:lower, left:right]
            cv2.imwrite(os.path.join(args.save_path, args.dataset, f'FA_{fa_index}.png'), fa_patch)
            with open(train_file_path, 'a') as file:
                file.write(f'FA_{fa_index} a photo of the background\n')
            fa_index += 1