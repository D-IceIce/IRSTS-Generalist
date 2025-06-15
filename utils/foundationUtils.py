import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import label, center_of_mass
from torch.nn import functional as F
from torch.nn import Upsample
from matplotlib import pyplot as plt

from foundation.dino import vision_transformer as vits
from foundation.SAM import sam_model_registry, SamPredictor
from foundation.SAM.utils.transforms import ResizeLongestSide

def load_model(args):
    if args.model == 'DINO':
        foundation_model = vits.__dict__['vit_small'](patch_size=8, num_classes=0)
        foundation_model = foundation_model.cuda()
        foundation_model.eval()
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        foundation_model.load_state_dict(state_dict, strict=True)

        input_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
        ])

    elif args.model == 'SAM':
        sam_type, sam_ckpt = 'vit_t', 'foundation\SAM\mobile_sam.pt'
        foundation_model = sam_model_registry[sam_type](checkpoint=sam_ckpt)
        foundation_model = foundation_model.cuda()
        foundation_model.eval()

        input_transform = ResizeLongestSide(foundation_model.image_encoder.img_size)

    return foundation_model, input_transform

def extract_ref_local_features(args, foundation_model, input_transform, image, mask):

    target_patch, mask_patch = imageCrop(args, image, mask)

    if args.model == 'DINO':
        feature = extract_features(args, foundation_model, input_transform, target_patch)
        feature = feature.reshape(int(feature.shape[0] ** 0.5), int(feature.shape[0] ** 0.5), feature.shape[1])
    elif args.model == 'SAM':
        feature = extract_features(args, foundation_model, input_transform, target_patch)
        feature = feature.permute(1, 2, 0)

    mask_feature = transform_mask(feature, mask_patch)

    return feature, mask_feature

def extract_features(args, foundation_model, input_transform, image):

    if args.model == 'DINO':
        image = Image.fromarray(np.uint8(image))
        image = input_transform(image).unsqueeze(0).cuda()
        feature = foundation_model.featureExtract(image).squeeze()
    elif args.model == 'SAM':
        image = input_transform.apply_image(image)
        image = torch.from_numpy(image).cuda()
        image = image.permute(2, 0, 1).contiguous()[:, :, :]
        image = foundation_model.preprocess(image).unsqueeze(0)
        feature = foundation_model.image_encoder(image).squeeze()

    return feature

def extract_attention(args, foundation_model, input_transform, test_image):

    if args.model == 'DINO':
        image = Image.fromarray(np.uint8(test_image))
        image = input_transform(image).unsqueeze(0).cuda()
        attentions = foundation_model.get_last_selfattention(image) #1*6*4097*4097
        nh = attentions.shape[1]
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1) #6*4096
        attentions = attentions.reshape(nh, int(attentions.shape[1] ** 0.5), int(attentions.shape[1] ** 0.5))
        attention = attentions.mean(0)
        attention = F.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(test_image.shape[0], test_image.shape[1]),
                                mode='bilinear').squeeze()
    elif args.model == 'SAM':
        image = input_transform.apply_image(test_image)
        image = torch.from_numpy(image).cuda()
        image = image.permute(2, 0, 1).contiguous()[:, :, :]
        image = foundation_model.preprocess(image).unsqueeze(0)
        feature = foundation_model.image_encoder(image).squeeze()
        feature = feature.mean(0)
        feature = F.interpolate(feature.unsqueeze(0).unsqueeze(0), size=(image.shape[2], image.shape[3]),
                                  mode='bilinear', align_corners=False).squeeze()
        attention = F.interpolate(feature.unsqueeze(0).unsqueeze(0), size=(test_image.shape[0], test_image.shape[1]),
                                mode='bilinear', align_corners=False).squeeze()
        attention = attention

    elif args.model == 'SAM0':
        image = input_transform.apply_image(test_image)
        image = torch.from_numpy(image).cuda()
        image = image.permute(2, 0, 1).contiguous()[:, :, :]
        image = foundation_model.preprocess(image).unsqueeze(0)
        attentions = foundation_model.image_encoder.get_last_selfattention(image) #1*10*4096*4096
        nh = attentions.shape[1]  # Number of heads
        attentions = attentions[0, :, 0, :].reshape(nh, -1)
        attentions = attentions.reshape(nh, int(attentions.shape[1] ** 0.5), int(attentions.shape[1] ** 0.5))
        # attention = attentions.mean(0)
        attention = attentions[6]
        attention = F.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(test_image.shape[0], test_image.shape[1]),
                                  mode='bilinear').squeeze()

    return attention

def imageCrop(args, image, mask):

    labeled_mask, num_features = label(mask)
    centroid = center_of_mass(mask, labeled_mask, range(1, num_features + 1))
    centroid = centroid[0]

    M, N, C = image.shape
    patch_size = ensure_even(M // args.win_size)
    center_y, center_x = int(centroid[0]), int(centroid[1])
    start_y = max(center_y - patch_size // 2, 0)
    end_y = min(center_y + patch_size // 2, M)
    start_x = max(center_x - patch_size // 2, 0)
    end_x = min(center_x + patch_size // 2, N)

    target_patch = np.zeros((patch_size, patch_size, C), dtype=image.dtype)
    mask_patch = np.zeros((patch_size, patch_size, C), dtype=mask.dtype)

    copy_start_y = max(patch_size // 2 - center_y, 0)
    copy_start_x = max(patch_size // 2 - center_x, 0)
    copy_end_y = copy_start_y + (end_y - start_y)
    copy_end_x = copy_start_x + (end_x - start_x)

    target_patch[copy_start_y:copy_end_y, copy_start_x:copy_end_x] = image[start_y:end_y, start_x:end_x]
    mask_patch[copy_start_y:copy_end_y, copy_start_x:copy_end_x] = mask[start_y:end_y, start_x:end_x]

    return target_patch, mask_patch

def ensure_even(size):
    return size if size % 2 == 0 else size - 1

def transform_mask(ref_feature, ref_mask):
    w, h, _ = ref_feature.shape

    mask_transform = transforms.Compose([
        transforms.Resize([w, h]),
        transforms.ToTensor()])
    ref_mask = Image.fromarray(np.uint8(ref_mask))
    mask_feature = mask_transform(ref_mask).permute(1, 2, 0)
    mask_feature = mask_feature[:, :, 0]

    return mask_feature


def extract_test_local_features(args, foundation_model, input_transform, test_image):
    patches = Image2Patches(args, test_image)

    if args.model == 'DINO':
        local_features = []
        for patch in patches:
            local_feature = extract_features(args, foundation_model, input_transform, patch)
            local_features.append(local_feature)
    elif args.model == 'SAM':
        local_features = []
        for patch in patches:
            local_feature = extract_features(args, foundation_model, input_transform, patch)
            C, h, w = local_feature.shape
            local_feature = local_feature / local_feature.norm(dim=0, keepdim=True)
            local_feature = local_feature.reshape(C, h * w)
            local_feature = local_feature.permute(1, 0)
            local_features.append(local_feature)

    return local_features


def Image2Patches(args, image):
    patch_size = ensure_even(image.shape[0] // args.win_size)
    step_size = int(patch_size * (1 - args.overlap_ratio))

    patches = []
    for i in range(0, image.shape[0], step_size):
        for j in range(0, image.shape[1], step_size):
            end_i = min(i + patch_size, image.shape[0])
            end_j = min(j + patch_size, image.shape[1])

            patch = image[i:end_i, j:end_j]

            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                full_patch = np.zeros((patch_size, patch_size, patch.shape[2]), dtype=patch.dtype)
                full_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = full_patch
            patches.append(patch)

    return patches

def Patches2Image(args, patches, image):
    patch_size = ensure_even(image.shape[0] // args.win_size)
    step_size = int(patch_size * (1 - args.overlap_ratio))

    reconstructed_image = torch.zeros(image.shape[:2]).cuda()
    count = torch.zeros(image.shape[:2]).cuda()

    patch_index = 0
    for i in range(0, image.shape[0], step_size):
        for j in range(0, image.shape[1], step_size):
            end_i = min(i + patch_size, image.shape[0])
            end_j = min(j + patch_size, image.shape[1])

            reconstructed_image[i:end_i, j:end_j] += patches[patch_index][:end_i - i, :end_j - j]
            count[i:end_i, j:end_j] += 1
            patch_index += 1

    count = torch.clamp(count, min=1)
    reconstructed_image = reconstructed_image / count

    return reconstructed_image