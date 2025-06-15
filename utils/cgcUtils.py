import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def load_clip(args):
    clip_model, preprocess = clip.load("ViT-B/16", device='cuda')
    classnames = ['a photo of a small object', 'a photo of the background']
    prompt_learner = PromptLearner(classnames, clip_model, nctx=args.nctx, token_position=args.token_position)
    model_path = os.path.join('weight', args.dataset, 'prompt_learner', 'model.pth.tar-50')
    pretrained_dict = torch.load(model_path, map_location=torch.device('cuda'))
    prompt_learner.load_state_dict(pretrained_dict['state_dict'])
    prompt_learner.to('cuda')

    text_encoder = TextEncoder(clip_model)
    prompts = prompt_learner()
    tokenized_prompts = prompt_learner.tokenized_prompts
    text_features = text_encoder(prompts, tokenized_prompts)

    return clip_model, preprocess, text_features

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, nctx, token_position):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = nctx
        ctx_init = ''
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to('cuda')
        embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


def classification(args, sim, ori_image):
    masks, patches = crop_patches(sim, ori_image, args.patch_size)

    clip_model, preprocess, text_features = load_clip(args)

    if patches:
        all_patch = []
        for k, patch in enumerate(patches):
            # plt.imshow(patch.cpu(), cmap='viridis')
            # plt.show()
            patch = preprocess(torchvision.transforms.ToPILImage()(patch.permute(2, 0, 1))).to('cuda')
            all_patch.append(patch)

        all_patch = torch.stack(all_patch)
        batch_size = 5
        mask = np.zeros_like(ori_image[:, :, 0])

        for i in range(0, len(all_patch), batch_size):
            batch_patches = all_patch[i:i + batch_size]
            image_features = clip_model.encode_image(batch_patches)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T

            for j, value in enumerate(similarity):
                if value[0] > value[1] * 1:
                    mask += masks[i + j]

        mask = np.uint8(mask)
        mask = Image.fromarray(mask)
    else:
        mask = np.zeros_like(ori_image[:, :, 0])
        mask = np.uint8(mask)
        mask = Image.fromarray(mask)

    return mask

def crop_patches(sim, ori_image, patch_size):

    masks = []
    sim = (sim > 0.5).float().squeeze()
    sim = sim.cpu()
    num_labels_cluster, labels_im_cluster = cv2.connectedComponents(np.uint8(sim * 255))
    for j in range(1, num_labels_cluster):
        mask_single_region = np.zeros_like(sim, dtype=np.uint8)
        mask_single_region[labels_im_cluster == j] = 255
        if np.sum(mask_single_region > 0) <= 900:
            masks.append(mask_single_region)

    patches = []
    for mask in masks:
        non_zero_coords = np.column_stack(np.where(mask > 0))
        if non_zero_coords.size == 0:
            continue

        y_min, x_min = non_zero_coords.min(axis=0)
        y_max, x_max = non_zero_coords.max(axis=0)
        center_x = x_min + (x_max - x_min) // 2
        center_y = y_min + (y_max - y_min) // 2
        half_size_x, half_size_y = patch_size // 2, patch_size // 2
        start_x = max(center_x - half_size_x, 0)
        end_x = min(start_x + patch_size, ori_image.shape[1])
        start_x = end_x - patch_size
        start_y = max(center_y - half_size_y, 0)
        end_y = min(start_y + patch_size, ori_image.shape[0])
        start_y = end_y - patch_size

        patch = ori_image[start_y:end_y, start_x:end_x]
        patches.append(patch)

    return masks, patches