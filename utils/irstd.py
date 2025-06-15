import os.path

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

class IRSTD(Dataset):
    def __init__(self, args, transform):
        self.transform = transform
        self.image_dir = os.path.join(args.base_path, 'images', args.dataset)
        self.image_path = [os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_name = os.path.basename(self.image_path[idx])

        ori_image = cv2.imread(self.image_path[idx])
        original_size = ori_image.shape[:2]
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image)

        return ori_image, image, image_name, original_size