import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class one_shot(DatasetBase):

    dataset_dir = "one-shot"

    def __init__(self, cfg):
        root = 'one-shot'
        self.dataset_dir = os.path.join(root, cfg.DATASETNAME)
        self.image_dir = self.dataset_dir

        # Load class names and create a mapping from class names to labels
        # Modify this part according to how IRSTD class names are stored
        classnames = ['a photo of a small object', 'a photo of the background']
        cname2lab = {c: i for i, c in enumerate(classnames)}

        # Modify file names according to IRSTD dataset
        trainval = self.read_data(cname2lab, "train.txt")
        test = self.read_data(cname2lab, "train.txt")

        # Split trainval into train and val
        train, val = OxfordPets.split_trainval(trainval)

        # Few-shot learning setup
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
            data = {"train": train, "val": val}


        # Subsample classes if needed
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        # Modify this part according to how IRSTD dataset stores image file names and class names
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Adjust parsing logic as per IRSTD dataset format
                words = line.strip().split()
                imname = words[0]
                classname = ' '.join(words[1:])

                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname + '.png')
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
