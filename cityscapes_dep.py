import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(1024, 2048),
        augmentations=None,
        img_norm=True,
        version="cityscapes",
        test_mode = False,
        n_classes = 1,

    # ---------------------------------------------------------
    # remove all image related inputs (e.g. rgb)

    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = n_classes
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)

        # add depth path
        self.depths_base = os.path.join(self.root, "disparity", self.split)


        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()

        # -------------------------------------------------
        # add depth path

        dep_path = os.path.join(
            self.depths_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "disparity.png",
        )

        img = m.imread(img_path)
        img = np.array(img, dtype='uint8')

        # -------------------------------------------------
        # add depth
        dep = m.imread(dep_path)
        #dep = np.array(dep, dtype=np.float)

        # -------------------------------------------------
        # add masks
        mask = np.array(dep != 0, dtype ='uint8')

        dep=(dep-1) / 256
        dep=dep/100

        # -------------------------------------------------
        # augmentations (remove label)

        if self.augmentations is not None:
            img, mask, dep = self.augmentations(img, mask, dep)

        #print(img.shape)
        if self.is_transform:
            img, dep = self.transform(img, dep)

        # -------------------------------------------------
        # convert masks
        mask = torch.from_numpy(mask)

        return img, dep, mask

    def transform(self, img, dep):
        """transform

        :param img:
        :param dep:
        :param mask:
        """
        #img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        dep = dep.astype(float)
        #dep = m.imresize(dep, (self.img_size[0], self.img_size[1]), "bilinear", mode="F")

        img = torch.from_numpy(img).float()
        dep = torch.from_numpy(dep).float()

        return img, dep