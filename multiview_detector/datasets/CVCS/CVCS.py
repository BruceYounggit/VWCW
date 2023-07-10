import os
import random

import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset


class CVCS(VisionDataset):
    def __init__(self, root):
        super(CVCS, self).__init__(root)
        self.__name__ = 'CVCS'
        self.img_shape = [1080, 1920]
        self.view_size = 5
        self.num_cam = self.view_size
        self.patch_num = 5  # 5
        self.num_frame = 5 * 23 * 100

        self.train_data = os.path.join(root, 'train/')
        self.val_data = os.path.join(root, 'val/')
        self.train_label = os.path.join(root + '/labels/100frames_labels_reproduce_640_480_CVCS/'
                                               '100frames_labels_reproduce_640_480_CVCS/', 'train/')
        self.val_label = os.path.join(root + '/labels/100frames_labels_reproduce_640_480_CVCS/'
                                             '100frames_labels_reproduce_640_480_CVCS/', 'val/')


if __name__ == '__main__':
    root = os.path.expanduser('~/Data/CVCS')
    base = CVCS(root)
    pass
    # vp = city.get_img_fpath()
    # print(vp[1][636])
