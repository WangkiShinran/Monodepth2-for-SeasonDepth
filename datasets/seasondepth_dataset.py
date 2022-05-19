# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

import cv2
from .mono_dataset import MonoDataset


class SeasonDepthDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(SeasonDepthDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.

        # c0
        
        self.K = np.array([[0.848626, 0, 0.513616, 0],
                           [0, 1.127686, 0.546930, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        '''
        # c1
        self.K = np.array([[0.852913, 0, 0.516918, 0],
                           [0, 1.141262, 0.517282, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        '''
        

        self.full_res_shape = (1024, 768)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split("/")
        folder = self.filenames[0]
        frame_index = ''
        side = ''

        if line[4] == 'images':  # train_set
            name = line[5].split(".")
            velo_filename = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth","train", line[1], line[2], line[3], 'depth_map', name[0] + ".png")
        elif line[1] == 'images':  # val_set
            name = line[4].split(".")
            velo_filename = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth","val", "depth", line[2], line[3], name[0] + ".png")
        velo_filename = '/' + velo_filename
        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class SDDataset(SeasonDepthDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(SDDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        line = folder.split("/")

        if line[4] == 'images':  # train_set
            img_pth = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth", "train", line[1], line[2], line[3], line[4],line[5])
        elif line[1] == 'images':  # val_set
            img_pth = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth", "val", line[1], line[2], line[3], line[4])
        img_pth = '/' + img_pth
        return img_pth

    def get_depth(self, folder, frame_index, side, do_flip):
        line = folder.split("/")

        if line[4] == 'images':  # train_set
            name = line[5].split(".")
            velo_filename = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth", "train", line[1], line[2], line[3], 'depth_map',
                name[0] + ".png")
        elif line[1] == 'images':  # val_set
            name = line[4].split(".")
            velo_filename = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth", "val", "depth", line[2], line[3], name[0] + ".png")
        velo_filename = '/' + velo_filename
        depth_gt = cv2.imread(velo_filename, -1)
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt