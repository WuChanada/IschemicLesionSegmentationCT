# ---------------------------------------------------------
# Tensorflow MPC-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Hulin Kuang
# ---------------------------------------------------------
import os
import random
import numpy as np
from datetime import datetime

import utils as utils


class Dataset(object):
    def __init__(self, X_data,gt,mask, flags):

        self.flags = flags
        self.img = X_data
        self.gt = gt
        self.mask = mask

        self.image_size = (512, 512)
        self.ori_shape = (512, 512)
        self.val_ratio = 0.1  # 10% of the training data are used as validation data


        self.num_train, self.num_val, self.num_test = 0, 0, 0

        self._read_data()  # read training, validation, and test data)
        print('num of training images: {}'.format(self.num_train))
        print('num of validation images: {}'.format(self.num_val))
        print('num of test images: {}'.format(self.num_test))

    def _read_data(self):
        if self.flags.is_test:
            # real test images and vessels in the memory
            self.test_imgs, self.test_vessels, self.test_masks, self.test_mean_std = utils.get_test_imgs(
                self.img, self.gt, self.mask,img_size=self.image_size)


            self.num_test = self.test_imgs.shape[0]

        elif not self.flags.is_test:
            random.seed(datetime.now())  # set random seed
            # self.train_img_files, self.train_vessel_files, mask_files = utils.get_img_path(
            #     self.train_dir, self.dataset)
            N = self.img.shape[0]

            self.num_train = int(N)
            self.num_val = int(np.floor(self.val_ratio * N))
            self.num_train -= self.num_val

            self.val_img_raw = self.img[-self.num_val:,:,:,:]
            self.val_gt_raw = self.gt[-self.num_val:,:,:]
            self.val_mask_raw = self.mask[-self.num_val:,:,:]
            self.train_img_raw = self.img[:-self.num_val,:,:,:]
            self.train_gt_raw = self.gt[:-self.num_val,:,:]

            # read val images and vessels in the memory
            self.val_imgs, self.val_vessels, self.val_masks, self.val_mean_std = utils.get_val_imgs(
                self.val_img_raw, self.val_gt_raw, self.val_mask_raw, img_size=self.image_size)

            self.num_val = self.val_imgs.shape[0]

    def train_next_batch(self, batch_size):
        train_indices = np.random.choice(self.num_train, batch_size, replace=True)
        train_imgs, train_vessels = utils.get_train_batch(
            self.train_img_raw, self.train_gt_raw, train_indices.astype(np.int32),
            img_size=self.image_size)
        train_vessels = np.expand_dims(train_vessels, axis=3)

        return train_imgs, train_vessels
