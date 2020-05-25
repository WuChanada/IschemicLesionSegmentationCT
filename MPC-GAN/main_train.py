# ---------------------------------------------------------
# Tensorflow MPC-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Hulin Kuang
# ---------------------------------------------------------
import os
import tensorflow as tf
from solver import Solver
import numpy as np
import nibabel as nib
import scipy.io as sio
import time

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('train_interval', 1, 'training interval between discriminator and generator, default: 1')
tf.flags.DEFINE_integer('ratio_gan2seg', 10, 'ratio of gan loss to seg loss, default: 10')
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_string('dataset', 'Hemorrhage', 'dataset name [Hemorrhage|Infarct], default: Infarct')
tf.flags.DEFINE_bool('is_test', False, 'default: False (train)')#False True

tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 2e-4')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of adam, default: 0.5')
tf.flags.DEFINE_integer('iters', 50000, 'number of iteratons, default: 50000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency, default: 100')
tf.flags.DEFINE_integer('eval_freq', 500, 'evaluation frequency, default: 500')
tf.flags.DEFINE_integer('sample_freq', 200, 'sample frequency, default: 200')

tf.flags.DEFINE_string('checkpoint_dir', './checkpoints', 'models are saved here')
tf.flags.DEFINE_string('sample_dir', './sample', 'sample are saved here')
tf.flags.DEFINE_string('test_dir', './test', 'test images are saved here')


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

im_root = '../traindata/'
lst = os.listdir(im_root)
idname = lst #lst[1:]
N = len(idname)

nx = 512
ny = 512
img = np.empty((nx, ny, 0), dtype=np.float32)
gt = np.empty((nx, ny, 0), dtype=np.float32)
dif7 = np.empty((nx, ny, 0), dtype=np.float32)


mask = np.empty((nx, ny, 0), dtype=np.float32)
dist = np.empty((nx, ny, 0), dtype=np.float32)
loc = np.empty((nx, ny, 0), dtype=np.float32)


for ind in range(0,N):
    # load GT
    niiname = im_root + idname[ind] + '/lesionGT.nii.gz'
    if os.path.exists(niiname) is False:
        continue
    gt_sub = nib.load(niiname)
    gt_data = gt_sub.get_data()
    nx0 = gt_data.shape[0]
    ny0 = gt_data.shape[1]

    if nx0 != nx | ny0 != ny:
        continue

    # load mask
    niiname = im_root + idname[ind] + '/brain_mask.nii.gz'
    img_sub = nib.load(niiname)
    img_data = img_sub.get_data()
    mask_data = img_data
    # print(img_data.shape)
    mask_size_z = np.sum(np.sum(img_data, axis=0), axis=0)
    ind0 = np.where(mask_size_z >= 2500)

    mask = np.append(mask, img_data[:, :, ind0[0]], axis=2)

    gt_data = gt_sub.get_data()
    #lesion_size_z = np.sum(np.sum(gt_data, axis = 0), axis = 0)
    #ind0 = np.where(lesion_size_z >= 0)

    gt_data = np.float32(gt_data)


    gt = np.append(gt, gt_data[:,:,ind0[0]], axis=2)


    # load cropped norm 0-1 img
    niiname = im_root + idname[ind] + '/ncct_brainRZNorm.nii.gz'
    img_sub = nib.load(niiname)

    img_data = img_sub.get_data()
    img = np.append(img, img_data[:,:,ind0[0]], axis=2)

   # load difference med7
    niiname = im_root + idname[ind] + '/DifMed7.nii.gz'
    img_sub = nib.load(niiname)
    img_data = img_sub.get_data()
    # print(img_data.shape)
    dif7 = np.append(dif7, img_data[:,:,ind0[0]], axis=2)


    # load distance
    niiname = im_root + idname[ind] + '/dist.nii.gz'
    img_sub = nib.load(niiname)
    img_data = img_sub.get_data()
    # print(img_data.shape)
    dist = np.append(dist, img_data[:, :, ind0[0]], axis=2)

    # load location prob
    niiname = im_root + idname[ind] + '/locprob.nii.gz'
    img_sub = nib.load(niiname)
    img_data = img_sub.get_data()
    # print(img_data.shape)
    loc = np.append(loc, img_data[:, :, ind0[0]], axis=2)


img = np.transpose(img, [2, 0, 1])
dif7 = np.transpose(dif7, [2, 0, 1])
mask = np.transpose(mask, [2, 0, 1])
gt = np.transpose(gt, [2, 0, 1])
dist = np.transpose(dist, [2, 0, 1])
loc = np.transpose(loc, [2, 0, 1])

img = np.multiply(img,mask)
dif7 = np.multiply(dif7,mask)
gt = np.multiply(gt,mask)
loc = np.multiply(loc,mask)
dist = np.multiply(dist,mask)


gt[gt<=0] = 0
gt[gt>0] = 1
nz_all = img.shape[0]
ny_all = img.shape[1]
nx_all = img.shape[2]

X_data = np.zeros((nz_all, ny_all, nx_all, 4), dtype=np.float32)
X_data[:, :, :, 0] = img
X_data[:, :, :, 1] = dif7
X_data[:, :, :, 2] = dist
X_data[:, :, :, 3] = loc

solver = Solver(FLAGS)
if not FLAGS.is_test:
    solver.train(X_data,gt,mask)



