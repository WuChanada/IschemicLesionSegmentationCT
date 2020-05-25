# ---------------------------------------------------------
# Tensorflow MPC-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Hulin Kuang
# ---------------------------------------------------------
import tensorflow as tf
# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
import numpy as np
import utils as utils

class MPCGAN(object):
    def __init__(self, sess, flags, image_size,channels):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.channels = channels

        self.alpha_recip = 1. / self.flags.ratio_gan2seg if self.flags.ratio_gan2seg > 0 else 0
        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c, self.dis_c = 32, 32

        self._build_net()
        self._init_assign_op()  # initialize assign operations

        print('Initialized MPCGAN SUCCESS!\n')

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, *self.image_size, 1], name='image')
        self.X1 = tf.placeholder(tf.float32, shape=[None, *self.image_size, 1], name='image1')
        self.X2 = tf.placeholder(tf.float32, shape=[None, *self.image_size, 1], name='image2')
        self.X3 = tf.placeholder(tf.float32, shape=[None, *self.image_size, 1], name='image3')
        self.Y = tf.placeholder(tf.float32, shape=[None, *self.image_size, 1], name='vessel')

        # self.g_samples_mirror = self.generator(self.X)
        self.g_samples = self.generator(self.X, self.X1, self.X2, self.X3)

        self.real_pair = tf.concat([self.X, self.X1, self.X2, self.X3, self.Y], axis=3)
        self.fake_pair = tf.concat([self.X, self.X1, self.X2, self.X3, self.g_samples], axis=3)
        # self.real_pair = tf.concat([self.X, self.g_samples_mirror, self.Y], axis=3)
        # self.fake_pair = tf.concat([self.X, self.g_samples_mirror, self.g_samples], axis=3)
        

        d_real, d_logit_real = self.discriminator(self.real_pair)
        d_fake, d_logit_fake = self.discriminator(self.fake_pair, is_reuse=True)#True

        # discrminator loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # generator loss
        gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
        seg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_samples, labels=self.Y))
        #seg_loss = -dice_coeff
        self.g_loss = self.alpha_recip * gan_loss + seg_loss

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        dis_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
            .minimize(self.d_loss, var_list=d_vars)
        dis_ops = [dis_op] + self._dis_train_ops
        self.dis_optim = tf.group(*dis_ops)

        gen_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
            .minimize(self.g_loss, var_list=g_vars)
        gen_ops = [gen_op] + self._gen_train_ops
        self.gen_optim = tf.group(*gen_ops)

    def _init_assign_op(self):
        self.best_auc_sum_placeholder = tf.placeholder(tf.float32, name='best_auc_sum_placeholder')
        self.auc_pr_placeholder = tf.placeholder(tf.float32, name='auc_pr_placeholder')
        self.auc_roc_placeholder = tf.placeholder(tf.float32, name='auc_roc_placeholder')
        self.dice_coeff_placeholder = tf.placeholder(tf.float32, name='dice_coeff_placeholder')
        self.acc_placeholder = tf.placeholder(tf.float32, name='acc_placeholder')
        self.sensitivity_placeholder = tf.placeholder(tf.float32, name='sensitivity_placeholder')
        self.specificity_placeholder = tf.placeholder(tf.float32, name='specificity_placeholder')
        self.score_placeholder = tf.placeholder(tf.float32, name='score_placeholder')
        self.gloss_save_placeholder = tf.placeholder(tf.float32, name='gloss_placeholder')
        self.dloss_save_placeholder = tf.placeholder(tf.float32, name='dloss_placeholder')

        self.best_auc_sum = tf.Variable(0., trainable=False, dtype=tf.float32, name='best_auc_sum')
        auc_pr = tf.Variable(0., trainable=False, dtype=tf.float32, name='auc_pr')
        auc_roc = tf.Variable(0., trainable=False, dtype=tf.float32, name='auc_roc')
        dice_coeff = tf.Variable(0., trainable=False, dtype=tf.float32, name='dice_coeff')
        acc = tf.Variable(0., trainable=False, dtype=tf.float32, name='acc')
        sensitivity = tf.Variable(0., trainable=False, dtype=tf.float32, name='sensitivity')
        specificity = tf.Variable(0., trainable=False, dtype=tf.float32, name='specificity')
        score = tf.Variable(0., trainable=False, dtype=tf.float32, name='score')
        dloss_save = tf.Variable(0., trainable=False, dtype=tf.float32, name='dloss_save')
        gloss_save = tf.Variable(0., trainable=False, dtype=tf.float32, name='gloss_save')

        self.best_auc_sum_assign_op = self.best_auc_sum.assign(self.best_auc_sum_placeholder)
        auc_pr_assign_op = auc_pr.assign(self.auc_pr_placeholder)
        auc_roc_assign_op = auc_roc.assign(self.auc_roc_placeholder)
        dice_coeff_assign_op = dice_coeff.assign(self.dice_coeff_placeholder)
        acc_assign_op = acc.assign(self.acc_placeholder)
        sensitivity_assign_op = sensitivity.assign(self.sensitivity_placeholder)
        specificity_assign_op = specificity.assign(self.specificity_placeholder)
        dloss_save_op = dloss_save.assign(self.dloss_save_placeholder)
        gloss_save_op = gloss_save.assign(self.gloss_save_placeholder)
        score_assign_op = dloss_save.assign(self.score_placeholder)

        self.measure_assign_op = tf.group(auc_pr_assign_op, auc_roc_assign_op, dice_coeff_assign_op,
                                          acc_assign_op, sensitivity_assign_op, specificity_assign_op,
                                          score_assign_op)

        self.measure_loss_op = tf.group(dloss_save_op, gloss_save_op)

        # for tensorboard
        if not self.flags.is_test:
            self.writer = tf.summary.FileWriter("{}/logs/{}_{}_{}".format(
                self.flags.dataset, self.flags.discriminator, self.flags.train_interval, self.flags.batch_size))
            # self.writer_loss = tf.summary.FileWriter("{}/logs_loss/{}_{}_{}".format(
            #     self.flags.dataset, self.flags.discriminator, self.flags.train_interval, self.flags.batch_size))

        auc_pr_summ = tf.summary.scalar("auc_pr_summary", auc_pr)
        auc_roc_summ = tf.summary.scalar("auc_roc_summary", auc_roc)
        dice_coeff_summ = tf.summary.scalar("dice_coeff_summary", dice_coeff)
        acc_summ = tf.summary.scalar("acc_summary", acc)
        sensitivity_summ = tf.summary.scalar("sensitivity_summary", sensitivity)
        specificity_summ = tf.summary.scalar("specificity_summary", specificity)
        score_summ = tf.summary.scalar("score_summary", score)
        gloss_summ = tf.summary.scalar("gloss_summary", gloss_save)
        dloss_summ = tf.summary.scalar("dloss_summary", dloss_save)

        self.measure_summary = tf.summary.merge([auc_pr_summ, auc_roc_summ, dice_coeff_summ, acc_summ,
                                                 sensitivity_summ, specificity_summ, score_summ])

        self.measure_loss_summary = tf.summary.merge([ gloss_summ, dloss_summ])

    def generator(self, data, data1, data2, data3, name='g_'):
        with tf.variable_scope(name):
            # difference path
            # conv1_1: (N, 512, 512, 1) -> (N, 320, 320, 32)
            conv1_1 = tf_utils.conv2d(data1, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_1_conv1_1')
            conv1_1 = tf_utils.batch_norm(conv1_1, name='conv1_1_batch1', _ops=self._gen_train_ops)
            conv1_1 = tf.nn.relu(conv1_1, name='conv1_1_relu1')
            conv1_1 = tf_utils.conv2d(conv1_1, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_1_conv1_2')
            conv1_1 = tf_utils.batch_norm(conv1_1, name='conv1_1_batch2', _ops=self._gen_train_ops)
            conv1_1 = tf.nn.relu(conv1_1, name='conv1_1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv1_1, name='maxpool1')

            # conv1_2: (N, 320, 320, 32) -> (N, 160, 160, 64)
            conv1_2 = tf_utils.conv2d(pool1, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_2_conv1_1')
            conv1_2 = tf_utils.batch_norm(conv1_2, name='conv1_2_batch1', _ops=self._gen_train_ops)
            conv1_2 = tf.nn.relu(conv1_2, name='conv1_2_relu1')
            conv1_2 = tf_utils.conv2d(conv1_2, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_2_conv1_2')
            conv1_2 = tf_utils.batch_norm(conv1_2, name='conv1_2-batch2', _ops=self._gen_train_ops)
            conv1_2 = tf.nn.relu(conv1_2, name='conv1_2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv1_2, name='maxpool2')

            # conv1_3: (N, 160, 160, 64) -> (N, 80, 80, 128)
            conv1_3 = tf_utils.conv2d(pool2, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_3_conv1_1')
            conv1_3 = tf_utils.batch_norm(conv1_3, name='conv1_3_batch1', _ops=self._gen_train_ops)
            conv1_3 = tf.nn.relu(conv1_3, name='conv1_3_relu1')
            conv1_3 = tf_utils.conv2d(conv1_3, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_3_conv1_2')
            conv1_3 = tf_utils.batch_norm(conv1_3, name='conv1_3_batch2', _ops=self._gen_train_ops)
            conv1_3 = tf.nn.relu(conv1_3, name='conv1_3_relu2')
            pool3 = tf_utils.max_pool_2x2(conv1_3, name='maxpool3')

            # conv1_4: (N, 80, 80, 128) -> (N, 40, 40, 256)
            conv1_4 = tf_utils.conv2d(pool3, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_4_conv1_1')
            conv1_4 = tf_utils.batch_norm(conv1_4, name='conv1_4_batch1', _ops=self._gen_train_ops)
            conv1_4 = tf.nn.relu(conv1_4, name='conv1_4_relu1')
            conv1_4 = tf_utils.conv2d(conv1_4, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_4_conv1_2')
            conv1_4 = tf_utils.batch_norm(conv1_4, name='conv1_4_batch2', _ops=self._gen_train_ops)
            conv1_4 = tf.nn.relu(conv1_4, name='conv1_4_relu2')
            pool4 = tf_utils.max_pool_2x2(conv1_4, name='maxpool4')

            # conv1_5: (N, 40, 40, 256) -> (N, 40, 40, 512)
            conv1_5 = tf_utils.conv2d(pool4, 16 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_5_conv1_1')
            conv1_5 = tf_utils.batch_norm(conv1_5, name='conv1_5_batch1', _ops=self._gen_train_ops)
            conv1_5 = tf.nn.relu(conv1_5, name='conv1_5_relu1')
            conv1_5 = tf_utils.conv2d(conv1_5, 16 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_5_conv1_2')
            conv1_5 = tf_utils.batch_norm(conv1_5, name='conv1_5_batch2', _ops=self._gen_train_ops)
            conv1_5 = tf.nn.relu(conv1_5, name='conv1_5_relu2')

            # conv1_6: (N, 40, 40, 512) -> (N, 80, 80, 256)
            up1 = tf_utils.upsampling2d(conv1_5, size=(2, 2), name='conv1_6_up')
            conv1_6 = tf.concat([up1, conv1_4], axis=3, name='conv1_6_concat')
            conv1_6 = tf_utils.conv2d(conv1_6, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_6_conv1_1')
            conv1_6 = tf_utils.batch_norm(conv1_6, name='conv1_6_batch1', _ops=self._gen_train_ops)
            conv1_6 = tf.nn.relu(conv1_6, name='conv1_6_relu1')
            conv1_6 = tf_utils.conv2d(conv1_6, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_6_conv1_2')
            conv1_6 = tf_utils.batch_norm(conv1_6, name='conv1_6_batch2', _ops=self._gen_train_ops)
            conv1_6 = tf.nn.relu(conv1_6, name='conv1_6_relu2')

            # conv1_7: (N, 80, 80, 256) -> (N, 160, 160, 128)
            up2 = tf_utils.upsampling2d(conv1_6, size=(2, 2), name='conv1_7_up')
            conv1_7 = tf.concat([up2, conv1_3], axis=3, name='conv1_7_concat')
            conv1_7 = tf_utils.conv2d(conv1_7, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_7_conv1_1')
            conv1_7 = tf_utils.batch_norm(conv1_7, name='conv1_7_batch1', _ops=self._gen_train_ops)
            conv1_7 = tf.nn.relu(conv1_7, name='conv1_7_relu1')
            conv1_7 = tf_utils.conv2d(conv1_7, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_7_conv1_2')
            conv1_7 = tf_utils.batch_norm(conv1_7, name='conv1_7_batch2', _ops=self._gen_train_ops)
            conv1_7 = tf.nn.relu(conv1_7, name='conv1_7_relu2')

            # conv1_8: (N, 160, 160, 128) -> (N, 320, 320, 64)
            up3 = tf_utils.upsampling2d(conv1_7, size=(2, 2), name='conv1_8_up')
            conv1_8 = tf.concat([up3, conv1_2], axis=3, name='conv1_8_concat')
            conv1_8 = tf_utils.conv2d(conv1_8, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_8_conv1_1')
            conv1_8 = tf_utils.batch_norm(conv1_8, name='conv1_8_batch1', _ops=self._gen_train_ops)
            conv1_8 = tf.nn.relu(conv1_8, name='conv1_8_relu1')
            conv1_8 = tf_utils.conv2d(conv1_8, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_8_conv1_2')
            conv1_8 = tf_utils.batch_norm(conv1_8, name='conv1_8_batch2', _ops=self._gen_train_ops)
            conv1_8 = tf.nn.relu(conv1_8, name='conv1_8_relu2')

            # conv1_9: (N, 320, 320, 64) -> (N, 512, 512, 32)
            up4 = tf_utils.upsampling2d(conv1_8, size=(2, 2), name='conv1_9_up')
            conv1_9 = tf.concat([up4, conv1_1], axis=3, name='conv1_9_concat')
            conv1_9 = tf_utils.conv2d(conv1_9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_9_conv1_1')
            conv1_9 = tf_utils.batch_norm(conv1_9, name='conv1_9_batch1', _ops=self._gen_train_ops)
            conv1_9 = tf.nn.relu(conv1_9, name='conv1_9_relu1')
            conv1_9 = tf_utils.conv2d(conv1_9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_9_conv1_2')
            conv1_9 = tf_utils.batch_norm(conv1_9, name='conv1_9_batch2', _ops=self._gen_train_ops)
            conv1_9 = tf.nn.relu(conv1_9, name='conv1_9_relu2')


            # distance path
            # conv2_1: (N, 512, 512, 1) -> (N, 320, 320, 32)
            conv2_1 = tf_utils.conv2d(data2, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_1_conv2_1')
            conv2_1 = tf_utils.batch_norm(conv2_1, name='conv2_1_batch1', _ops=self._gen_train_ops)
            conv2_1 = tf.nn.relu(conv2_1, name='conv2_1_relu1')
            conv2_1 = tf_utils.conv2d(conv2_1, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_1_conv2_2')
            conv2_1 = tf_utils.batch_norm(conv2_1, name='conv2_1_batch2', _ops=self._gen_train_ops)
            conv2_1 = tf.nn.relu(conv2_1, name='conv2_1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv2_1, name='maxpool1')

            # conv2_2: (N, 320, 320, 32) -> (N, 160, 160, 64)
            conv2_2 = tf_utils.conv2d(pool1, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_2_conv2_1')
            conv2_2 = tf_utils.batch_norm(conv2_2, name='conv2_2_batch1', _ops=self._gen_train_ops)
            conv2_2 = tf.nn.relu(conv2_2, name='conv2_2_relu1')
            conv2_2 = tf_utils.conv2d(conv2_2, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_2_conv2_2')
            conv2_2 = tf_utils.batch_norm(conv2_2, name='conv2_2-batch2', _ops=self._gen_train_ops)
            conv2_2 = tf.nn.relu(conv2_2, name='conv2_2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv2_2, name='maxpool2')

            # conv2_3: (N, 160, 160, 64) -> (N, 80, 80, 128)
            conv2_3 = tf_utils.conv2d(pool2, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_3_conv2_1')
            conv2_3 = tf_utils.batch_norm(conv2_3, name='conv2_3_batch1', _ops=self._gen_train_ops)
            conv2_3 = tf.nn.relu(conv2_3, name='conv2_3_relu1')
            conv2_3 = tf_utils.conv2d(conv2_3, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_3_conv2_2')
            conv2_3 = tf_utils.batch_norm(conv2_3, name='conv2_3_batch2', _ops=self._gen_train_ops)
            conv2_3 = tf.nn.relu(conv2_3, name='conv2_3_relu2')
            pool3 = tf_utils.max_pool_2x2(conv2_3, name='maxpool3')

            # conv2_4: (N, 80, 80, 128) -> (N, 40, 40, 256)
            conv2_4 = tf_utils.conv2d(pool3, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_4_conv2_1')
            conv2_4 = tf_utils.batch_norm(conv2_4, name='conv2_4_batch1', _ops=self._gen_train_ops)
            conv2_4 = tf.nn.relu(conv2_4, name='conv2_4_relu1')
            conv2_4 = tf_utils.conv2d(conv2_4, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_4_conv2_2')
            conv2_4 = tf_utils.batch_norm(conv2_4, name='conv2_4_batch2', _ops=self._gen_train_ops)
            conv2_4 = tf.nn.relu(conv2_4, name='conv2_4_relu2')
            pool4 = tf_utils.max_pool_2x2(conv2_4, name='maxpool4')

            # conv2_5: (N, 40, 40, 256) -> (N, 40, 40, 512)
            conv2_5 = tf_utils.conv2d(pool4, 16 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_5_conv2_1')
            conv2_5 = tf_utils.batch_norm(conv2_5, name='conv2_5_batch1', _ops=self._gen_train_ops)
            conv2_5 = tf.nn.relu(conv2_5, name='conv2_5_relu1')
            conv2_5 = tf_utils.conv2d(conv2_5, 16 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_5_conv2_2')
            conv2_5 = tf_utils.batch_norm(conv2_5, name='conv2_5_batch2', _ops=self._gen_train_ops)
            conv2_5 = tf.nn.relu(conv2_5, name='conv2_5_relu2')

            # conv2_6: (N, 40, 40, 512) -> (N, 80, 80, 256)
            up1 = tf_utils.upsampling2d(conv2_5, size=(2, 2), name='conv2_6_up')
            conv2_6 = tf.concat([up1, conv2_4], axis=3, name='conv2_6_concat')
            conv2_6 = tf_utils.conv2d(conv2_6, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_6_conv2_1')
            conv2_6 = tf_utils.batch_norm(conv2_6, name='conv2_6_batch1', _ops=self._gen_train_ops)
            conv2_6 = tf.nn.relu(conv2_6, name='conv2_6_relu1')
            conv2_6 = tf_utils.conv2d(conv2_6, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_6_conv2_2')
            conv2_6 = tf_utils.batch_norm(conv2_6, name='conv2_6_batch2', _ops=self._gen_train_ops)
            conv2_6 = tf.nn.relu(conv2_6, name='conv2_6_relu2')

            # conv2_7: (N, 80, 80, 256) -> (N, 160, 160, 128)
            up2 = tf_utils.upsampling2d(conv2_6, size=(2, 2), name='conv2_7_up')
            conv2_7 = tf.concat([up2, conv2_3], axis=3, name='conv2_7_concat')
            conv2_7 = tf_utils.conv2d(conv2_7, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_7_conv2_1')
            conv2_7 = tf_utils.batch_norm(conv2_7, name='conv2_7_batch1', _ops=self._gen_train_ops)
            conv2_7 = tf.nn.relu(conv2_7, name='conv2_7_relu1')
            conv2_7 = tf_utils.conv2d(conv2_7, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_7_conv2_2')
            conv2_7 = tf_utils.batch_norm(conv2_7, name='conv2_7_batch2', _ops=self._gen_train_ops)
            conv2_7 = tf.nn.relu(conv2_7, name='conv2_7_relu2')

            # conv2_8: (N, 160, 160, 128) -> (N, 320, 320, 64)
            up3 = tf_utils.upsampling2d(conv2_7, size=(2, 2), name='conv2_8_up')
            conv2_8 = tf.concat([up3, conv2_2], axis=3, name='conv2_8_concat')
            conv2_8 = tf_utils.conv2d(conv2_8, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_8_conv2_1')
            conv2_8 = tf_utils.batch_norm(conv2_8, name='conv2_8_batch1', _ops=self._gen_train_ops)
            conv2_8 = tf.nn.relu(conv2_8, name='conv2_8_relu1')
            conv2_8 = tf_utils.conv2d(conv2_8, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_8_conv2_2')
            conv2_8 = tf_utils.batch_norm(conv2_8, name='conv2_8_batch2', _ops=self._gen_train_ops)
            conv2_8 = tf.nn.relu(conv2_8, name='conv2_8_relu2')

            # conv2_9: (N, 320, 320, 64) -> (N, 512, 512, 32)
            up4 = tf_utils.upsampling2d(conv2_8, size=(2, 2), name='conv2_9_up')
            conv2_9 = tf.concat([up4, conv2_1], axis=3, name='conv2_9_concat')
            conv2_9 = tf_utils.conv2d(conv2_9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_9_conv2_1')
            conv2_9 = tf_utils.batch_norm(conv2_9, name='conv2_9_batch1', _ops=self._gen_train_ops)
            conv2_9 = tf.nn.relu(conv2_9, name='conv2_9_relu1')
            conv2_9 = tf_utils.conv2d(conv2_9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_9_conv2_2')
            conv2_9 = tf_utils.batch_norm(conv2_9, name='conv2_9_batch2', _ops=self._gen_train_ops)
            conv2_9 = tf.nn.relu(conv2_9, name='conv2_9_relu2')

            # location path
            # conv3_1: (N, 512, 512, 1) -> (N, 320, 320, 32)
            conv3_1 = tf_utils.conv2d(data3, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_1_conv3_1')
            conv3_1 = tf_utils.batch_norm(conv3_1, name='conv3_1_batch1', _ops=self._gen_train_ops)
            conv3_1 = tf.nn.relu(conv3_1, name='conv3_1_relu1')
            conv3_1 = tf_utils.conv2d(conv3_1, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_1_conv3_2')
            conv3_1 = tf_utils.batch_norm(conv3_1, name='conv3_1_batch2', _ops=self._gen_train_ops)
            conv3_1 = tf.nn.relu(conv3_1, name='conv3_1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv3_1, name='maxpool1')

            # conv3_2: (N, 320, 320, 32) -> (N, 160, 160, 64)
            conv3_2 = tf_utils.conv2d(pool1, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_2_conv3_1')
            conv3_2 = tf_utils.batch_norm(conv3_2, name='conv3_2_batch1', _ops=self._gen_train_ops)
            conv3_2 = tf.nn.relu(conv3_2, name='conv3_2_relu1')
            conv3_2 = tf_utils.conv2d(conv3_2, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_2_conv3_2')
            conv3_2 = tf_utils.batch_norm(conv3_2, name='conv3_2-batch2', _ops=self._gen_train_ops)
            conv3_2 = tf.nn.relu(conv3_2, name='conv3_2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv3_2, name='maxpool2')

            # conv3_3: (N, 160, 160, 64) -> (N, 80, 80, 128)
            conv3_3 = tf_utils.conv2d(pool2, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_3_conv3_1')
            conv3_3 = tf_utils.batch_norm(conv3_3, name='conv3_3_batch1', _ops=self._gen_train_ops)
            conv3_3 = tf.nn.relu(conv3_3, name='conv3_3_relu1')
            conv3_3 = tf_utils.conv2d(conv3_3, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_3_conv3_2')
            conv3_3 = tf_utils.batch_norm(conv3_3, name='conv3_3_batch2', _ops=self._gen_train_ops)
            conv3_3 = tf.nn.relu(conv3_3, name='conv3_3_relu2')
            pool3 = tf_utils.max_pool_2x2(conv3_3, name='maxpool3')

            # conv3_4: (N, 80, 80, 128) -> (N, 40, 40, 256)
            conv3_4 = tf_utils.conv2d(pool3, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_4_conv3_1')
            conv3_4 = tf_utils.batch_norm(conv3_4, name='conv3_4_batch1', _ops=self._gen_train_ops)
            conv3_4 = tf.nn.relu(conv3_4, name='conv3_4_relu1')
            conv3_4 = tf_utils.conv2d(conv3_4, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_4_conv3_2')
            conv3_4 = tf_utils.batch_norm(conv3_4, name='conv3_4_batch2', _ops=self._gen_train_ops)
            conv3_4 = tf.nn.relu(conv3_4, name='conv3_4_relu2')
            pool4 = tf_utils.max_pool_2x2(conv3_4, name='maxpool4')

            # conv3_5: (N, 40, 40, 256) -> (N, 40, 40, 512)
            conv3_5 = tf_utils.conv2d(pool4, 16 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_5_conv3_1')
            conv3_5 = tf_utils.batch_norm(conv3_5, name='conv3_5_batch1', _ops=self._gen_train_ops)
            conv3_5 = tf.nn.relu(conv3_5, name='conv3_5_relu1')
            conv3_5 = tf_utils.conv2d(conv3_5, 16 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_5_conv3_2')
            conv3_5 = tf_utils.batch_norm(conv3_5, name='conv3_5_batch2', _ops=self._gen_train_ops)
            conv3_5 = tf.nn.relu(conv3_5, name='conv3_5_relu2')

            # conv3_6: (N, 40, 40, 512) -> (N, 80, 80, 256)
            up1 = tf_utils.upsampling2d(conv3_5, size=(2, 2), name='conv3_6_up')
            conv3_6 = tf.concat([up1, conv3_4], axis=3, name='conv3_6_concat')
            conv3_6 = tf_utils.conv2d(conv3_6, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_6_conv3_1')
            conv3_6 = tf_utils.batch_norm(conv3_6, name='conv3_6_batch1', _ops=self._gen_train_ops)
            conv3_6 = tf.nn.relu(conv3_6, name='conv3_6_relu1')
            conv3_6 = tf_utils.conv2d(conv3_6, 8 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_6_conv3_2')
            conv3_6 = tf_utils.batch_norm(conv3_6, name='conv3_6_batch2', _ops=self._gen_train_ops)
            conv3_6 = tf.nn.relu(conv3_6, name='conv3_6_relu2')

            # conv3_7: (N, 80, 80, 256) -> (N, 160, 160, 128)
            up2 = tf_utils.upsampling2d(conv3_6, size=(2, 2), name='conv3_7_up')
            conv3_7 = tf.concat([up2, conv3_3], axis=3, name='conv3_7_concat')
            conv3_7 = tf_utils.conv2d(conv3_7, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_7_conv3_1')
            conv3_7 = tf_utils.batch_norm(conv3_7, name='conv3_7_batch1', _ops=self._gen_train_ops)
            conv3_7 = tf.nn.relu(conv3_7, name='conv3_7_relu1')
            conv3_7 = tf_utils.conv2d(conv3_7, 4 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_7_conv3_2')
            conv3_7 = tf_utils.batch_norm(conv3_7, name='conv3_7_batch2', _ops=self._gen_train_ops)
            conv3_7 = tf.nn.relu(conv3_7, name='conv3_7_relu2')

            # conv3_8: (N, 160, 160, 128) -> (N, 320, 320, 64)
            up3 = tf_utils.upsampling2d(conv3_7, size=(2, 2), name='conv3_8_up')
            conv3_8 = tf.concat([up3, conv3_2], axis=3, name='conv3_8_concat')
            conv3_8 = tf_utils.conv2d(conv3_8, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_8_conv3_1')
            conv3_8 = tf_utils.batch_norm(conv3_8, name='conv3_8_batch1', _ops=self._gen_train_ops)
            conv3_8 = tf.nn.relu(conv3_8, name='conv3_8_relu1')
            conv3_8 = tf_utils.conv2d(conv3_8, 2 * self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_8_conv3_2')
            conv3_8 = tf_utils.batch_norm(conv3_8, name='conv3_8_batch2', _ops=self._gen_train_ops)
            conv3_8 = tf.nn.relu(conv3_8, name='conv3_8_relu2')

            # conv3_9: (N, 320, 320, 64) -> (N, 512, 512, 32)
            up4 = tf_utils.upsampling2d(conv3_8, size=(2, 2), name='conv3_9_up')
            conv3_9 = tf.concat([up4, conv3_1], axis=3, name='conv3_9_concat')
            conv3_9 = tf_utils.conv2d(conv3_9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_9_conv3_1')
            conv3_9 = tf_utils.batch_norm(conv3_9, name='conv3_9_batch1', _ops=self._gen_train_ops)
            conv3_9 = tf.nn.relu(conv3_9, name='conv3_9_relu1')
            conv3_9 = tf_utils.conv2d(conv3_9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_9_conv3_2')
            conv3_9 = tf_utils.batch_norm(conv3_9, name='conv3_9_batch2', _ops=self._gen_train_ops)
            conv3_9 = tf.nn.relu(conv3_9, name='conv3_9_relu2')




            # intensity path
            # conv1: (N, 512, 512, 1) -> (N, 320, 320, 32)
            conv1 = tf_utils.conv2d(data, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv1')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch1', _ops=self._gen_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')
            conv1 = tf_utils.conv2d(conv1, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch2', _ops=self._gen_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')
            conv1 = tf.concat([conv1, conv1_1, conv2_1, conv3_1], axis=3, name='conv1_concat')

            pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')

            # conv2: (N, 320, 320, 32) -> (N, 160, 160, 64)
            conv2 = tf_utils.conv2d(pool1, 2*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv1')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1', _ops=self._gen_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            conv2 = tf_utils.conv2d(conv2, 2*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
            conv2 = tf_utils.batch_norm(conv2, name='conv2-batch2', _ops=self._gen_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            conv2 = tf.concat([conv2, conv1_2, conv2_2, conv3_2], axis=3, name='conv2_concat')
            pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')

            # conv3: (N, 160, 160, 64) -> (N, 80, 80, 128)
            conv3 = tf_utils.conv2d(pool2, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1', _ops=self._gen_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = tf_utils.conv2d(conv3, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch2', _ops=self._gen_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')
            conv3 = tf.concat([conv3, conv1_3, conv2_3, conv3_3], axis=3, name='conv3_concat')
            pool3 = tf_utils.max_pool_2x2(conv3, name='maxpool3')

            # conv4: (N, 80, 80, 128) -> (N, 40, 40, 256)
            conv4 = tf_utils.conv2d(pool3, 8*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv1')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch1', _ops=self._gen_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu1')
            conv4 = tf_utils.conv2d(conv4, 8*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv2')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch2', _ops=self._gen_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu2')
            conv4 = tf.concat([conv4, conv1_4, conv2_4, conv3_4], axis=3, name='conv4_concat')
            pool4 = tf_utils.max_pool_2x2(conv4, name='maxpool4')

            # conv5: (N, 40, 40, 256) -> (N, 40, 40, 512)
            conv5 = tf_utils.conv2d(pool4, 16*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv1')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch1', _ops=self._gen_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu1')
            conv5 = tf_utils.conv2d(conv5, 16*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv2')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch2', _ops=self._gen_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu2')


            # conv6: (N, 40, 40, 512) -> (N, 80, 80, 256)
            up1 = tf_utils.upsampling2d(conv5, size=(2, 2), name='conv6_up')
            conv6 = tf.concat([up1, conv4], axis=3, name='conv6_concat')
            conv6 = tf_utils.conv2d(conv6, 8*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_conv1')
            conv6 = tf_utils.batch_norm(conv6, name='conv6_batch1', _ops=self._gen_train_ops)
            conv6 = tf.nn.relu(conv6, name='conv6_relu1')
            conv6 = tf_utils.conv2d(conv6, 8*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_conv2')
            conv6 = tf_utils.batch_norm(conv6, name='conv6_batch2', _ops=self._gen_train_ops)
            conv6 = tf.nn.relu(conv6, name='conv6_relu2')
            conv6 = tf.concat([conv6, conv1_6, conv2_6, conv3_6], axis=3, name='conv6_concat')


            # conv7: (N, 80, 80, 256) -> (N, 160, 160, 128)
            up2 = tf_utils.upsampling2d(conv6, size=(2, 2), name='conv7_up')
            conv7 = tf.concat([up2, conv3], axis=3, name='conv7_concat')
            conv7 = tf_utils.conv2d(conv7, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv7_conv1')
            conv7 = tf_utils.batch_norm(conv7, name='conv7_batch1', _ops=self._gen_train_ops)
            conv7 = tf.nn.relu(conv7, name='conv7_relu1')
            conv7 = tf_utils.conv2d(conv7, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv7_conv2')
            conv7 = tf_utils.batch_norm(conv7, name='conv7_batch2', _ops=self._gen_train_ops)
            conv7 = tf.nn.relu(conv7, name='conv7_relu2')
            conv7 = tf.concat([conv7, conv1_7, conv2_7, conv3_7], axis=3, name='conv7_concat')

            # conv8: (N, 160, 160, 128) -> (N, 320, 320, 64)
            up3 = tf_utils.upsampling2d(conv7, size=(2, 2), name='conv8_up')
            conv8 = tf.concat([up3, conv2], axis=3, name='conv8_concat')
            conv8 = tf_utils.conv2d(conv8, 2*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv8_conv1')
            conv8 = tf_utils.batch_norm(conv8, name='conv8_batch1', _ops=self._gen_train_ops)
            conv8 = tf.nn.relu(conv8, name='conv8_relu1')
            conv8 = tf_utils.conv2d(conv8, 2*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv8_conv2')
            conv8 = tf_utils.batch_norm(conv8, name='conv8_batch2', _ops=self._gen_train_ops)
            conv8 = tf.nn.relu(conv8, name='conv8_relu2')
            conv8 = tf.concat([conv8, conv1_8, conv2_8, conv3_8], axis=3, name='conv8_concat')

            # conv9: (N, 320, 320, 64) -> (N, 512, 512, 32)
            up4 = tf_utils.upsampling2d(conv8, size=(2, 2), name='conv9_up')
            conv9 = tf.concat([up4, conv1], axis=3, name='conv9_concat')
            conv9 = tf_utils.conv2d(conv9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv9_conv1')
            conv9 = tf_utils.batch_norm(conv9, name='conv9_batch1', _ops=self._gen_train_ops)
            conv9 = tf.nn.relu(conv9, name='conv9_relu1')
            conv9 = tf_utils.conv2d(conv9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv9_conv2')
            conv9 = tf_utils.batch_norm(conv9, name='conv9_batch2', _ops=self._gen_train_ops)
            conv9 = tf.nn.relu(conv9, name='conv9_relu2')
            conv9 = tf.concat([conv9, conv1_9, conv2_9, conv3_9], axis=3, name='conv1_concat')

            # output layer: (N, 512, 512, 32) -> (N, 512, 512, 1)
            output = tf_utils.conv2d(conv9, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

            return tf.nn.sigmoid(output)

    def discriminator(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # conv1: (N, 512, 512, 4) -> (N,, 512, 512, 32)
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv1')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu1')

            # conv2: (N, 512, 512, 32) -> (N, 512, 512, 64)
            conv2 = tf_utils.conv2d(conv1, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv1')
            conv2 = tf_utils.lrelu(conv2)

            # conv3: (N, 512, 512, 64) -> (N, 512, 512, 128)
            conv3 = tf_utils.conv2d(conv2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.lrelu(conv3)

            # conv4: (N, 512, 512, 128) -> (N, 512, 512, 256)
            conv4 = tf_utils.conv2d(conv3, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv1')
            conv4 = tf_utils.lrelu(conv4)

            # conv5: (N, 512, 512, 256) -> (N, 512, 512, 512)
            conv5 = tf_utils.conv2d(conv4, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv1')
            conv5 = tf_utils.lrelu(conv5)

            # conv6: (N, 512, 512, 512) -> (N, 512, 512, 1024)
            conv6 = tf_utils.conv2d(conv5, 32*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_conv1')
            conv6 = tf_utils.lrelu(conv6)


            # output layer: (N, 512, 512, 1024) -> (N, 512, 512, 1)
            output = tf_utils.conv2d(conv3, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

            return tf.nn.sigmoid(output), output


    def train_dis(self, x_data, y_data):
        feed_dict = {self.X: x_data, self.Y: y_data}
        # run discriminator
        _, d_loss = self.sess.run([self.dis_optim, self.d_loss], feed_dict=feed_dict)

        return d_loss

    def train_gen(self, x_data, y_data):
        x_data0 = x_data[:,:,:,0]
        x_data0 = np.expand_dims(x_data0, axis=3)
        x_data1 = x_data[:,:,:,1]
        x_data1 = np.expand_dims(x_data1, axis=3)
        x_data2 = x_data[:,:,:,2]
        x_data2 = np.expand_dims(x_data2, axis=3)
        x_data3 = x_data[:,:,:,3]
        x_data3 = np.expand_dims(x_data3, axis=3)
        feed_dict = {self.X: x_data0, self.X1: x_data1,self.X2: x_data2,self.X3: x_data3,self.Y: y_data}
        # feed_dict = {self.X: x_data, self.Y: y_data}
        # run generator
        _, g_loss = self.sess.run([self.gen_optim, self.g_loss], feed_dict=feed_dict)

        return g_loss

    def measure_assign(self, auc_pr, auc_roc, dice_coeff, acc, sensitivity, specificity, score, iter_time):
        feed_dict = {self.auc_pr_placeholder: auc_pr,
                     self.auc_roc_placeholder: auc_roc,
                     self.dice_coeff_placeholder: dice_coeff,
                     self.acc_placeholder: acc,
                     self.sensitivity_placeholder: sensitivity,
                     self.specificity_placeholder: specificity,
                     self.score_placeholder: score}

        self.sess.run(self.measure_assign_op, feed_dict=feed_dict)

        summary = self.sess.run(self.measure_summary)
        self.writer.add_summary(summary, iter_time)

    def best_auc_sum_assign(self, auc_sum):
        self.sess.run(self.best_auc_sum_assign_op, feed_dict={self.best_auc_sum_placeholder: auc_sum})

    def sample_imgs(self, x_data):
        x_data0 = x_data[:,:,:,0]
        x_data0 = np.expand_dims(x_data0, axis=3)
        x_data1 = x_data[:,:,:,1]
        x_data1 = np.expand_dims(x_data1, axis=3)
        x_data2 = x_data[:,:,:,2]
        x_data2 = np.expand_dims(x_data2, axis=3)
        x_data3 = x_data[:,:,:,3]
        x_data3 = np.expand_dims(x_data3, axis=3)

        return self.sess.run(self.g_samples, feed_dict={self.X: x_data0, self.X1: x_data1,self.X2: x_data2,self.X3: x_data3})

    def measure_loss(self, gloss, dloss, iter_time):
        feed_dict = {self.gloss_save_placeholder: gloss,
                     self.dloss_save_placeholder: dloss}

        self.sess.run(self.measure_loss_op, feed_dict=feed_dict)
        summary = self.sess.run(self.measure_loss_summary)
        self.writer.add_summary(summary, iter_time)
