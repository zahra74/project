import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from dataset3D import Train_dataset
from utils import subPixelConv3d
import math
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
import os
import skimage
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import numpy as np
tf.reset_default_graph()
import matplotlib.pyplot as plt
import nibabel as nib
import argparse
from math import log10, sqrt


path_prediction_Gen = 'select your path'
path_prediction_Gan='select your path'
path_volumes='select your path'
checkpoint_dir='select your path'
checkpoint_dir_restore='select your path'
saveoutput=open('','w')

tl.files.exists_or_mkdir(path_prediction_Gen)
tl.files.exists_or_mkdir(path_prediction_Gan)
tl.files.exists_or_mkdir(path_volumes)
tl.files.exists_or_mkdir(checkpoint_dir)
tl.files.exists_or_mkdir(checkpoint_dir_restore)

lr_init=0.0001

def lrelu1(x):
    return tf.maximum(x, 0.25 * x)


def lrelu2(x):
    return tf.maximum(x, 0.3 * x)

def sigmoid(z):
    s =1/(1+(np.exp(-z)))
    return s


def discriminator(input_disc, kernel, reuse, is_train=True):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    batch_size = 1
    img_width = 80
    img_height = 80
    img_depth = 80
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        input_disc.set_shape([1, img_width, img_height, img_depth, 1], )
        x = InputLayer(input_disc, name='in')
        print("++++++++++ ++++++++++ ++++++++++ DIS CONV input_disc: ",input_disc)
        print("++++++++++ ++++++++++ 1")
        net_h0 = Conv3dLayer(x, act=lrelu2, shape=[4, 4, 4, 1, 64], strides=[1, 2, 2, 2, 1],padding='SAME', W_init=w_init, name='conv1')
        print('***x1***',net_h0)
        print("++++++++++ ++++++++++ 2")
        net_h1 = Conv3dLayer(net_h0, shape=[4, 4, 4, 64, 128], strides=[1, 2, 2, 2, 1],padding='SAME', W_init=w_init, name='conv2')
        net_h1 = BatchNormLayer(net_h1, is_train=is_train,gamma_init=gamma_init, name='BN1-conv2', act=lrelu2)
        print('***x2***',net_h1)
        print("++++++++++ ++++++++++ 3")
        net_h2 = Conv3dLayer(net_h1, shape=[4, 4, 4, 128, 256], strides=[1, 2, 2, 2, 1],padding='SAME', W_init=w_init, name='conv3')
        net_h2 = BatchNormLayer(net_h2, is_train=is_train,gamma_init=gamma_init, name='BN1-conv3', act=lrelu2)
        print('***x3***',net_h2)
        print("++++++++++ ++++++++++ 4")
        net_h3 = Conv3dLayer(net_h2, shape=[4, 4, 4, 256, 512], strides=[1, 2, 2, 2, 1],padding='SAME', W_init=w_init, name='conv4')
        net_h3 = BatchNormLayer(net_h3, is_train=is_train,gamma_init=gamma_init, name='BN1-conv4', act=lrelu2)
        print('***x4***',net_h3)
        print("++++++++++ ++++++++++ 5")
        net_h4 = Conv3dLayer(net_h3, shape=[4, 4, 4, 512, 1024], strides=[1, 2, 2, 2, 1],padding='SAME', W_init=w_init, name='conv5')
        net_h4 = BatchNormLayer(net_h4, is_train=is_train,gamma_init=gamma_init, name='BN1-conv5', act=lrelu2)
        print('***x5***',net_h4)
        print("++++++++++ ++++++++++ 6")
        net_h5 = Conv3dLayer(net_h4, shape=[4, 4, 4, 1024, 2048], strides=[1, 2, 2, 2, 1],padding='SAME', W_init=w_init, name='conv6')
        net_h5 = BatchNormLayer(net_h5, is_train=is_train,gamma_init=gamma_init, name='BN1-conv6', act=lrelu2)
        print('***x6***',net_h5)
        print("++++++++++ ++++++++++ 7")
        net_h6 = Conv3dLayer(net_h5, shape=[1, 1, 1, 2048, 1024], strides=[1, 1, 1, 1, 1],padding='SAME', W_init=w_init, name='conv7')
        net_h6 = BatchNormLayer(net_h6, is_train=is_train,gamma_init=gamma_init, name='BN1-conv7', act=lrelu2)
        print('***x7***',net_h6)
        print("++++++++++ ++++++++++ 8")
        net_h7 = Conv3dLayer(net_h6, shape=[1, 1, 1, 1024, 512], strides=[1, 1, 1, 1, 1],padding='SAME', W_init=w_init, name='conv8')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train,gamma_init=gamma_init, name='BN1-conv8', act=lrelu2)
        print('***x8***',net_h7)
        print("++++++++++ ++++++++++ 9")
        net= Conv3dLayer(net_h7, shape=[1, 1, 1, 512, 128], strides=[1, 1, 1, 1, 1],padding='SAME', W_init=w_init, name='res')
        net = BatchNormLayer(net, is_train=is_train,gamma_init=gamma_init, name='BN1-res', act=lrelu2)
        print('***x9***',net)
        print("++++++++++ ++++++++++ 10")
        net= Conv3dLayer(net, shape=[3, 3, 3, 128, 128], strides=[1, 1, 1, 1, 1],padding='SAME', W_init=w_init, name='res2')
        net= BatchNormLayer(net, is_train=is_train,gamma_init=gamma_init, name='BN1-res2', act=lrelu2)
        print('***x10***',net)
        print("++++++++++ ++++++++++ 11")
        net= Conv3dLayer(net, shape=[3, 3, 3, 128, 512], strides=[1, 1, 1, 1, 1],padding='SAME', W_init=w_init, name='res3')
        net= BatchNormLayer(net, is_train=is_train,gamma_init=gamma_init, name='BN1-res3', act=lrelu2)
        print('***x11***',net)
        print("++++++++++ ++++++++++ FINAL DISCRIMINATOR")
        net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='add')
        net_h8.outputs = lrelu2(net_h8.outputs)
        print('net_h8.outputs',net_h8.outputs)
        
        net_ho = FlattenLayer(net_h8, name='flatten')
        print('FlattenLayer',net_ho.outputs)
        x = DenseLayer(net_ho, n_units=1,act=tf.identity,W_init=w_init, name='dense')
        print('***x11***',x)
        logits = x.outputs
        print("++++++++++ logits ",logits)  
        print('              ****DESCRIMINATOR FINISH****                    ')
        return x, logits

def generator(input_gen, kernel, nb, upscaling_factor, reuse, feature_size, img_width, img_height, img_depth,
              subpixel_NN, nn, is_train=True):
    w_init = tf.random_normal_initializer(stddev=0.02)
    print(" input_gen ------<<<<<< ",input_gen)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        x = InputLayer(input_gen, name='in')
        print("++++++++++ ++++++++++ ++++++++++ GEN 1 CONV ++++++++++ ++++++++++ feature_size: ",feature_size)
        print(" input_gen-------------->>>>>>>> ",input_gen)
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 1, feature_size*2], strides=[1, 1, 1, 1, 1],padding='SAME', W_init=w_init, name='conv1')
        x = BatchNormLayer(x, act=lrelu1, is_train=is_train,gamma_init=g_init, name='BN-conv1')
        print('%%%%GEN-x1',x)
        inputRB = x
        inputadd = x
        print(" RRRRRRRBBBBBBBBBB   inputRB: ",inputRB.inputs)
        print(" IIINNNNNPPPPPUUTT   inputadd: ",inputadd.inputs)
        print(" nb ",nb)
        # residual blocks
        for i in range(nb):
            print('i =',i)
            print("residual blocks ++++++++++ ++++++++++ feature_size: ",feature_size*2)
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size*2, feature_size*2], strides=[1, 1, 1, 1, 1],padding='SAME', W_init=w_init, name='conv1-rb/%s' % i)
            x = BatchNormLayer(x, act=lrelu1, is_train=is_train,gamma_init=g_init, name='BN1-rb/%s' % i)

            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size*2, feature_size*2], strides=[1, 1, 1, 1, 1],padding='SAME', W_init=w_init, name='conv2-rb/%s' % i)
            #x = BatchNormLayer(x, is_train=is_train, name='BN2-rb/%s' % i, ) # coma
            x = BatchNormLayer(x, is_train=is_train,gamma_init=g_init, name='BN2-rb/%s' % i)
            # short skip connection
            x = ElementwiseLayer([inputadd,x], tf.add, name='add-rb/%s' % i)
            inputadd = x
            print('%%%%GEN-x2',x)

        # large skip connection
        print("++++++++++ ++++++++++ ++++++++++ ++++++++++ ++++++++++ ++++++++++ feature_size: ",feature_size*2)
        print(" ",kernel,feature_size*2)
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size*2, feature_size*2], strides=[1, 1, 1, 1, 1],padding='SAME', W_init=w_init, name='conv2')
        x = BatchNormLayer(x, is_train=is_train,gamma_init=g_init, name='BN-conv2')
        x = ElementwiseLayer([x, inputRB], tf.add, name='add-conv2')
        print('%%%%GEN-x3',x)
        x = ElementwiseLayer([x, inputRB], tf.add, name='ERROR_add-conv2')
        print('%%%%GEN-x4',x)
        print('           ****GENERATOR FINISH****                           ')
        # ____________SUBPIXEL-NN______________#
        if subpixel_NN:
            # upscaling block 1
            print("UUUUUUUUUUUUUUUU SUBPIXEL-NN   upscaling block 1  ")
            print(" ",upscaling_factor)
            if upscaling_factor == 4:
                img_height_deconv = int(img_height / 2)
                img_width_deconv = int(img_width / 2)
                img_depth_deconv = int(img_depth / 2)
            else:
                img_height_deconv = img_height
                img_width_deconv = img_width
                img_depth_deconv = img_depth
            print("-- tf.shape(input_gen)[0], -- upscaling_factor: ", tf.shape(input_gen)[0], upscaling_factor)
            print("u4 img_height_deconv, img_width_deconv, img_depth_deconv: ", img_height_deconv,img_width_deconv,img_depth_deconv)
            print("-- img_height, img_width, img_depth: ", img_height,img_width,img_depth)
            print(" cccccc SUBPIXEL-NN ub1A  ")
            print(" ",kernel, feature_size)
            x = DeConv3dLayer(x, shape=[kernel * 2, kernel * 2, kernel * 2, 64, feature_size],
                              act=lrelu1, strides=[1, 2, 2, 2, 1],
                              output_shape=[tf.shape(input_gen)[0], img_height_deconv, img_width_deconv,
                                            img_depth_deconv, 64],
                              #padding='SAME', W_init=w_init_subpixel1_last, name='conv1-ub-subpixelnn/1')
                  padding='SAME', W_init=w_init_subpixel1_last, name='ERROR/1')
            print("uab                                                                 ")
        # upscaling block 2
            print("UUUUUUUUUUUUUUUU SUBPIXEL-NN  upscaling block 2  ")
            if upscaling_factor == 4:
                print(" cccccc SUBPIXEL-NN ub2A  ")
                x = DeConv3dLayer(x, shape=[kernel * 2, kernel * 2, kernel * 2, 64, 64],
                                  act=lrelu1, strides=[1, 2, 2, 2, 1],
                                  output_shape=[tf.shape(input_gen)[0], img_height, img_width,
                                                img_depth, 64],
                                  padding='SAME', W_init=w_init_subpixel2_last, name='conv1-ub-subpixelnn/2')

            print(" cccccc SUBPIXEL-NN FINAL  ")
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 64, 1], strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='convlast-subpixelnn')

        # ____________RC______________#

        elif nn:
            # upscaling block 1
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size, 64], act=lrelu1,
                            strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv1-ub/1')
            x = UpSampling3D(name='UpSampling3D_1')(x.outputs)
            x = Conv3dLayer(InputLayer(x, name='in ub1 conv2'),
                            shape=[kernel, kernel, kernel, 64, 64],
                            act=lrelu1,
                            strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv2-ub/1')

            # upscaling block 2
            if upscaling_factor == 4:
                x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 64, 64], act=lrelu1,
                                strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='conv1-ub/2')
                x = UpSampling3D(name='UpSampling3D_1')(x.outputs)
                x = Conv3dLayer(InputLayer(x, name='in ub2 conv2'), 
                shape=[kernel, kernel, kernel, 64, 64], 
                act=lrelu1,
                                strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='conv2-ub/2')

            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 64, 1], strides=[1, 1, 1, 1, 1],
                            act=tf.nn.tanh, padding='SAME', W_init=w_init, name='convlast')

        # ____________SUBPIXEL - BASELINE______________#

        else:

            if upscaling_factor == 4:
                steps_to_end = 2
            else:
                steps_to_end = 1

            # upscaling block 1
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size*2, feature_size*4], act=lrelu1,strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv1-ub/1')
            print('%%%%GEN-x5',x)
            arguments = {'img_width': img_width, 'img_height': img_height, 'img_depth': img_depth,'stepsToEnd': steps_to_end,
                         'n_out_channel': int(128 / 8)}
            x = LambdaLayer(x, fn=subPixelConv3d, fn_args=arguments, name='SubPixel1')
            print('%%%%GEN-x6',x)

            # upscaling block 2
            if upscaling_factor == 4:
                x = Conv3dLayer(x, shape=[kernel, kernel, kernel, int((128) / 8), 128], act=lrelu1,
                                strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='conv1-ub/2')
                print('%%%%GEN-x7',x)
                arguments = {'img_width': img_width, 'img_height': img_height, 'img_depth': img_depth, 'stepsToEnd': 1,
                             'n_out_channel': int(128 / 8)}
                x = LambdaLayer(x, fn=subPixelConv3d, fn_args=arguments, name='SubPixel2')
                print('%%%%GEN-x8',x)

            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, int(128 / 8), 1],act=tf.nn.tanh, strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='convlast')
            print('%%%%GEN-x9',x)
        return x

print('*****4******')
def train(upscaling_factor, residual_blocks, feature_size, path_prediction,path_prediction_Gan, checkpoint_dir, img_width, img_height,
          img_depth, subpixel_NN, nn, restore, batch_size=1, div_patches=4, epochs=13):
    print('**************train satart********************')
    traindataset = Train_dataset(batch_size)
    print('traindataset',traindataset)
    iterations_train=751
    print("-ooo2--> len(traindataset.subject_list): ",len(traindataset.subject_list))    
    # ##========================== DEFINE MODEL ============================##
    t_input_gen = tf.placeholder('float32', [1, None,None, None, 1],name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [1,img_width, img_height, img_depth, 1],name='t_target_image')

    print("sssssssssssssssssssss   input_gen: ",t_input_gen)
    print("xxxxxxxxxxxxxxxxxxxxx   input_disc: ",t_target_image)

    net_gen = generator(input_gen=t_input_gen, kernel=3, nb=residual_blocks, upscaling_factor=upscaling_factor,
                        reuse=False,feature_size=feature_size, img_width=img_width,img_height=img_height, img_depth=img_depth,
                        subpixel_NN=subpixel_NN, nn=nn, is_train=True, )

    net_d, disc_out_real = discriminator(input_disc=t_target_image, kernel=3, reuse=False, is_train=True)
    print('net_d',net_d)
    print('disc_out_real',disc_out_real)

    _, disc_out_fake = discriminator(input_disc=net_gen.outputs, kernel=3, reuse=True, is_train=True)

    # test
    gen_test = generator(t_input_gen, kernel=3, nb=residual_blocks, upscaling_factor=upscaling_factor,
                         reuse=True,feature_size=feature_size,img_width=img_width, img_height=img_height,  img_depth=img_depth,
                         subpixel_NN=subpixel_NN,nn=nn,is_train=False)
    # ###========================== DEFINE TRAIN OPS ==========================###

    d_loss_real = tl.cost.sigmoid_cross_entropy(disc_out_real, tf.ones_like(disc_out_real), name='d_loss_real')
    print('d_loss_real',d_loss_real)
    
    d_loss_fake = tl.cost.sigmoid_cross_entropy(disc_out_fake, tf.zeros_like(disc_out_real), name='d_loss_fake')
    print('d_loss_fake',d_loss_fake)
    
    d_loss = d_loss_real + d_loss_fake
    print('d_loss',d_loss)

    g_gan_loss = 1e-1 * tl.cost.sigmoid_cross_entropy(disc_out_fake, tf.ones_like(disc_out_fake), name='g_loss_gan')
    print('g_gan_loss',g_gan_loss)

    print(' t_target_image', t_target_image)
    print('net_gen.outputs',net_gen.outputs)
    target=t_target_image[:,:,:,:,0]
    produced=net_gen.outputs[:,:,:,:,0]
    print(' t_target_image', target)
    print('net_gen.outputs',produced)
    mse_loss = tl.cost.mean_squared_error(produced, target, is_mean=True,name='g_loss_mse')
    print('mse_loss',mse_loss)
    
    
    dx_real = t_target_image[:, 1:, :, :, :] - t_target_image[:, :-1, :, :, :]
    dy_real = t_target_image[:, :, 1:, :, :] - t_target_image[:, :, :-1, :, :]
    dz_real = t_target_image[:, :, :, 1:, :] - t_target_image[:, :, :, :-1, :]
    dx_fake = net_gen.outputs[:, 1:, :, :, :] - net_gen.outputs[:, :-1, :, :, :]
    dy_fake = net_gen.outputs[:, :, 1:, :, :] - net_gen.outputs[:, :, :-1, :, :]
    dz_fake = net_gen.outputs[:, :, :, 1:, :] - net_gen.outputs[:, :, :, :-1, :]

    gd_loss = 0.0001*(tf.reduce_sum(tf.square(tf.abs(dx_real) - tf.abs(dx_fake))) + \
              tf.reduce_sum(tf.square(tf.abs(dy_real) - tf.abs(dy_fake))) + \
              tf.reduce_sum(tf.square(tf.abs(dz_real) - tf.abs(dz_fake))))

    
    g_loss = mse_loss + g_gan_loss + gd_loss
    print('g_loss',g_loss)

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)
    print('**var finished')

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
        
    beta1=0.9
    g_optim_init = tf.train.AdamOptimizer(lr_v,beta1=beta1).minimize(mse_loss, var_list=g_vars)
    g_optim = tf.train.AdamOptimizer(lr_v,beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v,beta1=beta1).minimize(d_loss, var_list=d_vars)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    step = 0
    saver = tf.train.Saver()

    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format('srgan'), network=net_gen) is False:
        print('false')
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format('srgan'), network=net_gen)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format('srgan'), network=net_d)
    
    val_restore=0
    n_epoch_init=101
    #for epoch in range(val_restore, 2):
    for epoch in range(val_restore, n_epoch_init):
        sess.run(tf.assign(lr_v, lr_init))
        saveoutput.write('learning_rate_gen is')
        saveoutput.write(str(sess.run(lr_v))+ '\n')
        total_mse_loss=0
        n_iter = 0
        #for i in range(0,  2):
        for i in range(0,  iterations_train):
            # ====================== LOAD DATA =========================== #
            xt_total = traindataset.data_true(i)

                # NORMALIZING
            for t in range(0, xt_total .shape[0]):
                
                normfactor = (np.amax(xt_total [t])) / 2
                if normfactor != 0:
                    xt_total [t] = ((xt_total [t] - normfactor) / normfactor)
                   
                x_generator = gaussian_filter(xt_total , sigma=1)
                x_generator = zoom(x_generator, [1.0, (float(1.0 / upscaling_factor)), float(1.0 / upscaling_factor),
                                                 float(1.0 / upscaling_factor), 1], prefilter=False, order=0)
                xgenin = x_generator
                #print('xgenin.shape',xgenin.shape)
                
                # ========================= train SRGAN ========================= #
                #print('%%%%%%%%%%%%%%%%%%update Generator')
                errM, _ = sess.run([mse_loss, g_optim_init], {t_target_image: xt_total, t_input_gen: xgenin})
                print("Epoch [%2d/%2d] [%4d/%4d]: mse: %.6f " % (epoch+1, n_epoch_init , i+1, iterations_train, errM))
                n_iter += 1
                total_mse_loss += errM
            if epoch % 10 == 0 and i != 0 and i % 750 == 0:
                #if i != 0 and i % 1 == 0:
                    if epoch - 0 == 0:
                        x_true_img = xt_total[0]
                        if normfactor != 0:
                            x_true_img = ((x_true_img + 1) * normfactor)  # denormalize
                        img_true = nib.Nifti1Image(x_true_img, np.eye(4))
                        img_true.to_filename(os.path.join(path_prediction_Gen, str(epoch) + str(i) + 'true.nii.gz'))
                        
                        x_gen_img = xgenin[0]
                        if normfactor != 0:
                            x_gen_img = ((x_gen_img + 1) * normfactor)  # denormalize
                        img_gen = nib.Nifti1Image(x_gen_img, np.eye(4))
                        img_gen.to_filename(os.path.join(path_prediction_Gen, str(epoch) + str(i) + 'gen.nii.gz'))
                    
                    x_pred = sess.run(gen_test.outputs, {t_input_gen: xgenin})
                    x_pred_img = x_pred[0]
                    if normfactor != 0:
                        x_pred_img = ((x_pred_img + 1) * normfactor)  # denormalize
                    img_pred = nib.Nifti1Image(x_pred_img, np.eye(4))
                    img_pred.to_filename(os.path.join(path_prediction_Gen, str(epoch) + str(i) + 'prod.nii.gz'))
        log = "[*] Epoch: [%2d/%2d], mse: %.8f" % (epoch+1, n_epoch_init, total_mse_loss / n_iter)
        print('log is',log)
        saveoutput.write('log is')
        +saveoutput.write(str(log) + '\n')
        
        ## save model
        #if (epoch != 0) and (epoch % 5 == 0):
        if (epoch % 10 == 0):
            tl.files.save_npz(net_gen.all_params, name=checkpoint_dir+ '/g_{}_init.npz'.format('srgan') , sess=sess)
    print('************FINISH generator****************')
    count=range(iterations_train)
    #count=range(2)
    loss_Dis=list()
    loss_Gen=list()
    for j in range(val_restore, epochs):
    #for j in range(val_restore, 1):
        #print('j is=',j)
        total_d_loss=0
        total_g_loss=0
        n_iter=0
        saveoutput.write('learning_rate_gan is')
        saveoutput.write(str(sess.run(lr_v))+ '\n')
        #list_real=np.zeros((2,2))
        list_real=np.zeros((2,iterations_train))
        #list_fake=np.zeros((2,2))
        list_fake=np.zeros((2,iterations_train))
        for i in range(0,  iterations_train):
        #for i in range(0, 2):
            # ====================== LOAD DATA =========================== #
            xt_total = traindataset.data_true(i)
                # NORMALIZING
            for t in range(0, xt_total.shape[0]):
            
                normfactor = (np.amax(xt_total[t])) / 2
                if normfactor != 0:
                    xt_total[t] = ((xt_total[t] - normfactor) / normfactor)
              
                x_generator = gaussian_filter(xt_total, sigma=1)
                x_generator = zoom(x_generator, [1.0, (float(1.0 / upscaling_factor)), float(1.0 / upscaling_factor),
                                                 float(1.0 / upscaling_factor), 1], prefilter=False, order=0)
                xgenin = x_generator
                #print('xgenin is',xgenin.shape)
                
                # ========================= train SRGAN ========================= #
                errD, _ = sess.run([d_loss, d_optim], {t_target_image: xt_total, t_input_gen: xgenin})

                errG, errM, errgan, errgd, _ = sess.run([g_loss, mse_loss, g_gan_loss, gd_loss, g_optim],
                                                             {t_input_gen: xgenin, t_target_image: xt_total})

                print(
                    "Epoch [%2d/%2d] [%4d/%4d]: d_loss: %.8f g_loss: %.8f (mse: %.6f gdl: %.6f adv_Gen: %.6f)" % (
                        j+1, epochs , i+1, iterations_train, errD, errG, errM, errgd,errgan))
                total_d_loss += errD
                total_g_loss += errG
                n_iter += 1
                #if i != 0 and i % 1 == 0:
                if i != 0 and i % 750 == 0:
                    x_pred_f = sess.run(gen_test.outputs, {t_input_gen: xgenin})
                    x_pred_img = x_pred_f[0]
                    if normfactor != 0:
                        x_pred_img = ((x_pred_img + 1) * normfactor)  # denormalize
                    img_pred = nib.Nifti1Image(x_pred_img, np.eye(4))
                    img_pred.to_filename(os.path.join(path_prediction_Gan, str(j) + str(i) + '.nii.gz'))    
                
                if j % 1 == 0:
                #if j % 5 == 0 and k==1:
                    output_real= sess.run([disc_out_real], {t_target_image: xt_total})
                    output_fake= sess.run([disc_out_fake], {t_input_gen: xgenin})
                    for d in range(0,1):
                        list_real[d,i]=output_real[0][d,0]
                    for d in range(0,1):
                        list_fake[d,i]=output_fake[0][d,0]

        log = "[*] Epoch: [%2d/%2d], d_loss: %.8f g_loss: %.8f" % (j+1, epochs, total_d_loss / n_iter,total_g_loss / n_iter)
        print(log)
        saveoutput.write('log is')
        saveoutput.write(str(log) + '\n')
        
        loss_Dis.append(total_d_loss / n_iter)
        loss_Gen.append(total_g_loss / n_iter)
        
        plt.subplot(221)
        plt.scatter(count,list_real[0],marker='o',color='g')
        #plt.scatter(count,sigmoid(list_real[1]),marker='o',color='g')
        plt.legend()
        plt.axis([0,750,-10,10])
        plt.ylabel('output of disc for real image')
        plt.xlabel('number of output')
        plt.title(' It is output of disc in Epoch={}'.format(j))
        
        plt.subplot(222)
        plt.scatter(count,sigmoid(list_real[0]),marker='o',color='g')
        #plt.scatter(count,sigmoid(list_real[1]),marker='o',color='g')
        plt.legend()
        plt.axis([0,750,-1,1])
        plt.ylabel('output(s) of disc for real image')
        plt.xlabel('number of output')
        plt.title(' It is output(s) of disc in Epoch={}'.format(j))
        
        plt.subplot(223)
        plt.scatter(count,list_fake[0],marker='*',color='r')
        #plt.scatter(count,sigmoid(list_fake[1]),marker='*',color='r')
        plt.legend()
        plt.axis([0,750,-10,10])
        plt.ylabel('output of disc for fake image')
        plt.xlabel('number of output')
        plt.title(' It is output of disc in Epoch={}'.format(j))
        
        plt.subplot(224)
        plt.scatter(count,sigmoid(list_fake[0]),marker='*',color='r')
        #plt.scatter(count,sigmoid(list_fake[1]),marker='*',color='r')
        plt.legend()
        plt.axis([0,750,-1,1])
        plt.ylabel('output(s) of disc for fake image')
        plt.xlabel('number of output')
        plt.title(' It is output(s) of disc in Epoch={}'.format(j))
        
        plt.show()

        print('list_real',list_real)
        print('list_fake',list_fake)
        print('sigmoid(list_real)',sigmoid(list_real))
        print('sigmoid(list_fake)',sigmoid(list_fake))
        ## save model
        if (j % 1 == 0):
        #if (j != 0) and (j % 5 == 0):
            tl.files.save_npz(net_gen.all_params, name=checkpoint_dir + '/g_{}.npz'.format('srgan'), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format('srgan'), sess=sess) 
    plt.subplot(2, 1, 1)
    plt.plot(loss_Dis, label='loss_Dis')
    plt.subplot(2, 2, 4)
    plt.plot(loss_Gen, label='loss_Gen')
    plt.legend()
    plt.show()

def evaluate(upsampling_factor, residual_blocks, feature_size, checkpoint_dir_restore, path_volumes, nn, subpixel_NN,img_height, img_width, img_depth):
    traindataset = Train_dataset(1)
    print(len(traindataset.subject_list))
    #train
    #iterations=5
    #test    
    iterations =46
    print(iterations)

    batch_size = 1
    div_patches = 4
    count=range(iterations)
    # define model
    t_input_gen = tf.placeholder('float32', [1, None, None, None, 1],
                                 name='t_image_input_to_SRGAN_generator')
    net_gen = generator(input_gen=t_input_gen, kernel=3, nb=residual_blocks,
                              upscaling_factor=upsampling_factor, feature_size=feature_size, subpixel_NN=subpixel_NN,
                              img_height=img_height, img_width=img_width, img_depth=img_depth, nn=nn,
                              is_train=False, reuse=False)

    # restore g
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    
    print(checkpoint_dir)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_init.npz', network=net_gen)
    
    mse_GAN=[]
    mse_bicubic=[]
    PSNR_GAN=[]
    PSNR_bicubic=[]
    for i in range(0, iterations):
        print(' i is=',i)

        xt_total = traindataset.data_true(750 + i)
        print('xt_total.shape',xt_total.shape)
        normfactor = float((np.amax(xt_total[0])) / 2.0)
        x_generator = ((xt_total[0] - normfactor) / normfactor)
        x_HR=x_generator
        x_HR=x_HR[np.newaxis, :]
        print('x_HR.shape',x_HR.shape)
        res = float(1.0 / upsampling_factor)

        x_generator = gaussian_filter(x_generator, sigma=1)
        x_generator = zoom(x_generator, [res, res, res, 1], prefilter=False)
        print('x_generator.shape',x_generator.shape)
        x_generator=x_generator[np.newaxis, :]
        print('2x_generator.shape',x_generator.shape)
        xg_generated = sess.run(net_gen.outputs, {t_input_gen: x_generator})
        xg=xg_generated
        print('xg.shape',xg.shape)
        xg_generated = ((xg_generated + 1) * normfactor)
        
        volume_real = xt_total[0]
        print('volume_real.shape',volume_real.shape)
        
        volume_generated = xg_generated[0]
        print('volume_generated.shape',volume_generated.shape)
        
        
        volume_gen_img = x_generator[0]
        if normfactor != 0:
            volume_gen_img = ((volume_gen_img + 1) * normfactor)  # denormalize
            print('volume_gen_img',volume_gen_img.shape)
        
        # save volumes
        filename_gen = os.path.join(path_volumes, str(i) + 'gen.nii.gz')
        img_volume_gen = nib.Nifti1Image(volume_generated, np.eye(4))
        img_volume_gen.to_filename(filename_gen)
        
        filename_real = os.path.join(path_volumes, str(i) + 'real.nii.gz')
        img_volume_real = nib.Nifti1Image(volume_real, np.eye(4))
        img_volume_real.to_filename(filename_real)
        
        filename_LR = os.path.join(path_volumes, str(i) + 'LR.nii.gz')
        img_volume_LR = nib.Nifti1Image(volume_gen_img, np.eye(4))
        img_volume_LR.to_filename(filename_LR)
        
        xg=xg[:,:,:,:,0]
        x_HR=x_HR[:,:,:,:,0]
        
        x_HR = tf.cast(x_HR, dtype='float64')
        xg = tf.cast(xg, dtype='float64')
        
        mse_loss_GAN = tl.cost.mean_squared_error(xg, x_HR, is_mean=True)
        mse_loss_GAN=sess.run(mse_loss_GAN)
        psnr_GAN = 20 * log10(1 / sqrt(mse_loss_GAN))
        
        x_generator=x_generator[:,:,:,:,0]
        x_SR_bicubic = skimage.transform.resize(x_generator, [1,80, 80, 80])
        x_SR_bicubic = tf.cast(x_SR_bicubic, dtype='float64')
 
        mse_loss_bicubic = tl.cost.mean_squared_error(x_SR_bicubic, x_HR, is_mean=True)
        mse_loss_bicubic=sess.run(mse_loss_bicubic)
        psnr_bicubic = 20 * log10(1 / sqrt( mse_loss_bicubic))
        
        mse_GAN.append( mse_loss_GAN)
        mse_bicubic.append( mse_loss_bicubic)
        PSNR_GAN.append( psnr_GAN)
        PSNR_bicubic.append( psnr_bicubic)

    print('mse_GAN=',mse_GAN)
    print('mse_bicubic=',mse_bicubic)
    
    
    #plt.subplot(221)
    plt.plot(count,mse_GAN,marker='o',linestyle='-',color='r',label='mse_GAN')
    plt.ylabel('mse_loss')
    plt.xlabel('number of sample')
    plt.legend()
    plt.savefig("mse_GAN", format="png")
    plt.show()
    
    plt.plot(count,mse_bicubic,marker='o',linestyle='--',color='g',label='mse_bicubic')
    plt.ylabel('mse_bicubic')
    plt.xlabel('number of sample')
    plt.legend()
    plt.savefig("mse_bicubic", format="png")
    plt.show()
    
    plt.plot(count,PSNR_GAN,marker='o',linestyle='-',color='r',label='PSNR_GAN')
    plt.plot(count,PSNR_bicubic,marker='o',linestyle='--',color='g',label='PSNR_bicubic')
    plt.ylabel('PSNR')
    plt.xlabel('number of sample')
    plt.legend() 
    plt.savefig("PSNR", format="png")
    
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict script')
    parser.add_argument('-residual_blocks', default=16, help='Number of residual blocks')
    parser.add_argument('-upsampling_factor', default=4, help='Upsampling factor')
    parser.add_argument('-evaluate', default=False, help='Test the model')
    parser.add_argument('-subpixel_NN', default=False, help='Use subpixel nearest neighbour')
    parser.add_argument('-nn', default=False, help='Use Upsampling3D + nearest neighbour, RC')
    parser.add_argument('-feature_size', default=32, help='Number of filters')
    parser.add_argument('-restore', help='Checkpoint path to restore training')
    args = parser.parse_args()

    if args.evaluate:
        evaluate(upsampling_factor=int(args.upsampling_factor), feature_size=int(args.feature_size),
                 residual_blocks=int(args.residual_blocks), checkpoint_dir_restore=checkpoint_dir_restore,
                 path_volumes=path_volumes, subpixel_NN=args.subpixel_NN, nn=args.nn, img_width=80,
                 img_height=80, img_depth=80)
    else:

        train(upscaling_factor=int(args.upsampling_factor), feature_size=int(args.feature_size),
              subpixel_NN=args.subpixel_NN, nn=args.nn, residual_blocks=int(args.residual_blocks),
              path_prediction=path_prediction_Gen,path_prediction_Gan=path_prediction_Gan, checkpoint_dir=checkpoint_dir, img_width=80,
              img_height=80, img_depth=80, batch_size=1, restore=args.restore)

