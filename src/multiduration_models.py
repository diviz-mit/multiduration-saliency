import numpy as np
import keras
import sys
import os
from keras.layers import Layer, Input, Multiply, Dropout, TimeDistributed, LSTM, Activation, Lambda, Conv2D, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, BatchNormalization, Concatenate
import keras.backend as K
from keras.models import Model
import tensorflow as tf
from keras.utils import Sequence
import cv2
import scipy.io
import math
from attentive_convlstm_new import AttentiveConvLSTM2D
from dcn_resnet_new import dcn_resnet
from gaussian_prior_new import LearningPrior
from sal_imp_utilities import *
from xception_custom import Xception_wrapper


# Multiple ways of processing outputs in a multiduration model:
# 1. TimeDistributed (each branch shares parameters)
# 2. Separate branches for each slice
# 3. 3D convs (would combine infomration from different timesteps, which could be good for acc)


def decoder_block_timedist(x, dil_rate=(2,2), print_shapes=True, dec_filt=1024):
    # Dilated convolutions
    x = TimeDistributed(Conv2D(dec_filt, 3, padding='same', activation='relu', dilation_rate=dil_rate))(x)
    x = TimeDistributed(Conv2D(dec_filt, 3, padding='same', activation='relu',  dilation_rate=dil_rate))(x)
    x = TimeDistributed(UpSampling2D((2,2), interpolation='bilinear'))(x)

    x = TimeDistributed(Conv2D(dec_filt//2, 3, padding='same', activation='relu',  dilation_rate=dil_rate))(x)
    x = TimeDistributed(Conv2D(dec_filt//2, 3, padding='same', activation='relu',  dilation_rate=dil_rate))(x)
    x = TimeDistributed(UpSampling2D((2,2), interpolation='bilinear'))(x)

    x = TimeDistributed(Conv2D(dec_filt//4, 3, padding='same', activation='relu',  dilation_rate=dil_rate))(x)
    x = TimeDistributed(Conv2D(dec_filt//4, 3, padding='same', activation='relu',  dilation_rate=dil_rate))(x)
    x = TimeDistributed(UpSampling2D((4,4), interpolation='bilinear'))(x)
    if print_shapes: print('Shape after last ups:',x.shape)

    # Final conv to get to a heatmap
    x = TimeDistributed(Conv2D(1, kernel_size=1, padding='same', activation='relu'))(x)
    if print_shapes: print('Shape after 1x1 conv:',x.shape)

    return x


def decoder_block_simple_timedist(x, dil_rate=(1,1), print_shapes=True, dec_filt=1024):

    x = TimeDistributed(Conv2D(dec_filt, 3, padding='same', activation='relu', dilation_rate=dil_rate))(x)
    x = TimeDistributed(UpSampling2D((2,2), interpolation='bilinear'))(x)

    x = TimeDistributed(Conv2D(dec_filt//2, 3, padding='same', activation='relu', dilation_rate=dil_rate))(x)
    x = TimeDistributed(UpSampling2D((2,2), interpolation='bilinear'))(x)

    x = TimeDistributed(Conv2D(dec_filt//4, 3, padding='same', activation='relu',  dilation_rate=dil_rate))(x)
    x = TimeDistributed(UpSampling2D((4,4), interpolation='bilinear'))(x)
    if print_shapes: print('Shape after last ups:',x.shape)

    # Final conv to get to a heatmap
    x = TimeDistributed(Conv2D(1, kernel_size=1, padding='same', activation='relu'))(x)
    if print_shapes: print('Shape after 1x1 conv:',x.shape)

    return x




def xception_decoder_timedist(input_shape = (shape_r, shape_c, 3),
                                verbose=True,
                                print_shapes=True,
                                n_outs=1,
                                ups=8,
                                dil_rate = (1,1)):
    inp = Input(shape=input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception:',xception.output.shape)

    x = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=1), nb_timestep, axis=1),
                          lambda s: (s[0], nb_timestep) + s[1:])(xception.output)

    ## DECODER ##
    outs_dec = decoder_block_timedist(x, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=512)

    outs_final = [outs_dec]*n_outs

    # Building model
    m = Model(inp, outs_final)
    if verbose:
        m.summary()
    return m

def resnet_decoder_timedist(input_shape = (shape_r, shape_c, 3),
                                verbose=True,
                                print_shapes=True,
                                n_outs=1,
                                ups=8,
                                dil_rate = (1,1)):

    inp = Input(shape=input_shape)

    ### ENCODER ###
    dcn = dcn_resnet(input_tensor=inp)
    if print_shapes: print('resnet output shape:',dcn.output.shape)
    x = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=1), nb_timestep, axis=1),
                      lambda s: (s[0], nb_timestep) + s[1:])(dcn.output)
    ## DECODER ##
    outs_dec = decoder_block_timedist(x, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=256)

    outs_final = [outs_dec]*n_outs

    # Building model
    m = Model(inp, outs_final)
    if verbose:
        m.summary()
    return m


def lstm_timedist(input_shape = (224, 224, 3), conv_filters=512, lstm_filters=512, att_filters=512,
                   verbose=True, print_shapes=True, n_outs=1, nb_timestep = 3, ups=upsampling_factor):

    '''LSTM with outputs at each timestep of the LSTM. Needs to be trained with the singlestream loss.
        DOESNT WORK! Imagenet pretrained models are essential for this task'''

    inp = Input(shape=input_shape)
    x = Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1')(inp)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    if print_shapes: print('x.shape after input block',x.shape)

    # Attentive ConvLSTM
    att_convlstm = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=1), nb_timestep, axis=1),
                          lambda s: (s[0], nb_timestep) + s[1:])(x)
    att_convlstm = AttentiveConvLSTM2D(filters=lstm_filters, attentive_filters=att_filters, kernel_size=(3,3),
                            attentive_kernel_size=(3,3), padding='same', return_sequences=True)(att_convlstm)

    if print_shapes: print('(att_convlstm.shape',att_convlstm.shape)

    outs_final = TimeDistributed(Conv2D(1, kernel_size=1, padding='same', activation='relu'))(att_convlstm)
    outs_final = TimeDistributed(UpSampling2D((ups,ups)))(outs_final)

    if print_shapes:
        print('outs_final shape:', outs_final.shape)
    outs_final = [outs_final]*n_outs

    m = Model(inp, outs_final)
    if verbose:
        m.summary()
    return m



def sam_xception_timedist(input_shape = (shape_r, shape_c, 3), conv_filters=512, lstm_filters=512, att_filters=512,
                   verbose=True, print_shapes=True, n_outs=1, ups=8, nb_gaussian=nb_gaussian):
    '''SAM-ResNet ported from the original code.'''


    inp = Input(shape=input_shape)

    # Input CNN
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception:',xception.output.shape)

    conv_feat = Conv2D(conv_filters, 3, padding='same', activation='relu')(xception.output)
    if print_shapes:
        print('Shape after first conv after dcn_resnet:',conv_feat.shape)

    # Attentive ConvLSTM
    att_convlstm = Lambda(repeat, repeat_shape)(conv_feat)
    att_convlstm = AttentiveConvLSTM2D(filters=lstm_filters, attentive_filters=att_filters, kernel_size=(3,3),
                            attentive_kernel_size=(3,3), padding='same', return_sequences=True)(att_convlstm)

    # Learned Prior (1)
    priors1 = TimeDistributed(LearningPrior(nb_gaussian=nb_gaussian))(att_convlstm)
    concat1 = Concatenate(axis=-1)([att_convlstm, priors1])
    dil_conv1 = TimeDistributed(Conv2D(conv_filters, 5, padding='same', activation='relu', dilation_rate=(4, 4)))(concat1)

    # Learned Prior (2)
    priors2 = TimeDistributed(LearningPrior(nb_gaussian=nb_gaussian))(att_convlstm)
    concat2 = Concatenate(axis=-1)([dil_conv1, priors2])
    dil_conv2 = TimeDistributed(Conv2D(conv_filters, 5, padding='same', activation='relu', dilation_rate=(4, 4)))(concat2)

    # Final conv to get to a heatmap
    outs = TimeDistributed(Conv2D(1, kernel_size=1, padding='same', activation='relu'))(dil_conv2)
    if print_shapes:
        print('Shape after 1x1 conv:',outs.shape)

    # Upsampling back to input shape
    outs_up = TimeDistributed(UpSampling2D(size=(ups,ups), interpolation='bilinear'))(outs)
    if print_shapes:
        print('shape after upsampling',outs_up.shape)


    outs_final = [outs_up]*n_outs


    # Building model
    m = Model(inp, outs_final)
    if verbose:
        m.summary()

    return m


def sam_resnet_timedist(input_shape = (shape_r, shape_c, 3), conv_filters=512, lstm_filters=512, att_filters=512,
                   verbose=True, print_shapes=True, n_outs=1, ups=8, nb_gaussian=nb_gaussian):
    '''SAM-ResNet ported from the original code.'''


    inp = Input(shape=input_shape)

    # Input CNN
    dcn = dcn_resnet(input_tensor=inp)
    conv_feat = Conv2D(conv_filters, 3, padding='same', activation='relu')(dcn.output)
    if print_shapes:
        print('Shape after first conv after dcn_resnet:',conv_feat.shape)

    # Attentive ConvLSTM
    att_convlstm = Lambda(repeat, repeat_shape)(conv_feat)
    att_convlstm = AttentiveConvLSTM2D(filters=lstm_filters, attentive_filters=att_filters, kernel_size=(3,3),
                            attentive_kernel_size=(3,3), padding='same', return_sequences=True)(att_convlstm)

    # Learned Prior (1)
    priors1 = TimeDistributed(LearningPrior(nb_gaussian=nb_gaussian))(att_convlstm)
    concat1 = Concatenate(axis=-1)([att_convlstm, priors1])
    dil_conv1 = TimeDistributed(Conv2D(conv_filters, 5, padding='same', activation='relu', dilation_rate=(4, 4)))(concat1)

    # Learned Prior (2)
    priors2 = TimeDistributed(LearningPrior(nb_gaussian=nb_gaussian))(att_convlstm)
    concat2 = Concatenate(axis=-1)([dil_conv1, priors2])
    dil_conv2 = TimeDistributed(Conv2D(conv_filters, 5, padding='same', activation='relu', dilation_rate=(4, 4)))(concat2)

    # Final conv to get to a heatmap
    outs = TimeDistributed(Conv2D(1, kernel_size=1, padding='same', activation='relu'))(dil_conv2)
    if print_shapes:
        print('Shape after 1x1 conv:',outs.shape)

    # Upsampling back to input shape
    outs_up = TimeDistributed(UpSampling2D(size=(ups,ups), interpolation='bilinear'))(outs)
    if print_shapes:
        print('shape after upsampling',outs_up.shape)


    outs_final = [outs_up]*n_outs


    # Building model
    m = Model(inp, outs_final)
    if verbose:
        m.summary()

    return m




def sam_resnet_3d(input_shape = (224, 224, 3), filt_3d = 128, conv_filters=128, lstm_filters=128, att_filters=128,
                   verbose=True, print_shapes=True, n_outs=1, nb_timestep = 3, ups=upsampling_factor):


    inp = Input(shape=input_shape)

    # Input CNN
    dcn = dcn_resnet(input_tensor=inp)
    conv_feat = Conv2D(conv_filters, 3, padding='same', activation='relu')(dcn.output)
    if print_shapes:
        print('Shape after first conv after dcn_resnet:',conv_feat.shape)


    # Attentive ConvLSTM
    att_convlstm = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=1), nb_timestep, axis=1),
                          lambda s: (s[0], nb_timestep) + s[1:])(x)
    att_convlstm = AttentiveConvLSTM2D(filters=lstm_filters, attentive_filters=att_filters, kernel_size=(3,3),
                            attentive_kernel_size=(3,3), padding='same', return_sequences=True)(att_convlstm)

    if print_shapes: print('att_convlstm output shape',att_convlstm.shape)

    # Output flow
    x = Conv3D(filt_3d, (3,3,3), strides=(1, 1, 1),
                 padding='same',
                 dilation_rate=(4, 4, 1),
                 activation=None,
                 kernel_initializer='he_normal')(att_convlstm)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = Conv3D(filt_3d, (3,3,1), strides=(1, 1, 1),
                 padding='same',
                 dilation_rate=(4, 4, 1),
                 activation=None,
                 kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(1, (1,1,1), strides=(1, 1, 1),
                 padding='same',
                 activation='relu',
                 kernel_initializer='he_normal')(x)

    out_final = TimeDistributed(UpSampling2D(size=(ups,ups), interpolation='bilinear'))(x)

    if print_shapes: print('outs_final shape:', outs_final.shape)
    outs_final = [outs_final]*n_outs

    m = Model(inp, outs_final)
    if verbose:
        m.summary()
    return m



def xception_lstm_md(input_shape = (shape_r, shape_c, 3),
                     conv_filters=256,
                     lstm_filters=256,
                     att_filters=256,
                     verbose=True,
                     print_shapes=True,
                     n_outs=1,
                     ups=8):

    inp = Input(shape = input_shape)
    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception:',xception.output.shape)

    conv_feat = Conv2D(conv_filters, 3, padding='same', activation='relu')(xception.output)
    if print_shapes: print('Shape after first conv after xception:',conv_feat.shape)

    # Attentive ConvLSTM
    att_convlstm = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=1), nb_timestep, axis=1),
                          lambda s: (s[0], nb_timestep) + s[1:])(conv_feat)
    att_convlstm = AttentiveConvLSTM2D(filters=lstm_filters, attentive_filters=att_filters, kernel_size=(3,3),
                            attentive_kernel_size=(3,3), padding='same', return_sequences=True)(att_convlstm)

    if print_shapes: print('(att_convlstm.shape',att_convlstm.shape)

    # Dilated convolutions (priors would go here)
    dil_conv1 = TimeDistributed(Conv2D(conv_filters, 5, padding='same', activation='relu', dilation_rate=(4, 4)))(att_convlstm)
    dil_conv2 = TimeDistributed(Conv2D(conv_filters, 5, padding='same', activation='relu', dilation_rate=(4, 4)))(dil_conv1)

    # Final conv to get to a heatmap
    outs = TimeDistributed(Conv2D(1, kernel_size=1, padding='same', activation='relu'))(dil_conv2)
    if print_shapes: print('Shape after 1x1 conv:',outs.shape)

    # Upsampling back to input shape
    outs_up = TimeDistributed(UpSampling2D(size=(ups,ups), interpolation='bilinear'))(outs)
    if print_shapes: print('shape after upsampling',outs_up.shape)


    outs_final = [outs_up]*n_outs


    # Building model
    m = Model(inp, outs_final)
    if verbose:
        m.summary()

    return m

def dcnn_3stream(input_shape = (224, 224, 3), conv_filters=512, n_streams=3, verbose=True, print_shapes=True):

    def out_shape(s):
        print("out shape", out_shape)
        return (s[0],2) + s[1:]

    def _output_stream(x):
        x = Conv2D(filters=conv_filters, kernel_size=3, padding='same', activation='relu')(x)

        # convolve with a kernel size of 1 to flatten the tensor (w x h x nfeatures) from
        # 40 x 30 x 512 to 40 x 30 x 1
        # use kernel size of 1 so you ONLY flatten instead of doing local computations
        # use one filter so that you only get one image channel
        x = Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(x)

        # upsample from 40x30 to 320x240 (8x upsample)
        x = UpSampling2D(size=(16, 16), interpolation='bilinear')(x)
        x = Lambda(lambda y: K.repeat_elements(K.expand_dims(y, axis=1),2,axis=1), output_shape = out_shape)(x)
        return x

    inp = Input(shape=input_shape)
    dcn = dcn_resnet(input_tensor=inp)

    outs = [_output_stream(dcn.output) for _ in range(n_streams)]

    model = Model (inputs = inp, outputs = outs)

    if verbose:
        model.summary()
    return model


def xception_3stream(input_shape = (shape_r, shape_c, 3),conv_filters=512,
                   verbose=True, print_shapes=True, n_streams=3, ups=16):
    inp = Input(shape=input_shape)
    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception:',xception.output.shape)
#     x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(inp)

    def out_stream(x):
        x = Conv2D(filters=conv_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(1, kernel_size=1, padding='same', activation='relu')(x)
        x = UpSampling2D(size=(ups,ups), interpolation='bilinear')(x)
        x = Lambda(lambda y: K.repeat_elements(K.expand_dims(y, axis=1),2,axis=1), output_shape = lambda s: (s[0],2) + s[1:])(x)
        if print_shapes:    print('Shape after ups:',x.shape)
        return x

    outs = [out_stream(xception.output) for _ in range(n_streams)]

    # print('len(outs)',len(outs))
    # print('outs[0].shape',outs[0].shape)

    m = Model(inp, outs)

    if verbose:
        m.summary()
    return m


def xception_se_lstm(input_shape = (shape_r, shape_c, 3),
                     conv_filters=256,
                     lstm_filters=512,
                     verbose=True,
                     print_shapes=True,
                     n_outs=1,
                     ups=8,
                     freeze_enc=False,
                     return_sequences=True):
    inp = Input(shape = input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception output shapes:',xception.output.shape)
    if freeze_enc:
        for layer in xception.layers:
	        layer.trainable = False

    ### LSTM over SE representation ###
    x = se_lstm_block_timedist(xception.output, nb_timestep, lstm_filters=lstm_filters, return_sequences=return_sequences)

    ### DECODER ###
    outs_dec = decoder_block_timedist(x, dil_rate=(2,2), print_shapes=print_shapes, dec_filt=conv_filters)

    outs_final = [outs_dec]*n_outs
    m = Model(inp, outs_final)
    if verbose:
        m.summary()
    return m

def xception_se_lstm_nodecoder(input_shape = (shape_r, shape_c, 3),
                     conv_filters=512,
                     lstm_filters=512,
                     verbose=True,
                     print_shapes=True,
                     n_outs=1,
                     ups=8,
                     freeze_enc=False,
                     return_sequences=True):
    inp = Input(shape = input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception output shapes:',xception.output.shape)
    if freeze_enc:
        for layer in xception.layers:
	        layer.trainable = False

    ### LSTM over SE representation ###
    x = se_lstm_block_timedist(xception.output, nb_timestep, lstm_filters=lstm_filters, return_sequences=return_sequences)

    ## DECODER ###
    x = TimeDistributed(Dropout(0.3))(x)
    x = TimeDistributed(Conv2D(filters=conv_filters, kernel_size=3, padding='same', activation='relu'))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = TimeDistributed(Conv2D(1, kernel_size=1, padding='same', activation='relu'))(x)
    outs_dec = TimeDistributed(UpSampling2D(size=(ups,ups), interpolation='bilinear'))(x)

    outs_final = [outs_dec]*n_outs
    m = Model(inp, outs_final)
    if verbose:
        m.summary()
    return m



def se_lstm_block_timedist(inp, nb_timestep, units=512, print_shapes=True, lstm_filters=512, return_sequences=True):

    inp_rep = Lambda(lambda y: K.repeat_elements(K.expand_dims(y, axis=1), nb_timestep, axis=1),
                     lambda s: (s[0], nb_timestep) + s[1:])(inp)
    x = TimeDistributed(GlobalAveragePooling2D())(inp_rep)
    if print_shapes: print('shape after AvgPool',x.shape)
    x = TimeDistributed(Dense(units, activation='relu'))(x)
    if print_shapes: print('shape after first dense',x.shape)

    # Normally se block would feed into another fully connected. Instead, we feed it to an LSTM.
    x = LSTM(lstm_filters, return_sequences=return_sequences, unroll=True, activation='relu')(x) #, activation='relu'
    if print_shapes: print('shape after lstm',x.shape)

    x = TimeDistributed(Dense(inp.shape[-1].value, activation='sigmoid'))(x)
    if print_shapes: print('shape after second dense:', x.shape)

    x = Lambda(lambda y: K.expand_dims(K.expand_dims(y, axis=2),axis=2),
                lambda s: (s[0], s[1], 1, 1, s[2]))(x)
    if print_shapes: print('shape before mult',x.shape)

    # x is (bs, t, 1, 1, 2048)
    # inp_rep is (bs, t, r, c, 2048)
    out = Multiply()([x,inp_rep])

    print('shape out',out.shape)
    # out is (bs, t, r, c, 2048)

    return out


def xception_nasnet_combination():

    # Load nasnet pretrained on saliency
    # Load xception prerained on saliency

    # Extract middle layers and pass them through 1x1

    # concatenate those outputs with final ds outputs from individual nmodels

    # Pass them through convlstm

    # Decode and upsample

    #1x1
    pass
