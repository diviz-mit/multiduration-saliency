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
from multiduration_models import decoder_block_timedist
from xception_custom import Xception_wrapper
from keras.applications import keras_modules_injection

def sam_resnet_new(input_shape, conv_filters=512, lstm_filters=512, att_filters=512, verbose=True, print_shapes=True, n_outs=1, ups=16, nb_gaussian=16, n_lstm_cells=3, **kwargs):
    '''SAM-ResNet ported from the original code.'''
    print("n_outs", n_outs)

    inp = Input(shape=input_shape)

    dcn = dcn_resnet(input_tensor=inp)
    conv_feat = Conv2D(conv_filters, 3, padding='same', activation='relu')(dcn.output)
    if print_shapes:
        print('Shape after first conv after dcn_resnet:',conv_feat.shape)

    # Attentive ConvLSTM
    att_convlstm = Lambda(repeat(n_lstm_cells), repeat_shape(n_lstm_cells))(conv_feat)
    att_convlstm = AttentiveConvLSTM2D(filters=lstm_filters, attentive_filters=att_filters, kernel_size=(3,3),
                            attentive_kernel_size=(3,3), padding='same', return_sequences=False)(att_convlstm)

    # Learned Prior (1)
    priors1 = LearningPrior(nb_gaussian=nb_gaussian)(att_convlstm)
    concat1 = Concatenate(axis=-1)([att_convlstm, priors1])
    dil_conv1 = Conv2D(conv_filters, 5, padding='same', activation='relu', dilation_rate=(4, 4))(concat1)

    # Learned Prior (2)
    priors2 = LearningPrior(nb_gaussian=nb_gaussian)(att_convlstm)
    concat2 = Concatenate(axis=-1)([dil_conv1, priors2])
    dil_conv2 = Conv2D(conv_filters, 5, padding='same', activation='relu', dilation_rate=(4, 4))(concat2)

    # Final conv to get to a heatmap
    outs = Conv2D(1, kernel_size=1, padding='same', activation='relu')(dil_conv2)
    if print_shapes:
        print('Shape after 1x1 conv:',outs.shape)

    # Upsampling back to input shape
    outs_up = UpSampling2D(size=(ups,ups), interpolation='bilinear')(outs)
    if print_shapes:
        print('shape after upsampling',outs_up.shape)


    outs_final = [outs_up]*n_outs


    # Building model
    m = Model(inp, outs_final)
    if verbose:
        m.summary()

    return m


