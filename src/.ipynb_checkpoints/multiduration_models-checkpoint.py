import numpy as np
from keras.layers import Layer, Input, Multiply, Dropout, TimeDistributed, LSTM, Activation, Lambda, Conv2D, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, BatchNormalization, Concatenate
from keras.models import Model
from attentive_convlstm_new import AttentiveConvLSTM2D
from dcn_resnet_new import dcn_resnet
from gaussian_prior_new import LearningPrior
from xception_custom import Xception_wrapper
from sal_imp_utilities import *

def md_sem(input_shape,
           nb_timestep,
           conv_filters=256,
           lstm_filters=512,
           verbose=True,
           print_shapes=True,
           n_outs=1,
           ups=8,
           freeze_enc=False,
           return_sequences=True, 
           **kwargs):
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

def sam_resnet_md(input_shape, nb_timestep, conv_filters=512, lstm_filters=512, att_filters=512,
                   verbose=True, print_shapes=True, n_outs=1, ups=16, nb_gaussian=16, **kwargs):
    '''SAM-ResNet ported from the original code and modified to output multiple durations.'''
    inp = Input(shape=input_shape)

    # Input CNN
    dcn = dcn_resnet(input_tensor=inp)
    conv_feat = Conv2D(conv_filters, 3, padding='same', activation='relu')(dcn.output)
    if print_shapes:
        print('Shape after first conv after dcn_resnet:',conv_feat.shape)

    # Attentive ConvLSTM
    att_convlstm = Lambda(repeat(nb_timestep), repeat_shape(nb_timestep))(conv_feat)
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


