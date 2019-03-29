import keras.backend as K
import numpy as np
from sal_imp_utilities import *

def loss_wrapper(loss, input_shape): 
    shape_r_out, shape_c_out = input_shape
    print("shape r out, shape c out", shape_r_out, shape_c_out)
    def _wrapper(y_true, y_pred): 
        return loss(y_true, y_pred, shape_r_out, shape_c_out)
    return _wrapper

# KL-Divergence Loss
def kl_divergence(y_true, y_pred, shape_r_out, shape_c_out):
    print("INSIDE KL", shape_r_out, shape_c_out)

    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)

    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())


    return K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=1), axis=1)

def kl_time(y_true, y_pred, shape_r_out, shape_c_out):
    if len(y_true.shape) == 5:
        ax = 2
    else:
        ax = 1
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)

    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    kl_out = K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=ax), axis=ax)

    if len(y_true.shape) == 5:
        kl_out = K.mean(kl_out, axis = 1)

    return kl_out

# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred, shape_r_out, shape_c_out):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=1), axis=1)
    sum_x = K.sum(K.sum(y_true, axis=1), axis=1)
    sum_y = K.sum(K.sum(y_pred, axis=1), axis=1)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=1), axis=1)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=1), axis=1)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return num / den

def cc_time(y_true, y_pred, shape_r_out, shape_c_out):
    if len(y_true.shape) == 5:
        ax = 2
    else:
        ax = 1
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=ax), axis=ax)
    sum_x = K.sum(K.sum(y_true, axis=ax), axis=ax)
    sum_y = K.sum(K.sum(y_pred, axis=ax), axis=ax)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=ax), axis=ax)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=ax), axis=ax)


    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    if len(y_true.shape) == 5:
        cc_out = K.mean(num / den, axis = 1)
    else:
        cc_out = num / den

    return cc_out

# Normalized Scanpath Saliency Loss
def nss_time(y_true, y_pred, shape_r_out, shape_c_out):
    if len(y_true.shape) == 5:
        ax = 2
    else:
        ax = 1

#     print('y_pred.shape',y_pred.shape)

    maxi = K.max(K.max(y_pred, axis=ax), axis=ax)
#     print('maxi.shape',maxi.shape)
    first_rep = K.repeat_elements(K.expand_dims(maxi, axis=ax),shape_r_out, axis=ax)
#     print('first_rep.shape',first_rep.shape)
    max_y_pred = K.repeat_elements(K.expand_dims(first_rep, axis=ax+1), shape_c_out, axis=ax+1)
    y_pred /= max_y_pred
#     y_pred /= K.expand_dims(K.expand_dims(maxi))


#     print('y_pred.shape after max divison:',y_pred.shape)

    if len(y_true.shape) == 5:
        y_pred_flatten = K.reshape(y_pred, (K.shape(y_pred)[0],K.shape(y_pred)[1],K.shape(y_pred)[2]*K.shape(y_pred)[3]*K.shape(y_pred)[4])) #K.batch_flatten(y_pred)
    else:
        y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)),
                                                               shape_r_out, axis=ax)), shape_c_out, axis=ax+1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)),
                                                              shape_r_out, axis=ax)), shape_c_out, axis=ax+1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    num = K.sum(K.sum(y_true * y_pred, axis=ax), axis=ax)
    den = K.sum(K.sum(y_true, axis=ax), axis=ax) + K.epsilon()

    if len(y_true.shape) == 5:
        nss_out = K.mean(num/den, axis = 1)
    else:
        nss_out = num/den

    return nss_out


def nss(y_true, y_pred, shape_r_out, shape_c_out):
    ax = 1
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=ax), axis=ax), axis=ax+1),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)

#     maxi = K.max(K.max(y_pred, axis=ax), axis=ax)
    y_pred /= max_y_pred
#     y_pred /= K.expand_dims(K.expand_dims(maxi))


    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)),
                                                               shape_r_out, axis=ax)), shape_c_out, axis=ax+1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)),
                                                              shape_r_out, axis=ax)), shape_c_out, axis=ax+1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    den = K.sum(K.sum(y_true * y_pred, axis=ax), axis=ax)
    nom = K.sum(K.sum(y_true, axis=ax), axis=ax) + K.epsilon()

    nss_out = den/nom

    return nss_out

def cc_match(y_true, y_pred, shape_r_out, shape_c_out):
    '''Calculates CC between initial, mid and final timestep from both y_true and y_pred
    and calculates the mean absolute error between the CCs from y_true and from y_pred.
    Requires a y_true and y_pred to be tensors of shape (bs, t, r, c, 1)'''

    mid = 1 # y_true.shape[1].value//2
    ccim_true = cc_time(y_true[:,0,...], y_true[:,mid,...])
    # ccif_true = cc_time(y_true[:,0,...], y_true[:,-1,...])
    ccmf_true = cc_time(y_true[:,mid,...], y_true[:,-1,...])

    ccim_pred = cc_time(y_pred[:,0,...], y_pred[:,mid,...])
    # ccif_pred = cc_time(y_pred[:,0,...], y_pred[:,-1,...])
    ccmf_pred = cc_time(y_pred[:,mid,...], y_pred[:,-1,...])

    return  (K.abs(ccim_true-ccim_pred) + K.abs(ccmf_true-ccmf_pred) )/2  #+ K.abs(ccif_true-ccif_pred) + K.abs(ccmf_true-ccmf_pred) )/3

#def kl_cc_nss_combined(lw=[10,-2,-1]):
#    '''Loss function that combines cc, nss and kl. Beacuse nss receives a different ground truth than kl and cc (maps),
#        the function requires y_true to contains both maps. It has to be a tensor with dimensions [bs, 2, r, c, 1]. y_pred also
#        has to be a tensor of the same dim, so the model should add a 5th dimension between bs and r and repeat the predict map
#        twice along that dim.
#    '''
#    def loss(y_true, y_pred):
#
#        map_true = y_true[:,0,...]
#        fix_true = y_true[:,1,...]
#        pred = y_pred[:,0,...]
#
#    #     print('fct kl_cc_nss_combined: y_pred.shape',y_pred.shape)
#    #     print('fct kl_cc_nss_combined: map_true.shape',map_true.shape)
#    #     print('fct kl_cc_nss_combined: fix_true.shape',fix_true.shape)
#
#        k = kl_divergence(map_true, pred)
#        c = correlation_coefficient(map_true, pred)
#        n = nss(fix_true, pred)
#
#    #     print('kl',K.get_value(k))
#    #     print('cc',K.get_value(c))
#    #     print('nss',K.get_value(n))
#
#        return lw[0]*k+lw[1]*c+lw[2]*n
#
#    return loss
#     return nss(y_true, y_pred) + kl_divergence(y_true, y_pred) + correlation_coefficient(y_true, y_pred)


# def multi_duration_loss_kl_cc(y_true, y_pred):
#     return kl_time(y_true, y_pred) + cc_time(y_true, y_pred)

# def multi_duration_loss_kl_cc_nss(y_true, y_pred):
#     '''Loss for multi_duration saliency models. Both y_true and y_pred
#     are tensors of dimension [b_s, time, row, col, ch].'''

#     return nss_time(y_true, y_pred) + kl_time(y_true, y_pred) + cc_time(y_true, y_pred)
