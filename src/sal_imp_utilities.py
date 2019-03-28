import numpy as np
import keras
import matplotlib.pyplot as plt
import sys
import os
from keras.layers import Input, TimeDistributed, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
import keras.backend as K
from keras.models import Model
import tensorflow as tf
from keras.utils import Sequence
from keras.optimizers import Adam, RMSprop, SGD
import cv2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from PIL import Image
from IPython.display import clear_output
import scipy.io
from copy import deepcopy

# DEBUG
DEBUG = False
## number of rows of input images
#cat2000_c = 1920
#cat2000_r = 1080
##cat2000_r_out = 1088 # this is divisible by 16
#cat2000_r_out = 1104 # divible by 48
#cat2000_c_out = cat2000_c # already divisible by 16
#
#cc_c = 300
#cc_r = 225
#cc_c_out = 1776
#cc_r_out = 1344
#
##shape_r = int(cat2000_r/6)
#shape_r = 240
##shape_r = cc_r
## number of cols of input images
##shape_c = int(cat2000_c/6)
#shape_c = 320
##shape_c = cc_c
## number of rows of downsampled maps
#shape_r_gt = 30
## number of cols of downsampled maps
#shape_c_gt = 40
## number of rows of model outputs
##shape_r_out = cat2000_r_out
#shape_r_out = 480
##shape_r_out = cc_r_out
## number of cols of model outputs
##shape_c_out = cat2000_c_out
#shape_c_out = 640
##shape_c_out = cc_c_out
## final upsampling factor
#upsampling_factor = 16
## number of epochs
#nb_epoch = 50
## number of timesteps
#nb_timestep = 3
## number of learned priors
#nb_gaussian = 16

def repeat(x):
    return K.repeat_elements(K.expand_dims(x,axis=1), nb_timestep, axis=1)
#     return K.reshape(K.repeat(K.batch_flatten(x), nb_timestep), (1, nb_timestep, shape_r_gt, shape_c_gt, 512))

def repeat_shape(s):
    return (s[0], nb_timestep) + s[1:]


def padding(img, shape_r, shape_c, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out

def padding_fixation(img, shape_r, shape_c):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_fixmaps(paths, shape_r, shape_c, fix_as_mat=False, fix_key=""):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

#     print('ims.shape:',ims.shape)
    for i, path in enumerate(paths):
        if fix_as_mat:
            mat = scipy.io.loadmat(path)
            if DEBUG:
                print('mat',mat)
            fix_map = mat[fix_key]

        else:
            fix_map = cv2.imread(path, 0)

        if DEBUG:
            print('fix_map shape, np.max(fix_map),np.min(fix_map),np.mean(fix_map)',fix_map.shape,np.max(fix_map),np.min(fix_map),np.mean(fix_map))

        ims[i, :, :, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims

def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

        # DEBUG
#     plt.figure()
#     plt.subplot(1,2,1)
#     plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#     plt.subplot(1,2,2)
#     plt.imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
#     plt.suptitle(path)

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68

    return ims

def reverse_preprocess(img):
    im = deepcopy(img)

    im[:, :, 0] += 103.939
    im[:, :, 1] += 116.779
    im[:, :, 2] += 123.68

#     print(np.max(im), np.min(im), type(im[0][0][0]))

    im = im[...,::-1]


    im = np.array(im, dtype=np.uint8)

    return im

def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i,:,:, 0] = padded_map.astype(np.float32)
        ims[i,:,:, 0] /= 255.0

    return ims


def postprocess_predictions(pred, shape_r, shape_c, blur=False, normalize=False):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    pred = pred / np.max(pred) * 255

#    print('Preparing to resize...')

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]


#    print('Resized')

    if blur:
        sigma=blur
        img = scipy.ndimage.filters.gaussian_filter(img, sigma=sigma)
    if normalize:
        img = img / np.max(img) * 255

    return img



class MultidurationGenerator(Sequence):
    def __init__(self, img_size, map_size, img_filenames, map_filenames=None, fix_filenames=None, batch_size=1, shuffle=True, augment=False, n_output_maps=1, n_output_fixs=1, mode = 'multistream_concat', return_names=False, fix_as_mat=False, fix_key=""):
        '''
        Generator for multi-duration saliency data. Receives lists of images, and t lists of heatmaps and fixations, where t
        is the number of saliency time steps to yield. The generator will automatically infer t from the length of map_filenames.

        This generator has 3 different modes:
        1. multistream_concat: concatenates fix and maps for a given timestep into one tensor of shape (bs, 2, r, c, 1). Then appends
        all these tensors in a list of size t, and yields that tensor as y_true. This mode is made to work with losses that recuperate the
        map and fixation by slicing the y_true tensor internally.

        2. multistream_full: doesn't concatenate the fix and maps; instead, yields all fixations and maps needed for each timestep as a
        different element in the final output list. For example, if we are training with 3 losses and 2 timesteps, this generator will
        yield a list of length 6 as y_true output: 3 maps/fis for timestep1, and 3 maps/fixs for timestep2.

        3. singlestream: concatenates all timesteps in one tensor. for each loss, the generator will yield a tensor of shape
        (bs, time, r, c, 1). If we are working with kl, cc and nss, for example, the generator will output a list of length 3,
        where each element is a tensor of the mentioned shape. This mode should be used with losses that are adapted to tensors with
        a time dimension.

        '''

        print('Instantiating MultidurationGenerator. Number of files received: %d. Batch size: %d. Image size: %s. Augmentation: %d. Mode: %s' % (len(img_filenames), batch_size, str(img_size), augment,mode ))

        if (mode == 'multistream_concat') and (map_filenames is None or fix_filenames is None):
            print('Multistream concat can only be used when both fixations and maps are provided. If only one is enough, use `multistream_full`.')


        self.n_output_maps = n_output_maps
        self.n_output_fixs = n_output_fixs
        self.fix_as_mat = fix_as_mat
        self.fix_key = fix_key

        self.img_filenames = np.array(img_filenames)


        # check that maps make sense
        if map_filenames is not None:
            self.map_filenames = np.array(map_filenames)
            assert all([len(self.img_filenames) == len(elt) for elt in self.map_filenames]), "Mismatch between images and maps. Images size: " + self.img_filenames.shape.__str__() + " Maps size: " + self.map_filenames.shape.__str__()
            self.timesteps = len(map_filenames)
        else:
            self.n_output_maps = 0
            self.map_filenames = None
            print('Warning: No maps filenames provided, no outputs of that kind will be generated')


        # check that fixs make sense
        if fix_filenames is not None:
            self.fix_filenames = np.array(fix_filenames)
            assert all([len(self.img_filenames) == len(elt) for elt in self.fix_filenames]), "Mismatch between images and fixations. Images size: " + self.img_filenames.shape.__str__() + " Fix size: " + self.fix_filenames.shape.__str__()
            self.timesteps = len(fix_filenames)
        else:
            self.n_output_fixs = 0
            self.fix_filenames = None
            print('Warning: No fix filenames provided, no outputs of that kind will be generated')

        self.batch_size = batch_size
        self.img_size = img_size
        self.map_size = map_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        self.return_names = return_names

        # Defining augmentation sequence
        if augment:
            sometimes = lambda aug: iaa.Sometimes(0.4, aug)
            self.seq = iaa.Sequential([
                    sometimes(iaa.CropAndPad(px=(0, 20))), # crop images from each side by 0 to 16px (randomly chosen)
                    iaa.Fliplr(0.5), # horizontally flip 50% of the images
                    sometimes(iaa.CoarseDropout(p=0.1, size_percent=0.05)),
                    sometimes(iaa.Affine(rotate=(-15, 15)))
                ], random_order=True)


        if shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):

        # Get input images
        batch_imgs = self.img_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        images = preprocess_images(batch_imgs, self.img_size[0], self.img_size[1])

        # Get ground truth maps for all times
        if self.n_output_maps>=1:
            maps = []
            for i in range(self.timesteps):
                maps_names_t = self.map_filenames[i][idx * self.batch_size : (idx + 1) * self.batch_size]
                maps_t = preprocess_maps(maps_names_t, self.map_size[0], self.map_size[1])
                maps.append(maps_t)

        # Get fix maps for all times
        if self.n_output_fixs>=1:
            fixs = []
            for i in range(self.timesteps):
                fix_names_t = self.fix_filenames[i][idx * self.batch_size : (idx + 1) * self.batch_size]
                fix_t = preprocess_fixmaps(fix_names_t, self.map_size[0], self.map_size[1], fix_as_mat=self.fix_as_mat, fix_key=self.fix_key)
                fixs.append(fix_t)

        if self.augment:
            seq_det = self.seq.to_deterministic()
            images = seq_det.augment_images(images)
            for i in range(len(maps)):
                if self.n_output_maps>=1:
                    maps[i] = seq_det.augment_heatmaps(maps[i])
                if self.n_output_fixs>=1:
                    fixs[i] = seq_det.augment_heatmaps(fixs[i])

        if self.mode == 'singlestream':
            # Returns a list of n_output_maps+n_output_fixs elements. Each element is a 5D tensor: (bs, timesteps, r, c, 1)
            outs = []
            if self.n_output_maps>=1:
                maps_with_time = np.zeros((len(batch_imgs),self.timesteps,self.map_size[0],self.map_size[1],1))
                for i in range(self.timesteps):
                    # print("maps size", maps[i].shape)
                    maps_with_time[:,i,...] = maps[i]
                outs.extend([maps_with_time]*self.n_output_maps)

            if self.n_output_fixs>=1:
                fixs_with_time = np.zeros((len(batch_imgs),self.timesteps,self.map_size[0],self.map_size[1],1))
                for i in range(self.timesteps):
                    fixs_with_time[:,i,...] = fixs[i]
                outs.extend([fixs_with_time]*self.n_output_fixs)

        elif self.mode == 'multistream_concat':
            # returns a list of t elements: [ [maps_t1,fix_t1], [maps_t2,fix_t2] , [maps_t3,fix_t3], ...]
            outs=[]
            for i in range(self.timesteps):
                outs.append(np.concatenate([np.expand_dims(maps[i],axis=1),np.expand_dims(fixs[i],axis=1)], axis=1))
                # print('len(outs) multistream concat:',len(outs))

        elif self.mode == 'multistream_full':
            # returns a list of size timestep*losses. If 2 losses maps, 1 loss fix, 2 timesteps, we have: [m1, m1, m2, m2, fix1, fix2]
            outs = []
            if self.n_output_maps >= 1:
                for i in range(self.timesteps):
                    outs.extend([maps[i]]*self.n_output_maps)
            if self.n_output_fixs >= 1:
                for i in range(self.timesteps):
                    outs.extend([fixs[i]]*self.n_output_fixs)
        
        if self.return_names:
            return images, outs, batch_imgs
        return images, outs


    def on_epoch_end(self):
        if self.shuffle:
            idxs = list(range(len(self.img_filenames)))
            np.random.shuffle(idxs)
            self.img_filenames = self.img_filenames[idxs]
            for i in range(len(self.map_filenames)):
                self.map_filenames[i] = self.map_filenames[i][idxs]
                if self.fix_filenames is not None:
                    self.fix_filenames[i] = self.fix_filenames[i][idxs]

class SalImpGenerator(Sequence):

    def __init__(
        self,
        img_size,
        map_size,
        img_filenames,
        imp_filenames,
        fix_filenames=None,
        batch_size=1,
        shuffle=True,
        augment=False,
        n_output_maps=1,
        concat_fix_and_maps=True,
        fix_as_mat=False,
        fix_key=""):

        print('Instantiating SalImpGenerator. Number of files received: %d. Batch size: %d. Image size: %s. Map size: %s. Augmentation: %d' %
              (len(img_filenames), batch_size, str(img_size), str(map_size), augment ))

        self.img_filenames = np.array(img_filenames)
        self.imp_filenames = np.array(imp_filenames)
        self.batch_size = batch_size
        self.img_size = img_size
        self.map_size = map_size
        self.shuffle = shuffle
        self.augment = augment
        self.n_output_maps = n_output_maps
        self.concat_fix_and_maps = concat_fix_and_maps
        self.fix_as_mat=fix_as_mat
        self.fix_key = fix_key

        if fix_filenames is not None:
            self.fix_filenames = np.array(fix_filenames)
        else:
            self.fix_filenames = None

        if augment:
            sometimes = lambda aug: iaa.Sometimes(0.4, aug)
            self.seq = iaa.Sequential([
                    sometimes(iaa.CropAndPad(px=(0, 20))), # crop images from each side by 0 to 16px (randomly chosen)
                    iaa.Fliplr(0.5), # horizontally flip 50% of the images
                    sometimes(iaa.CoarseDropout(p=0.1, size_percent=0.05)),
                    sometimes(iaa.Affine(rotate=(-15, 15)))
                ], random_order=True)


        if shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.img_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.imp_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]

#         print('img names in this batch:', batch_x)
#         print('imp names in this batch:', batch_y)

        images = preprocess_images(batch_x, self.img_size[0], self.img_size[1])
        maps = preprocess_maps(batch_y, self.map_size[0], self.map_size[1])

        if self.fix_filenames is not None:
            fixs = preprocess_fixmaps(
                self.fix_filenames[idx * self.batch_size : (idx + 1) * self.batch_size],
                self.map_size[0],
                self.map_size[1],
                fix_as_mat=self.fix_as_mat,
                fix_key=self.fix_key)

        if self.augment:
            seq_det = self.seq.to_deterministic()
            images = seq_det.augment_images(images)
            maps = seq_det.augment_heatmaps(maps)
            if self.fix_filenames is not None:
                fixs = seq_det.augment_heatmaps(fixs)

        if self.fix_filenames is not None and self.concat_fix_and_maps:
            outs = np.concatenate([np.expand_dims(maps,axis=1),np.expand_dims(fixs,axis=1)], axis=1)
            if self.n_output_maps >1:
                outs = [outs]*self.n_output_maps
        else:
            if self.n_output_maps ==1:
                if self.fix_filenames is not None:
                    outs=[maps,fixs]
                else:
                    outs=maps
            else:
                outs = [maps]*self.n_output_maps
                if self.fix_filenames is not None:
                    outs.append(fixs)

#         print('generator: len(outs) should be 3:', len(outs))
#         print('generator: outs[0].shape (should be bs,2,r,c,1):', outs[0].shape)
#         print('generator: outs[0][0][0].shape (should be first map of batch)',outs[0][0][0].shape)

        return images, outs


    def on_epoch_end(self):
        if self.shuffle:
            idxs = list(range(len(self.img_filenames)))
            np.random.shuffle(idxs)
            self.img_filenames = self.img_filenames[idxs]
            self.imp_filenames = self.imp_filenames[idxs]
            if self.fix_filenames is not None:
                self.fix_filenames = self.fix_filenames[idxs]

def eval_generator(
    img_filenames,
    map_filenames,
    fixmap_filenames,
    fixcoord_filenames,
    inp_size,
    fix_as_mat=False,
    fix_key="",
    fixcoord_filetype='mat'):
    """
        Returns tuples img, heatmap, fixmap, fix_coords to be used for data eval

        img_filenames, map_filesnames, fixmap_filenames should a length-n list where
        n is the number of timestamps

        heatmap, fixmap, fixcoords are all also length-n

    """
    assert len(map_filenames) == len(fixmap_filenames)
    n_times = len(map_filenames)
    n_img = len(map_filenames[0])
    for i in range(n_img):
        imgs = []
        maps = []
        fixmaps = []
        fixcoords = []
        img = preprocess_images([img_filenames[i]], inp_size[0], inp_size[1])
        for t in range(n_times):
            # load the image
            #img = cv2.imread(img_filenames[i])
            map_ = cv2.imread(map_filenames[t][i], cv2.IMREAD_GRAYSCALE)
            mapshape = map_.shape
            if fix_as_mat:
                fixmap = preprocess_fixmaps(
                    [fixmap_filenames[t][i]],
                    mapshape[0],
                    mapshape[1],
                    fix_as_mat=fix_as_mat,
                    fix_key=fix_key
                )
                fixmap = np.squeeze(fixmap)
            else:
                fixmap = cv2.imread(fixmap_filenames[t][i], 0)
            if fixcoord_filenames:
                assert len(fixcoord_filenames) == n_times
                if fixcoord_filetype == 'mat':
                    fixdata = scipy.io.loadmat(fixcoord_filenames[t][i])
                    resolution = fixdata["resolution"][0]
                    #assert resolution[0] == img.shape[1] and resolution[1] == img.shape[2]
                    fix_coords_all_participants = fixdata["gaze"]
                    all_fixations = []
                    for participant in fix_coords_all_participants:
                        all_fixations.extend(participant[0][2])
                else:
                    raise RuntimeError("fixcoord filetype %s is unsupported" % fixcoord_filetype)
            else:
                all_fixations = None
            imgs.append(img)
            maps.append(map_)
            fixmaps.append(fixmap)
            fixcoords.append(all_fixations)
        yield (imgs, maps, fixmaps, fixcoords)