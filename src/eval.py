import numpy as np
from tqdm import tqdm_notebook as tqdm
import tqdm
import scipy.ndimage
import matplotlib.pyplot as plt
import scipy.stats
from keras.utils import Sequence
from sal_imp_utilities import *
#from eval_saliconeval import *

def visualize_samples(model, gen, times):
    images, maps = gen.__getitem__(np.random.randint(len(gen_val)))

    preds = model.predict(images)

    times = [500, 3000, 5000]
    n_times = len(preds)
    assert len(times) == n_times
    batch_sz = len(preds[0])
    copy=0
    # n_col, n_row = n_times + 2, batch_sz

    # plt.figure(figsize=[16,10*batch_sz])

    for batch in range(batch_sz):
        plt.imshow(reverse_preprocess(images[batch]))
        plt.title("original image %d" % batch)
        plt.show()

        plt.figure(figsize=[16, 10])
        n_row=n_times
        n_col=2

        for time in range(n_times):

    #         plt.subplot(n_row, n_col, batch*n_col+1)
    #         plt.imshow(reverse_preprocess(images[batch]))
    #         plt.title('Original')

            plt.subplot(n_row,n_col,time*n_col+1)
            plt.imshow(maps[time][batch][copy][:, :, 0])
            plt.title('Gt %dms' % times[time])

            plt.subplot(n_row,n_col,time*n_col+2)
            plt.imshow(preds[time][batch][copy][:, :, 0])
            plt.title('Prediction %dms' % times[time])

        plt.show()

    # plt.show()

def rmse(gr_truth, predicted):
    errors = gr_truth - predicted
    errors = errors**2
    rmse = np.sqrt(np.mean(errors))

    return rmse

def r2(gr_truth, predicted):

    truth_mean = np.mean(gr_truth)

    ssres = np.sum((predicted - gr_truth)**2)
    sstot = np.sum((gr_truth - truth_mean)**2)

    return 1 - ssres/sstot

def cc_npy(gt, predicted):
    M1 = np.divide(predicted - np.mean(predicted), np.std(predicted))
    M2 = np.divide(gt - np.mean(gt), np.std(gt))
    ret = np.corrcoef(M1.reshape(-1),M2.reshape(-1))[0][1]
    return ret

def nss_npy(gt_locs, predicted_map):
    assert gt_locs.shape == predicted_map.shape, 'dim missmatch in nss_npy: %s vs %s' % (gt_locs.shape, predicted_map.shape)
    predicted_map_norm = (predicted_map - np.mean(predicted_map))/np.std(predicted_map)
    dot = predicted_map_norm * gt_locs
    N = np.sum(gt_locs)
    ret = np.sum(dot)/N
    return ret

def kl_npy(gt, predicted):
    predicted = predicted/np.max(predicted)
    gt = gt/np.sum(gt)
    predicted = predicted/np.sum(predicted)
    kl_tensor = gt * np.log(gt / (predicted+1e-7) +1e-7)
    return np.sum(kl_tensor)

def sim_npy(gt, predicted):
    # Sum of min between distributions at each pixel
    gt = (gt-np.min(gt))/(np.max(gt)-np.min(gt))
    gt = gt/np.sum(gt)
    predicted = (predicted-np.min(predicted))/(np.max(predicted)-np.min(predicted))
    predicted = predicted/np.sum(predicted)
    diff = np.minimum(gt, predicted)
    return np.sum(diff)

def emd_npy(gt, predicted):
    gt = (gt-np.min(gt))/(np.max(gt)-np.min(gt))
    gt = gt/np.sum(gt)
    predicted = (predicted-np.min(predicted))/(np.max(predicted)-np.min(predicted))
    predicted = predicted/np.sum(predicted)

    gt_flat = gt.flatten()
    pred_flat = predicted.flatten()

    return scipy.stats.wasserstein_distance(gt_flat, pred_flat)

def predict_and_save(model, test_img, inp_size, savedir, mode='multistream_concat', blur=False, test_img_base_path="", ext="png"):
    # if test_img_base_path is specified, then preserves the original
    # nested structure of the directory from which the stuff is pulled
    c=0
    if blur:
        print('BLURRING PREDICTIONS')
        if 'blur' not in savedir:
            savedir = savedir+'_blur'
    else:
        print('NOT BLURRING PREDICTIONS')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for imfile in tqdm.tqdm(test_img):
        batch = 0
        time = 0
        map_idx = 0
        gt_shape = Image.open(imfile).size[::-1]
        img = preprocess_images([imfile], inp_size[0], inp_size[1])
        preds = model.predict(img)
        if mode == 'multistream_concat':
            p = preds[time][batch][map_idx][:, :, 0]
        elif mode == 'simple':
            p = preds[0][batch][:,:,0]
        elif mode == 'singlestream':
            p = preds[0][batch][time][:,:,0]
        else:
            raise ValueError('Unknown mode')
        p = postprocess_predictions(p, gt_shape[0], gt_shape[1], blur, normalize=False)
        p_norm = (p-np.min(p))/(np.max(p)-np.min(p))
        p_img = p_norm*255
        hm_img = Image.fromarray(np.uint8(p_img), "L")

        imname = os.path.splitext(os.path.basename(imfile))[0] + "." + ext
        if test_img_base_path:
            relpath = os.path.dirname(imfile).replace(test_img_base_path, "")
            relpath = os.path.join(savedir, relpath)
            if not os.path.exists(relpath):
                os.makedirs(relpath)
            savepath = os.path.join(relpath, imname)
        else:
            savepath = os.path.join(savedir, imname)
        hm_img.save(savepath)

def predict_and_save_md(model, test_img, inp_size, savedir, mode='multistream_concat', blur=False, test_img_base_path="", times=[500, 3000, 5000], ext="png"):
    # if test_img_base_path is specified, then preserves the original
    # nested structure of the directory from which the stuff is pulled
    c=0
    if blur:
        print('BLURRING PREDICTIONS')
        if 'blur' not in savedir:
            savedir = savedir+'_blur'
    else:
        print('NOT BLURRING PREDICTIONS')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for imfile in tqdm.tqdm(test_img):
        # if c%100 == 0:
        #     print(c)
        # c+=1
        batch = 0
        time = 0
        map_idx = 0
        gt_shape = Image.open(imfile).size[::-1]
        img = preprocess_images([imfile], inp_size[0], inp_size[1])
        preds = model.predict(img)
        for time, timelen in enumerate(times): 
            if mode == 'multistream_concat':
                p = preds[time][batch][map_idx][:, :, 0]
            elif mode == 'simple':
                p = preds[0][batch][:,:,0]
            elif mode == 'singlestream':
                p = preds[0][batch][time][:,:,0]
            else:
                raise ValueError('Unknown mode')
            p = postprocess_predictions(p, gt_shape[0], gt_shape[1], blur, normalize=False)
            p_norm = (p-np.min(p))/(np.max(p)-np.min(p))
            p_img = p_norm*255
            hm_img = Image.fromarray(np.uint8(p_img), "L")

            imname = os.path.splitext(os.path.basename(imfile))[0] + "_" + str(timelen) + "." + ext
#            if test_img_base_path:
#                relpath = os.path.dirname(imfile).replace(test_img_base_path, "")
#                relpath = os.path.join(savedir, relpath)
#                #relpath = os.path.splitext(relpath)[0] + ".png"
#                if not os.path.exists(relpath):
#                    os.makedirs(relpath)
#                savepath = os.path.join(relpath, imname)
            savepath = os.path.join(savedir, imname)
            hm_img.save(savepath)



def calculate_metrics(p, gt_map=None, gt_fix_map=None, gt_fix_points=None):
    '''Calculates meaningful metrics for saliency given a single predicted map, its corresponding
    ground truth map (2D real valued np array), ground truth fixation map (2D binary np array),
    and ground truth fixation points (list of [x,y] coordinates corresponding to the fixation positions, 1 indexed)

    Inputs
    ------
    p: real valued 2D np array. Saliency map predicted by the model.
    gt_map: real valued 2D np array. Ground truth saliency map.
    gt_fix_map: binary 2D np array. Ground truth fixation map.
    gt_fix_points: list of [x,y] coords. 1 indexed.

    Returns
    -------
    metrics: dictionary of metrics. The values of the metrics are encapsulated
        in a list element to play nicely with other functions in this file.
    '''
    metrics = {}

    p_norm = (p-np.min(p))/(np.max(p)-np.min(p))
    if np.max(gt_map)>1:
        gt_map = np.array(gt_map, dtype=np.float32)/255.

    # print('p_norm.shape, np.min(p_norm), np.max(p_norm), np.mean(p_norm)',p_norm.shape, np.min(p_norm), np.max(p_norm), np.mean(p_norm))
    # print('gt_map.shape, np.min(gt_map), np.max(gt_map), np.mean(gt_map)',gt_map.shape, np.min(gt_map), np.max(gt_map), np.mean(gt_map))

    if gt_map is not None:
        metrics['R2'] = [r2(gt_map, p_norm)]
        metrics['RMSE'] = [rmse(gt_map, p_norm)]
        metrics['CC'] = [cc_npy(gt_map, p_norm)]
        metrics['KL'] = [kl_npy(gt_map, p_norm)]
        metrics['SIM'] = [sim_npy(gt_map, p_norm)]
    if gt_fix_map is not None:
        metrics['NSS'] = [nss_npy(gt_fix_map, p_norm)]
#    if gt_fix_points is not None:
#        metrics['NSS (saliconeval)'] = [nss_saliconeval(gt_fix_points, p_norm)]
#        metrics['AUC'] = [auc_saliconeval(gt_fix_points, p)]

    return metrics

def get_stats_multiduration(model, gen_eval, mode='multistream_concat', blur=False,
                            start_at=False, compare_across_times=True, n_times=3,
                            return_top_bot_n=False, metrics_for_top_bot_n=['CC','KL']):
    '''Function to calculate metrics from a multiduration model. Calculates both the average over timesteps, as well
    as the per-timestep metrics. Can work in different modes:

    singlestream: assumes that both the model and the generator output a list of k elements (one per loss), where each element
    is a tensor of shape (bs, t, r, c, 1).

    multistream_concat: assumes that the model and generator both output a list of t elements, corresponding to
    t timesteps. Each one of those elements is a 5D tensor where the heatmap and fixation map (generator)
    or two copies of the pred map (model) are concatenated along the second dimension,
    resulting in a shape of (bs, 2, r, c, 1). To access the first fixmap from this 5D tensor, one would get
    the slice: (0,1,:,:,:). This mode should be used for 3stream models in concat mode, such as simple DCNNs
    with one output per timestep.
    '''
    # Setting starting point
    if start_at:
        gen = (next(gen_eval) for _ in range(start_at))
    else:
        gen = gen_eval

    # set up metric tracking code
    all_m = {}
    m_by_time = {i: {} for i in range(n_times)}

    batch = 0
    top_n = {}
    bot_n = {}
    idx = 0

    # used for comparing cc across groups
    if compare_across_times:
        combos = {}
        for t_lower in range(n_times):
            for t_upper in range(t_lower +1, n_times):
                if t_lower not in combos:
                    combos[t_lower] = {}
                if t_upper not in combos[t_lower]:
                    combos[t_lower][t_upper] = []

    for dat in tqdm.tqdm_notebook(gen):
        batch = 0
        imgs, gt_map_batch, gt_fix_map_batch, gt_fix_points_batch = dat
        prediction = model.predict(imgs[0])

        hms_by_time = {}
        mt = {}
        for t in range(n_times):
            if mode == 'singlestream':
                pred_batch = prediction[0]
            else:
                raise ValueError('Mode not implemented: %s' % mode)
            
            p = pred_batch[batch][t]

            gt_map = gt_map_batch[t]
            gt_fix = gt_fix_map_batch[t]

            gt_size = gt_map.shape
            p = postprocess_predictions(p, gt_size[0], gt_size[1], blur, normalize=False)
#             print('np.max(p),np.min(p)',np.max(p),np.min(p))
            hms_by_time[t] = (p-np.min(p))/(np.max(p)-np.min(p))

            gt_fix_points = gt_fix_points_batch[t] if gt_fix_points_batch else None
            assert p.shape == gt_map.shape, "prediction and ground truth should have same dimensions"

            m = calculate_metrics(p, gt_map, gt_fix, gt_fix_points)

            # If first pass, define metric dictionary as the dict returned from calculate_metrics
            for k,v in m.items():
                all_m[k] = all_m.get(k, []) + v # append list to list or int to int
                m_by_time[t][k] = m_by_time[t].get(k, []) + v # append list to list or int to int
            mt[t] = m
        # calculate pairwise
        if compare_across_times:
            for t_lower, others in combos.items():
                for t_higher in others:
                    combos[t_lower][t_higher].append(cc_npy(hms_by_time[t_lower], hms_by_time[t_higher]))

        if return_top_bot_n:
            for me in metrics_for_top_bot_n:
                if me not in top_n:
                    top_n[me] = []
                if len(top_n.get(me,[])) < return_top_bot_n:
                    item = (np.mean(all_m[me][-n_times:]), idx, mt, imgs[0], hms_by_time, gt_map_batch, [combos[0][1][-1],combos[0][2][-1],combos[1][2][-1]] )
                    top_n[me].append(item)
                    cur_min_idx = np.argmin([it[0] for it in top_n[me]])
                elif np.mean(all_m[me][-n_times:]) > top_n[me][cur_min_idx][0]:
                    del top_n[me][cur_min_idx]
                    item = (np.mean(all_m[me][-n_times:]), idx, mt, imgs[0], hms_by_time, gt_map_batch, [combos[0][1][-1],combos[0][2][-1],combos[1][2][-1]])
                    top_n[me].append(item)
                    cur_min_idx = np.argmin([it[0] for it in top_n[me]])
                if me not in bot_n:
                    bot_n[me] = []
                if len(bot_n.get(me,[])) < return_top_bot_n:
                    item = (np.mean(all_m[me][-n_times:]), idx, mt, imgs[0], hms_by_time, gt_map_batch, [combos[0][1][-1],combos[0][2][-1],combos[1][2][-1]])
                    bot_n[me].append(item)
                    cur_max_idx = np.argmax([it[0] for it in bot_n[me]])
                elif np.mean(all_m[me][-n_times:]) < bot_n[me][cur_max_idx][0]:
                    del bot_n[me][cur_max_idx]
                    item = (np.mean(all_m[me][-n_times:]), idx, mt, imgs[0], hms_by_time, gt_map_batch, [combos[0][1][-1],combos[0][2][-1],combos[1][2][-1]])
                    bot_n[me].append(item)
                    cur_max_idx = np.argmax([it[0] for it in bot_n[me]])
        idx+=1

    print("Overall metrics:")
    for k,v in all_m.items():
       print("\t", k,':', np.mean(v))

    print()
    print("Metrics by time:")
    for t, m in m_by_time.items():
        print("\tTime %d" % t)
        for k, v in m.items():
            print("\t\t", k, ":", np.mean(v))

    print()
    if compare_across_times:
        print("CC across time groups:")
        for t_lower, others in combos.items():
            for t_higher, v in others.items():
                print("CC for times %d and %d" % (t_lower, t_higher), np.mean(v))

    ret = [all_m, m_by_time]
    if compare_across_times: 
        ret.append(combos)
    if return_top_bot_n: 
        ret.extend([top_n, bot_n])
    return ret
    
def get_stats_oneduration(model, gen_eval, mode='multistream_concat', blur=False, start_at=False, t=0):
    ''' Function to calculate metrics from a model, based on the model object and a generator.
    This function will always assume that the model is trained on a dataset with only one timestep. If the
    model and generator given output multiple timesteps, the function will assume that all maps are the same
    for each timestep and will take the t-eth one.

    The function can operate in different modes, depending on the shape and form of the output
    of the generator and model:

    multistream_concat: assumes that the model and generator both output a list of t elements, corresponding to
    t timesteps. Each one of those elements is a 5D tensor where the heatmap and fixation map (generator)
    or two copies of the pred map (model) are concatenated along the second dimension,
    resulting in a shape of (bs, 2, r, c, 1). To access the first fixmap from this 5D tensor, one would get
    the slice: (0,1,:,:,:). This mode should be used for 3stream models in concat mode, such as simple DCNNs
    with one output per timestep.

    singlestream: assumes that the model and generator output a list of k 5D tensors. Each
    tensor matches one loss, and their shapes are (bs, time, r, c, 1). As this function considers that all t maps are
    the same, it will slice this tensor at (bs, ~t~, r, c, 1).

    single: For when the model doesn't deal with timesteps. The model should output a list of k
    elements, each corresponding ot a loss of the model. The generator should also output k ground truths,
    one for each loss.

    Inputs
    ------
    model: a Keras model.
    gen_eval: a geneartor with data to evaluate. Can be a Keras generator outputing imgs, maps and fixmaps or
    a python generator outputting imgs, maps, fixmaps and fixlocs (x,y).
    mode: str. mode to use.
    blur: bool. whether to blur the predictions before evaluating.
    start_at: int. Idx to start in the generator.
    t: int. tensor idx to get from the time dimension, if existing.

    Returns
    -------
    metrics: a dictionary of metrics, where for each metric, a list of values for each element in the set is available.
    '''

    c = 0
    first_pass = True

    # Setting starting point
    if start_at:
        gen = (next(gen_eval) for _ in range(start_at))
    else:
        gen = gen_eval

    # Iterating over generator
    metrics = {}
    for dat in tqdm.tqdm_notebook(gen):
#         if not c%10:
#             print(c)
#         c+=1

        ## Get ground truths
        if isinstance(gen, Sequence):   # If the generator is a Keras Sequence
            gt_fix_points_batch = None
            imgs, gt_set = dat
            if mode == 'multistream_concat':
                gt_map_batch = gt_set[t][:,0,...] #batch of gt_maps
                gt_fix_map_batch = gt_set[t][:,-1,...] #batch of gt_fix_maps
            if mode == 'singlestream':
                gt_map_batch = gt_set[0][:,t,...] #batch of gt_maps
                gt_fix_map_batch = gt_set[-1][:,t,...] #batch of gt_fix_maps
            elif mode == 'simple':
                gt_map_batch = gt_set[0]
                gt_fix_map_batch = gt_set[-1]
            else:
                raise ValueError('Unknown mode: '+str(mode))

        else:  # If the generator is a conventional python generator
            imgs, gt_map_batch, gt_fix_map_batch, gt_fix_points_batch = dat

        ## Get prediction batch
        prediction = model.predict(imgs)
        if mode == 'multistream_concat':
            pred_batch = prediction[t][:,0,...]
        elif mode == 'singlestream':
            pred_batch = prediction[0][:,t,...]
        elif mode == 'simple':
            pred_batch = prediction[0]
            if not isinstance(gen, Sequence):
                pred_batch = pred_batch[:,:,:,0]
        else:
            raise ValueError('Unknown mode: '+str(mode))

        for i in range(len(pred_batch)): # loop over batch
            # the 0 is to get rid of the copy
            p = pred_batch[i]
            gt_map = gt_map_batch[i]
            gt_fix = gt_fix_map_batch[i]
            p = p.squeeze()
            gt_map = gt_map.squeeze()
            gt_fix = gt_fix.squeeze()

            gt_size = gt_map.shape
            p = postprocess_predictions(p, gt_size[0], gt_size[1], blur, normalize=False)

            if gt_fix_points_batch:
                gt_fix_points = gt_fix_points_batch[i]
            else:
                gt_fix_points = None
            
            
            assert p.shape == gt_map.shape, "prediction and ground truth should have same dimensions, but are %s and %s" % (p.shape, gt_map.shape)

            m = calculate_metrics(p, gt_map, gt_fix, gt_fix_points)

            # If first pass, define metric dictionary as the dict returned from calculate_metrics
            for k,v in m.items():
                metrics[k] = metrics.get(k, []) + v # append list to list or int to int

    # Print results
    for k,v in metrics.items():
        print(k,':', np.mean(v))

    return metrics
