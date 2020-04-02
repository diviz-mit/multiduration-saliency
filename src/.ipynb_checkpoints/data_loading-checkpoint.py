import os
import numpy as np
import json

VALID_FILE_KEYS = ['img_files_train', 'map_files_train', 'fix_files_train', 'img_files_val', 'map_files_val', 'fix_files_val', 'img_files_test', 'map_files_test', 'fix_files_test', 'fixcoords_files_train', 'fixcoords_files_val', 'fixcoords_files_test']

# keys that change format for single and multi-duration datasets
DIFFERENT_FORMAT_KEYS = ['map_files_train', 'fix_files_train', 'map_files_val', 'fix_files_val', 'map_files_test', 'fix_files_test', 'fixcoords_files_train', 'fixcoords_files_val', 'fixcoords_files_test']

def load_datasets_singleduration(dataset, bp="../../predimportance_shared/datasets", verbose=True, search_multidur=True, attime=5000):
    ret = {}
    ret['fix_as_mat'] = False

    if dataset == 'salicon':
        print('Using SALICON')

        img_path_train = os.path.join(bp, 'salicon', 'train')
        imp_path_train = os.path.join(bp, 'salicon', 'train_maps')

        img_path_val = os.path.join(bp, 'salicon', 'val')
        imp_path_val = os.path.join(bp, 'salicon', 'val_maps')

        img_path_test = os.path.join(bp, 'salicon', 'test')

        fix_path_train = os.path.join(bp, 'salicon', 'train_fix_png')
        fix_path_val = os.path.join(bp, 'salicon', 'val_fix_png')

        fixcoords_path_train = os.path.join(bp, 'salicon', 'train_fix')
        fixcoords_path_val = os.path.join(bp, 'salicon', 'val_fix')

        ret['img_files_train'] = sorted([os.path.join(img_path_train, f) for f in os.listdir(img_path_train)])
        ret['map_files_train'] = sorted([os.path.join(imp_path_train, f) for f in os.listdir(imp_path_train)])

        ret['img_files_val'] = sorted([os.path.join(img_path_val, f) for f in os.listdir(img_path_val)])
        ret['map_files_val'] = sorted([os.path.join(imp_path_val, f) for f in os.listdir(imp_path_val)])

        ret['img_files_test'] = sorted([os.path.join(img_path_test, f) for f in os.listdir(img_path_test)])

        ret['fix_files_train'] = sorted([os.path.join(fix_path_train, f) for f in os.listdir(fix_path_train) if f.endswith('.png')])
        ret['fix_files_val'] = sorted([os.path.join(fix_path_val, f) for f in os.listdir(fix_path_val) if f.endswith('.png')])

        ret['fixcoords_files_train'] = sorted([os.path.join(fixcoords_path_train, f) for f in os.listdir(fixcoords_path_train)])
        ret['fixcoords_files_val'] = sorted([os.path.join(fixcoords_path_val, f) for f in os.listdir(fixcoords_path_val)])

    elif dataset == 'mit1003':
        print('Using MIT1003')
        ret['fix_as_mat']=False

        img_path = os.path.join(bp, "mit1003/ALLSTIMULI")
        imp_path = os.path.join(bp, 'mit1003/ALLFIXATIONMAPS')
        fix_path = os.path.join(bp, 'datasets/mit1003/ALLFIXATIONMAPS')

        imgs = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpeg')])
        maps = sorted([os.path.join(imp_path, f) for f in os.listdir(imp_path) if 'fixMap' in f and f.endswith('.jpg')])
        fixs = sorted([os.path.join(imp_path, f) for f in os.listdir(imp_path) if 'fixPts' in f and f.endswith('.jpg')])

        # Randomly shuffling mit1003
        np.random.seed(42)
        idxs = list(range(len(imgs)))
        np.random.shuffle(idxs)
        imgs = np.array(imgs)[idxs]
        maps = np.array(maps)[idxs]
        fixs = np.array(fixs)[idxs]

        ret['img_files_train'] = imgs[:903]
        ret['map_files_train'] = maps[:903]
        ret['fix_files_train'] = fixs[:903]
        ret['img_files_val'] = imgs[903:]
        ret['map_files_val'] = maps[903:]
        ret['fix_files_val'] = fixs[903:]
        
    elif dataset == 'mit300':
        img_path = os.path.join(bp, 'mit300')
        #img_path = '../../predimportance_shared/datasets/mit300'
        imgs = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg')])
        ret['img_files_test'] = imgs

    elif dataset == 'cat2000':
        print('Using CAT2000')
        ret['fix_as_mat']=True
        ret['fix_key']="fixLocs"

        # TODO: MAKE SURE THAT THE VAL SET IS ALWAYS THE SAME
        np.random.seed(42)
        img_path = os.path.join(bp, 'cat2000', 'Stimuli')
        imp_path = os.path.join(bp, 'cat2000', 'FIXATIONMAPS')
        fix_path = os.path.join(bp, 'cat2000', 'FIXATIONLOCS')
        img_path_test = os.path.join(bp, 'cat2000', 'testStimuli')

        img_filenames_train = np.array([])
        imp_filenames_train = np.array([])
        fix_filenames_train = np.array([])
        img_filenames_val = np.array([])
        imp_filenames_val = np.array([])
        fix_filenames_val = np.array([])
        img_filenames_test = np.array([])

        for f in os.listdir(img_path):
            #print('Categ:',f)
            imgs = sorted([os.path.join(img_path, f, i) for i in os.listdir(os.path.join(img_path,f)) if i.endswith('.jpg')])
            maps = sorted([os.path.join(imp_path, f, i) for i in os.listdir(os.path.join(imp_path,f)) if i.endswith('.jpg')])
            fixs = sorted([os.path.join(fix_path, f, i) for i in os.listdir(os.path.join(fix_path,f)) if i.endswith('.mat')])

            idxs = list(range(len(imgs)))
            np.random.shuffle(idxs)
            imgs = np.array(imgs)[idxs]
            maps = np.array(maps)[idxs]
            fixs = np.array(fixs)[idxs]

            img_filenames_train = np.concatenate([img_filenames_train,imgs[:-10]], axis=None)
            img_filenames_val = np.concatenate([img_filenames_val,imgs[-10:]], axis=None)
            imp_filenames_train = np.concatenate([imp_filenames_train,maps[:-10]], axis=None)
            imp_filenames_val = np.concatenate([imp_filenames_val,maps[-10:]], axis=None)
            fix_filenames_train = np.concatenate([fix_filenames_train,fixs[:-10]], axis=None)
            fix_filenames_val = np.concatenate([fix_filenames_val,fixs[-10:]], axis=None)

        for f in os.listdir(img_path_test):
            new_files = sorted([os.path.join(img_path_test, f, i) for i in os.listdir(os.path.join(img_path_test, f)) if i.endswith('.jpg')])
            img_filenames_test = np.concatenate([img_filenames_test, new_files], axis=None)

        ret['img_files_train'] = img_filenames_train
        ret['img_files_val'] = img_filenames_val
        ret['map_files_train'] = imp_filenames_train
        ret['map_files_val'] = imp_filenames_val
        ret['fix_files_train'] = fix_filenames_train
        ret['fix_files_val'] = fix_filenames_val
        ret['img_files_test'] = img_filenames_test

    else: 
        if search_multidur: 
            print("WARNING: could not find single-duration dataset %s, searching multi-duration datasets" % dataset)
            ret = load_datasets_multiduration(dataset, times=[attime], bp=bp, test_splits=[0], verbose=False, search_singledur=False)
            print("Taking timestep %d from multiduration dataset" % attime)
            for key in DIFFERENT_FORMAT_KEYS: 
                if key in ret: 
                    ret[key] = ret[key][0]
        else: 
            raise RuntimeError("Could not find dataset %s" % dataset)

    if verbose: 
        print("Length of loaded files:")
        for key, val in ret.items(): 
            if key in VALID_FILE_KEYS:
                print(key, ":", len(val))
            else: 
                print(key, ":", val)

    return ret

def load_multiduration_data(img_folder, map_folder, fix_folder, times=[500,3000,5000]):
    '''
    Takes in a saliency heatmap groundtruth folder and a fixation groundtruth folder
    and searches for the specified exposure durations in `times`. The structure
    of the ground truth folders should be:
    map_or_fix_folder
        |_ time1
            |_ map1.png
            |_ ...
        |_ time2
        |_ time3
        |_ ...

    Inputs
    ------
    img_folder: string. Path to the folder containing input images. The names of
        these images should maimg_file://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-formattch the names of the corresponding fixations and heatmaps.
    map_folder: string. Path to the ground truth map folder. That folder should
        contain subfolders named as times and containing the images.
    fix_folder: string. Path to the ground truth fix folder. Idem as above.
    times: array of numbers or strings indicating what times to use. Should match
        the names of subfolders in map and fix folders.

    Returns:
    --------
    img_filenames: array of strings containing paths to images.
    map_filenames: array of length `times` containing arrays of strings corresponding to the paths to each image.
    fix_filenames: idem as above but for fixations.
    '''

    img_filenames = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg')])
    n_expected = len(img_filenames)

    map_filenames = []
    fix_filenames = []

    avlb_times = sorted([int(elt) for elt in os.listdir(map_folder)])
    print('avlb_times',avlb_times)
    
    for t in avlb_times:
        if t in times:
            print('APPENDING IMAGES FOR TIME:', t)
            t_map_filenames = sorted([os.path.join(map_folder, str(t), f) for f in os.listdir(os.path.join(map_folder, str(t))) if f.endswith('.png')])
            t_fix_filenames = sorted([os.path.join(fix_folder, str(t), f) for f in os.listdir(os.path.join(fix_folder, str(t))) if f.endswith('.png')])
            assert len(t_map_filenames) == n_expected
            assert len(t_fix_filenames) == n_expected

            map_filenames.append(t_map_filenames)
            fix_filenames.append(t_fix_filenames)

    # check that we have even numbers of everything
    return img_filenames, map_filenames, fix_filenames


def load_datasets_multiduration(dataset, times, bp="", test_splits=[0], verbose=True, search_singledur=True):
    ret = {}

    ret['fix_as_mat'] = False

    use_accum = False
    accum_suffix = "_accum" if use_accum else ""

    if dataset == 'codecharts':
        n_val=50

        img_path = os.path.join(bp, dataset, "raw_img" + accum_suffix)
        map_path = os.path.join(bp, dataset, "heatmaps" + accum_suffix)
        fix_path = os.path.join(bp, dataset, "fix_maps" + accum_suffix)

        data = load_multiduration_data(img_path, map_path, fix_path, times=times)

        img_files, map_files, fix_files = data

        # filter based on the split
        with open(os.path.join(bp, dataset, "splits.json")) as infile:
            splits = json.load(infile)
            non_test_splits = [s for s in splits.keys() if int(s) not in test_splits]
            # BY CONVENTION, split 1 is test
            train = [elt for i in non_test_splits for elt in splits[str(i)]]
            test = set([elt for i in test_splits for elt in splits[str(i)]])
            val = set(train[:n_val])
            train = set(train[n_val:])

        def _imname(imfile):
            return os.path.splitext(os.path.basename(imfile))[0]

        def img_in(imfile, imset):
            return _imname(imfile) in (_imname(elt) for elt in imset)

        ret['img_files_train'] = [f for f in img_files if img_in(f, train)]
        ret['img_files_val'] = [f for f in img_files if img_in(f, val)]
        ret['img_files_test'] = [f for f in img_files if img_in(f, test)]

        ret['map_files_train'] = []
        ret['map_files_val'] = []
        ret['map_files_test'] = []
        ret['fix_files_train'] = []
        ret['fix_files_val'] = []
        ret['fix_files_test'] = []

        for t in range(len(map_files)):
            for f in map_files[t]:
                bn = os.path.basename(f)
            ret['map_files_train'].append([f for f in map_files[t] if img_in(f, train)])
            ret['map_files_val'].append([f for f in map_files[t] if img_in(f, val)])
            ret['map_files_test'].append([f for f in map_files[t] if img_in(f, test)])

            ret['fix_files_train'].append([f for f in fix_files[t] if img_in(f, train)])
            ret['fix_files_val'].append([f for f in fix_files[t] if img_in(f, val)])
            ret['fix_files_test'].append([f for f in fix_files[t] if img_in(f, test)])

    elif dataset == "salicon_md" or dataset == "salicon_md_fixations":
        dsets = {}
        for mode in ["train", "val"]:
            img_path = os.path.join(bp, dataset, mode, "raw_img" + accum_suffix)
            map_path = os.path.join(bp, dataset, mode, "heatmaps" + accum_suffix)
            fix_path = os.path.join(bp, dataset, mode, "fix_maps" + accum_suffix)
            data = load_multiduration_data(img_path, map_path, fix_path, times=times)
            dsets[mode] = data

        ret['img_files_train'] = dsets["train"][0]
        ret['map_files_train'] = dsets["train"][1]
        ret['fix_files_train'] = dsets["train"][2]

        ret['img_files_val'] = dsets["val"][0]
        ret['map_files_val'] = dsets["val"][1]
        ret['fix_files_val'] = dsets["val"][2]

        # load the test images (only have images)
        img_path_test = os.path.join(bp, 'salicon', 'test')
        ret['img_files_test'] = sorted([os.path.join(img_path_test, f) for f in os.listdir(img_path_test)])

    else: 
        if search_singledur:
            print("WARNING: could not find multiduration dataset %s. Attempting to find a single-duration dataset." % dataset)
            ret = load_datasets_singleduration(dataset, bp=bp, verbose=False, search_multidur=False)
            
            _repeat = lambda x, rep: None if x is None else [x]*rep
            rep = len(times)

            if verbose: 
                print("Found single-duration dataset %s; making %d copies for compatability with multi-duration models" % (dataset, rep))
     
            for key in DIFFERENT_FORMAT_KEYS:
                if key in ret: 
                    ret[key] = rep*[ret[key]]
        else: 
            raise RuntimeError("Could not find multiduration dataset %s" % dataset)

    if verbose: 
        for key, val in ret.items(): 
            if "img" in key: 
                print("%s: %d" % (key, len(val)))
            elif key in VALID_FILE_KEYS:
                print(key, ":", [len(elt) for elt in val])
            else: 
                print(key, ":", val)
    
    return ret
