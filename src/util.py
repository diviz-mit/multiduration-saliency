from multiduration_models import md_sem, sam_resnet_md
from singleduration_models import sam_resnet_new
from losses_keras2 import loss_wrapper, kl_time, cc_time, nss_time, cc_match

MODELS = {
    'md-sem': (md_sem, 'singlestream'),
    'sam-md': (sam_resnet_md, 'singlestream'), 
    'sam-resnet': (sam_resnet_new, 'simple')
}

LOSSES = {
    'kl': (kl_time, 'heatmap'),
    'cc': (cc_time, 'heatmap'),
    'nss': (nss_time, 'fixmap'),
    'ccmatch': (cc_match, 'heatmap')
}

def get_model_by_name(name): 
    """ Returns a model and a string indicating its mode of use."""
    if name not in MODELS: 
        allowed_models = list(MODELS.keys())
        raise RuntimeError("Model %s is not recognized. Please choose one of: %s" % (name, ",".join(allowed_models)))
    else: 
        return MODELS[name]

def get_loss_by_name(name, out_size): 
    """Gets the loss associated with a certain name. 

    If there is no custom loss associated with name `name`, returns the string
    `name` so that keras can interpret it as a keras loss.
    """
    if name not in LOSSES: 
        print("WARNING: foudn no custom loss with name %s, defaulting to a string." % name)
        return name, 'heatmap'
    else: 
        loss, out_type = LOSSES[name]
        loss = loss_wrapper(loss, out_size)
        return loss, out_type

def create_losses(loss_dict, out_size): 
    """Given a dictionary that maps loss names to weights, returns loss functions and weights in the correct order. 

    By convention, losses that take in a heatmap (as opposed to a fixmap) come first in the array of losses. This function enforces that convention.

    This function looks up the correct loss function by name and outputs the correct functions, ordering, and weights to pass to the model/generator.
    """
    l_hm = []
    l_hm_w = [] 
    l_fm = []
    l_fm_w = []
    lstr = ""
    for lname, wt in loss_dict.items(): 
        loss, out_type = get_loss_by_name(lname, out_size)    
        if out_type == 'heatmap': 
            l_hm.append(loss)
            l_hm_w.append(wt)
        else: 
            l_fm.append(loss)
            l_fm_w.append(wt)
        lstr += lname + str(wt)

    l = l_hm + l_fm
    lw = l_hm_w + l_fm_w
    n_heatmaps = len(l_hm)
    return l, lw, lstr, n_heatmaps    
