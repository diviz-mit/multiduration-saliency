from multiduration_models import md_sem, sam_resnet_md
from losses_keras2 import loss_wrapper, kl_time, cc_time, nss_time

MODELS = {
    'md-sem': (md_sem, 'singlestream'),
    'sam-md': (sam_resnet_md, 'multistream-concat')
}

LOSSES = {
    'kl': kl_time, 
    'cc': cc_time,
    'nss': nss_time,
}

def get_model_by_name(name): 
    """ Returns a model and a string indicating its mode of use."""
    if name not in MODELS: 
        allowed_models = list(MODELS.keys())
        raise RuntimeError("Model %s is not recognized. Please choose one of: %s" % (name, allowed_models.join(",")))
    else: 
        return MODELS[name]

def get_loss_by_name(name, out_size): 
    """Gets the loss associated with a certain name. 

    If there is no custom loss associated with name `name`, returns the string
    `name` so that keras can interpret it as a keras loss.
    """
    if name not in LOSSES: 
        return name
    else: 
        return loss_wrapper(LOSSES[name], out_size)
