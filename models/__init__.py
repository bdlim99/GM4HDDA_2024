from models.base_arch import BaseArch
from models.dgcnn import DGCNN
from models.modules import MLP, MLP_PointCloud
import torch
import os 
# from omegaconf import OmegaConf

def get_model(cfg, *args, version=None, **kwargs):
    if 'model' in cfg:
        model_dict = cfg['model']
    elif 'arch' in cfg:
        model_dict = cfg
    else:
        raise ValueError(f'Invalid model configuration dictionary: {cfg}')

    name = model_dict["arch"]
    model = _get_model_instance(name)
    return model(**model_dict)

def _get_model_instance(name):
    try:
        return {
            "base_arch": get_base_arch
        }[name]
    except:
        raise ("Model {} not available".format(name))

def get_base_arch(**cfg):
    encoder_model = get_encoder(**cfg['encoder'])
    decoder_model = get_decoder(**cfg['decoder'])
    MED = cfg.get('MED', 0.0)
    k = cfg.get('k', 0.0)
    kl_reg = cfg.get('kl_reg', None)
    fm_reg = cfg.get('fm_reg', None)
    alpha_0 = cfg.get('alpha_0', 0.2)
    relaxation = cfg.get('relaxation', False)
    visualization = cfg.get('visualization', False)
    fm_val_show = bool(cfg.get('fm_val_show', False))
    latent_show = bool(cfg.get('latent_show', False))
    latent_G_show = bool(cfg.get('latent_G_show', False))
    use_identity = bool(cfg.get('use_identity', False))
    model = BaseArch(
        encoder_model, decoder_model, kl_reg=kl_reg, fm_reg=fm_reg, 
        alpha_0=alpha_0, relaxation=relaxation, sigma=MED * k, 
        fm_val_show=fm_val_show, visualization=visualization, 
        latent_show=latent_show, latent_G_show=latent_G_show,
        use_identity=use_identity)
    return model

def get_encoder(**kwargs):
    net_list = []
    for key in kwargs.keys():
        module_dict = kwargs[key]
        name = module_dict['arch']
        module_nn = _get_encoder_instance(name)
        module = module_nn(**module_dict)
        net_list.append(module)
    return net_list

def get_decoder(**kwargs):
    net_list = []
    for key in kwargs.keys():
        module_dict = kwargs[key]
        name = module_dict['arch']
        module_nn = _get_decoder_instance(name)
        module = module_nn(**module_dict)
        net_list.append(module)
    return net_list

def _get_encoder_instance(name):
    try:
        return {
            'dgcnn': get_dgcnn,
            'mlp': get_mlp,
        }[name]
    except:
        raise ("Model {} not available".format(name))

def _get_decoder_instance(name):
    try:
        return {
            'mlp_pointcloud': get_mlp_pointcloud,
        }[name]
    except:
        raise ("Model {} not available".format(name))

# encoder
def get_dgcnn(**cfg):
    model = DGCNN(**cfg)
    return model

# decoder
def get_mlp(**cfg):
    model = MLP(**cfg)
    return model

def get_mlp_pointcloud(**cfg):
    model = MLP_PointCloud(**cfg)
    return model

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    if "model" in cfg:
        model_name = cfg["model"]['arch']
    else:
        model_name = cfg['arch']
    
    model = get_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)

    return model, cfg