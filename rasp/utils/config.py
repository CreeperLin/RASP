# -*- coding: utf-8 -*-
from ..device import load_device
from ..frontend import load_frontend

class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value

default_conf = {
    'frontend':{
        'type': 'pytorch'
    },
    'profile': {
        'batch_size': 1,
        'num_batches': 5,
        'warmup_batches': 5,
        'timing_max_depth': -1,
        'compute_max_depth': -1,
        'verbose': False,
    },
    'device': {
        'type': 'frontend',
    }, 
    'analysis': {
        'regress_model': 'linear'
    }
}

class Config():
    def __getattr__(self, k):
        global global_config
        return global_config.__getattr__(k)
    
    def __setattr__(self, k, v):
        global global_config
        global_config.__setattr__(k, v)

    def __delattr__(self, k):
        global global_config
        global_config.__delattr__(k)

    def __init__(self):
        pass

CFG = Config()

def set_default_config():
    return set_config(default_conf)

def set_config(conf):
    global global_config
    global_config = Dotdict(conf)
    load_frontend(global_config)
    load_device(global_config)
    return CFG

def get_config():
    global global_config
    return global_config


global_config = None
set_default_config()