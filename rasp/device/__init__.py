# -*- coding: utf-8 -*-
import sys
import importlib

_DEVICE = 'frontend'
required_entries = [
    'reset',
    'init',
    'run',
    'add_node',
]

def load_device(config):
    global _DEVICE
    _DEVICE = config.device.get('type', 'frontend')
    
    if _DEVICE == 'frontend':
        module_name = '.frontend_device'
        package = 'rasp.device'
    elif _DEVICE == 'regressor':
        module_name = '.regressor_device'
        package = 'rasp.device'
    else:
        module_name = '.'+_DEVICE
        package = config.device.package
    # Try and load external device.
    try:
        device_module = importlib.import_module(module_name, package)
        entries = device_module.__dict__
        for e in required_entries:
            if e not in entries:
                raise ValueError('Device missing required entry : ' + e)
        namespace = globals()
        for k, v in entries.items():
            namespace[k] = v
    except ImportError as e:
        raise ValueError('Unable to import device : {} {}'.format(_DEVICE, str(e)))

def device():
    return _DEVICE