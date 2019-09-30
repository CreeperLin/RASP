# -*- coding: utf-8 -*-
import sys
import importlib
_FRONTEND = 'pytorch'

required_entries = [
    'get_stdtype',
    'reg_stats_node',
    'hook_compute',
    'hook_timing',
    'unhook_compute',
    'unhook_timing',
    'pre_run',
    'run',
    'get_random_data',
]

def load_frontend(config):
    global _FRONTEND
    try:
        _FRONTEND = config.frontend.type
    except:
        pass

    if _FRONTEND == 'pytorch':
        module_name = '.torch_frontend'
        package = 'rasp.frontend'
    elif _FRONTEND == 'tf':
        module_name = '.tf_frontend'
        package = 'rasp.frontend'
    else:
        module_name = _FRONTEND
        package = config.frontend.package
    
    try:
        frontend_module = importlib.import_module(module_name, package)
        entries = frontend_module.__dict__
        for e in required_entries:
            if e not in entries:
                raise ValueError('frontend missing required entry :' + e)
        namespace = globals()
        for k, v in entries.items():
            namespace[k] = v
        sys.stderr.write('RASP: using ' + _FRONTEND + ' frontend\n')
    except ImportError as e:
        raise ValueError('Unable to import frontend : {} {}'.format(_FRONTEND, str(e)))

def frontend():
    return _FRONTEND