# -*- coding: utf-8 -*-
import os
import logging
import torch

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger('rasp')

def get_torch_current_mem(dev=None):
    if dev == 'cuda':
        if not torch.cuda.is_available():
            logger.error('cuda unavailable')
            return 0
        gpu_mem = torch.cuda.memory_allocated()
        if torch.cuda.max_memory_allocated() != 0: return gpu_mem
    else:
        logger.error('unsupported device: {}'.format(dev))
        return 0


def get_torch_max_mem(dev=None):
    if dev == 'cuda':
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated()
            if max_mem != 0: return max_mem
    return get_torch_current_mem(dev)


class FrontendDevice():
    def __init__(self, frontend, device):
        super().__init__()
        self.base_mem = 0
        self.frontend = frontend
        self.device = None
        self.set_device(device)

    def reset(self):
        self.base_mem = 0
    
    def get_current_mem(self):
        if self.device == 'cpu':
            if psutil is None:
                logger.error('psutil not installed')
                return 0
            return psutil.Process(os.getpid()).memory_info().rss
        if self.frontend == 'torch':
            return get_torch_current_mem(self.device)
    
    def get_max_mem(self):
        if self.device == 'cpu':
            if not psutil is None:
                return psutil.Process(os.getpid()).memory_info().rss
            else:
                return self.get_current_mem()
        if self.frontend == 'torch':
            return get_torch_current_mem(self.device)
    
    def get_synchronize(self):
        if self.device == 'cpu':
            return lambda: None
        if torch.cuda.is_available():
            return torch.cuda.synchronize
        else:
            return lambda: None
    
    def set_device(self, dev):
        if dev is None:
            dev = 'cpu'
        self.device = dev


device = None


def init(*args, **kwargs):
    global device
    device = FrontendDevice(*args, **kwargs)


def get_current_mem():
    return device.get_current_mem()


def get_max_mem():
    return device.get_max_mem()


def get_base_mem():
    return device.base_mem


def set_device(dev):
    device.set_device(dev)


def get_synchronize():
    return device.get_synchronize()


def reset():
    device.reset()


def add_node(node):
    pass


def run_node(node):
    pass


def run():
    return 0
