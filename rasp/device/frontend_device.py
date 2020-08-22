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
    base_mem = 0
    max_mem = 0
    frontend = None
    device = None
    
    @staticmethod
    def init(frontend, device):
        FrontendDevice.base_mem = 0
        FrontendDevice.max_mem = 0
        FrontendDevice.frontend = frontend
        FrontendDevice.device = None
        FrontendDevice.set_device(device)

    @staticmethod
    def reset():
        FrontendDevice.base_mem = FrontendDevice.get_current_mem()
    
    @staticmethod
    def get_base_mem():
        return FrontendDevice.base_mem

    @staticmethod
    def get_current_mem():
        if FrontendDevice.device == 'cpu':
            if psutil is None:
                logger.error('psutil not installed')
                return 0
            return psutil.Process(os.getpid()).memory_info().rss
        if FrontendDevice.frontend == 'pytorch':
            return get_torch_current_mem(FrontendDevice.device)
        return 0
    
    @staticmethod
    def get_max_mem():
        if FrontendDevice.device == 'cpu':
            if not psutil is None:
                return psutil.Process(os.getpid()).memory_info().rss
            else:
                return FrontendDevice.get_current_mem()
        if FrontendDevice.frontend == 'pytorch':
            return get_torch_current_mem(FrontendDevice.device)
    
    @staticmethod
    def reset_max_mem():
        if FrontendDevice.frontend == 'pytorch':
            torch.cuda.reset_max_memory_allocated()

    @staticmethod
    def get_synchronize():
        if FrontendDevice.device == 'cpu':
            return lambda: None
        if torch.cuda.is_available():
            return torch.cuda.synchronize
        else:
            return lambda: None
    
    @staticmethod
    def set_device(dev):
        if dev is None:
            dev = 'cpu'
        FrontendDevice.device = dev


init = FrontendDevice.init

get_current_mem = FrontendDevice.get_current_mem

get_max_mem = FrontendDevice.get_max_mem

reset_max_mem = FrontendDevice.reset_max_mem

get_base_mem = FrontendDevice.get_base_mem

set_device = FrontendDevice.set_device

get_synchronize = FrontendDevice.get_synchronize

reset = FrontendDevice.reset


def add_node(node):
    pass


def run_node(node):
    pass


def run():
    return 0
