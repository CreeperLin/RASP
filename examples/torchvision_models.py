import torch
import torch.nn as nn
from torchvision import models
import argparse
import rasp

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__") and not "inception" in name
                     and callable(models.__dict__[name]))

def get_device(devdesc):
    torch.manual_seed(1)
    if devdesc == 'all':
        dev = list(range(torch.cuda.device_count()))
    elif devdesc != 'cpu':
        dev = [int(s) for s in devdesc.split(',')]
    if devdesc == 'cpu' or len(dev)==0:
        dev = None
        device = torch.device('cpu')
        return device, dev
    # set default gpu device id
    device = torch.device("cuda")
    torch.cuda.set_device(dev[0])
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    return device, dev

def main():
    parser = argparse.ArgumentParser(description='profile torchvision models')
    parser.add_argument('-d','--device',type=str,default="all", help="device ids")
    parser.add_argument('-v','--verbose',action='store_true', help="verbose msg")
    parser.add_argument('-t','--timing',action='store_true', help="enable timing")
    args = parser.parse_args()

    device, devlist = get_device(args.device)

    config = rasp.set_config({
        'frontend':{
            'type': 'pytorch'
        },
        'profile': {
            'batch_size': 1,
            'num_batches': 5,
            'warmup_batches': 5,
            'timing_max_depth': -1,
            'compute_max_depth': -1,
            'verbose': args.verbose,
        },
        'device': {
            'type': 'frontend',
        }, 
        'analysis': {
        }
    })

    print("%s | %s | %s" % ("Model", "Params", "FLOPs"))
    print("---|---|---")

    fields = ['name', 'type', 'in_shape', 'out_shape',
            'params', 'madds', 'net_lat', 'lat[%]', 'flops', 'mem_rw']

    input_shape = (1, 3, 224, 224)
    inputs = torch.randn(input_shape, device=device)
    
    for i, name in enumerate(model_names):
        model = models.__dict__[name]().to(device=device)
        stats = rasp.profile_compute_once(model, inputs=inputs)
        if args.timing: stats = rasp.profile_timing_once(model, inputs=inputs)
        summary, _ = rasp.summary_tape(stats, report_fields=fields)
        if args.verbose: print(summary)
        _, total_f = rasp.summary_node(stats, report_fields=fields)
        rasp.profile_off(model)
        total_ops, total_params = total_f['flops'], total_f['params']
        print("%s | %s | %s" % (name, rasp.round_value(total_params), rasp.round_value(total_ops)))

if __name__ == '__main__':
    main()