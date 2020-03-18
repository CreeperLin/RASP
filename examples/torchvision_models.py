import torch
import torch.nn as nn
from torchvision import models
import argparse
import rasp

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith('__') and not 'inception' in name
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
    device = torch.device('cuda')
    torch.cuda.set_device(dev[0])
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    return device, dev

def main():
    parser = argparse.ArgumentParser(description='profile torchvision models')
    parser.add_argument('-d','--device',type=str,default='all', help='device ids')
    parser.add_argument('-v','--verbose',action='store_true', help='verbose msg')
    parser.add_argument('-t','--timing',action='store_true', help='enable timing')
    parser.add_argument('-i','--input',type=str,default='(1, 3, 224, 224)', help='input shape')
    args = parser.parse_args()

    device, devlist = get_device(args.device)

    config = rasp.set_config({
        'profile': {
            'batch_size': 1,
            'num_batches': 50,
            'warmup_batches': 5,
            'timing_max_depth': -1,
            'compute_max_depth': -1,
            'verbose': args.verbose,
        },
    })

    print('%s | %s | %s | %s | %s' % ('Model', 'Params', 'FLOPs', 'latency', 'FLOPS'))
    print('---|---|---|---|---')

    fields = ['name', 'type', 'in_shape', 'out_shape',
            'params', 'lat', 'net_lat', 'lat[%]', 'flops', 'FLOPS', 'mem_rw', 'dev_mem_alloc', 'dev_mem_delta']

    input_shape = tuple(eval(args.input))
    inputs = torch.randn(input_shape, device=device)
    
    for i, name in enumerate(model_names):
        model = models.__dict__[name]().to(device=device)
        summary, df = rasp.stat(model, inputs=inputs, device=device, report_fields=fields, timing=args.timing, print_only=False)
        if args.verbose:
            print(summary)
        total_f = df.tail(1)
        rasp.profile_off(model)
        flops, params = total_f.flops[0], total_f.params[0]
        if args.timing:
            latency = total_f.lat[0]
            FLOPS = total_f.FLOPS[0]
        else:
            latency = 0
            FLOPS = 0
        print('%s | %s | %s | %s | %s' % (name, rasp.round_value(params),
            rasp.round_value(flops), rasp.round_value(latency), rasp.round_value(FLOPS)))

if __name__ == '__main__':
    main()