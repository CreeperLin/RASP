import torch
from torchvision import models
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
    torch.backends.cudnn.benchmark = True
    return device, dev

def main():

    dev_desc = 'all'
    device, devlist = get_device(dev_desc)

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
            'verbose': True,
        },
        'module': {
        },
        'device': {
            'type': 'frontend',
        }, 
        'analysis': {
        }
    })

    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("---|---|---")

    for name in model_names:
        model = models.__dict__[name]().to(device)
        input_shape = (1, 3, 224, 224)
        inputs = torch.randn(input_shape, device=device)
        stats = rasp.profile_compute_once(model, inputs=inputs)
        summary, _ = rasp.summary_leaves(stats)
        print(summary)
        summary, total_f = rasp.summary_node(stats)
        print(summary)
        total_ops, total_params = total_f['flops'], total_f['params']
        print("%s | %.2f | %.2f" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))

if __name__ == '__main__':
    main()