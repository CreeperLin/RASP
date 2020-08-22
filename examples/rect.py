import torch
import torch.nn as nn
import argparse
import rasp


class RectNet(nn.Module):
    def __init__(self, C, w, l):
        super().__init__()
        stages = nn.ModuleList()
        for i in range(w):
            stack = nn.ModuleList()
            for j in range(l):
                stack.append(nn.Conv2d(C, C, 3, 1, 1))
            stages.append(stack)
        self.stages = stages

    def forward(self, x):
        states_out = []
        keep_states = False
        for stack in self.stages:
            stack_out = [x]
            for net in stack:
                if keep_states:
                    stack_out.append(net(stack_out[-1]))
                else:
                    stack_out[0] = net(stack_out[-1])
            states_out.append(stack_out[-1])
        return sum(states_out)


def get_device(devdesc):
    torch.manual_seed(1)
    if devdesc == 'all':
        dev = list(range(torch.cuda.device_count()))
    elif devdesc != 'cpu':
        dev = [int(s) for s in devdesc.split(',')]
    if devdesc == 'cpu' or len(dev) == 0:
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
    parser.add_argument('-d',
                        '--device',
                        type=str,
                        default="all",
                        help="device ids")
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help="verbose msg")
    parser.add_argument('-t',
                        '--timing',
                        action='store_true',
                        help="enable timing")
    args = parser.parse_args()

    device, devlist = get_device(args.device)

    rasp.set_config({
        'profile': {
            'batch_size': 1,
            'num_batches': 5,
            'warmup_batches': 5,
            'timing_max_depth': -1,
            'compute_max_depth': -1,
            'verbose': args.verbose,
        },
    })

    fields = [
        'name', 'type', 'in_shape', 'out_shape', 'params', 'lat', 'net_lat',
        'lat[%]', 'flops', 'mem_r', 'mem_w', 'mem_rw', 'dev_mem_alloc',
        'dev_mem_delta'
    ]

    C = 16
    w = 4
    l = 8
    model = RectNet(C, w, l).to(device)

    bsize = 128
    fm_size = 224
    inputs = torch.randn(bsize, C, fm_size, fm_size).to(device)

    stats = rasp.profile_compute_once(model, inputs=inputs)
    if args.timing: stats = rasp.profile_timing_once(model, inputs=inputs)
    summary, _ = rasp.summary_tape(stats, report_fields=fields)
    print(summary)
    summary, _ = rasp.summary_all(stats, report_fields=fields)
    print(summary)
    summary, _ = rasp.summary_node(stats, report_fields=fields)
    print(summary)
    rasp.profile_off(model)


if __name__ == '__main__':
    main()
