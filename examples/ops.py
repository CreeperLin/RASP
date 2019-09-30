import torch
import torch.nn as nn
import argparse
import rasp

class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

ops = {
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    # 'skip_connect': lambda C, stride, affine: \
    #     nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine),
    'conv_1x1': lambda C, stride, affine: StdConv(C, C, 1, stride, 0, affine=affine)
}

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

    print("%s | %s | %s | %s" % ("Model", "Params", "MAdds", "FLOPs"))
    print("---|---|---|---")

    fields = ['name', 'type', 'in_shape', 'out_shape', 'params', 'madds',
             'lat', 'net_lat', 'lat[%]', 'flops',
             'mem_r', 'mem_w', 'mem_rw', 'dev_mem_alloc', 'dev_mem_delta']

    chn_in = 16
    fm_size = 224
    input_shape = (1, chn_in, fm_size, fm_size)
    inputs = torch.randn(input_shape, device=device)
    
    for i, name in enumerate(ops):
        model = ops[name](chn_in, stride=1, affine=True).to(device=device)
        stats = rasp.profile_compute_once(model, inputs=inputs)
        if args.timing: stats = rasp.profile_timing_once(model, inputs=inputs)
        summary, _ = rasp.summary_tape(stats, report_fields=fields)
        if args.verbose: print(summary)
        _, total_f = rasp.summary_node(stats, report_fields=fields)
        rasp.profile_off(model)
        total_flops, total_madds, total_params = total_f['flops'], total_f['madds'], total_f['params']
        print("%s | %s | %s | %s" % (name, rasp.round_value(total_params),
             rasp.round_value(total_madds), rasp.round_value(total_flops)))

if __name__ == '__main__':
    main()