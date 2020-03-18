# RASP

Runtime Analyzer and Statistical Profiler

## Usage

Get your model

```python
import rasp
from torchvision.models import AlexNet
model = AlexNet()
```

For basic stats

```python
rasp.stat(model, input_shape=(1, 3, 224, 224))
```

For runtime stats

```python
rasp.stat(model, input_shape=(1, 3, 224, 224), timing=True)
```

Specify device

```python
rasp.stat(model, input_shape=(1, 3, 224, 224), device='cuda', timing=True)
```

Export DataFrame

```python
summary, df = rasp.stat(model, input_shape=(1, 3, 224, 224), device='cuda', timing=True, print_only=False)
```

Customize profiling parameters

```python
rasp.CFG.profile.num_batches=200
rasp.CFG.profile.warmup_batches=20
rasp.stat(model, input_shape=(1, 3, 224, 224), timing=True)
```

Use your own input

```python
inputs = torch.randn(8, 3, 224, 224)
rasp.stat(model, inputs=inputs, timing=True)
```

Set different report type

```python
for report_type in ['tape', 'subnodes', 'leaves', 'root', None]:
    rasp.stat(model, input_shape=(1, 3, 224, 224), report_type=report_type, timing=True)
```

Save report

```python
rasp.stat(model, input_shape=(1, 3, 224, 224), timing=True, save_path='./reports')
```

## Addons

### Energy Cost

(speculative)

```python
import rasp
import rasp.addons.energy
rasp.stat(model, input_shape=(1, 3, 224, 224), includes=['energy'])
```

## Results

torchvision models measured on x86 CPU

Model | Params | FLOPs | Energy (pJ) | Latency (ms) | FLOPS
---|---|---|---|---|---
alexnet | 61.1M | 715.51M | 11.59G | 51.28 | 13.95G
densenet121 | 7.98M | 2.91G | 25.53G | 236.69 | 12.28G
densenet161 | 28.68M | 7.87G | 60.26G | 465.83 | 16.89G
densenet169 | 14.15M | 3.45G | 31.04G | 267.22 | 12.91G
densenet201 | 20.01M | 4.41G | 39.96G | 325.93 | 13.53G
googlenet | 6.62M | 1.51G | 10.15G | 138.81 | 10.89G
mnasnet0_5 | 2.22M | 141.15M | 3.94G | 125.75 | 1.12G
mnasnet0_75 | 3.17M | 240.27M | 5.29G | 176.01 | 1.37G
mnasnet1_0 | 4.38M | 335.93M | 6.25G | 217.23 | 1.55G
mnasnet1_3 | 6.28M | 528.32M | 8.29G | 286.33 | 1.85G
mobilenet_v2 | 3.5M | 327.14M | 7.09G | 181.15 | 1.81G
resnet101 | 44.55M | 7.92G | 54.08G | 321.71 | 24.63G
resnet152 | 60.19M | 11.66G | 77.98G | 455.6 | 25.59G
resnet18 | 11.69M | 1.84G | 11.85G | 104.73 | 17.59G
resnet34 | 21.8M | 3.7G | 22.53G | 173.6 | 21.29G
resnet50 | 25.56M | 4.19G | 30.73G | 196.17 | 21.36G
resnext101_32x8d | 88.79M | 16.58G | 110.75G | 702.24 | 23.61G
resnext50_32x4d | 25.03M | 4.32G | 33.84G | 235.09 | 18.37G
shufflenet_v2_x0_5 | 1.37M | 44.62M | 1.19G | 41.67 | 1.07G
shufflenet_v2_x1_0 | 2.28M | 152.54M | 2.47G | 74.02 | 2.06G
shufflenet_v2_x1_5 | 3.5M | 306.46M | 3.92G | 104.28 | 2.94G
shufflenet_v2_x2_0 | 7.39M | 597.64M | 6.46G | 142.92 | 4.18G
squeezenet1_0 | 1.25M | 826.09M | 6.39G | 83.3 | 9.92G
squeezenet1_1 | 1.24M | 352.76M | 3.25G | 47.72 | 7.39G
vgg11 | 132.86M | 7.62G | 56.18G | 321.75 | 23.68G
vgg11_bn | 132.87M | 7.65G | 58.27G | 336.94 | 22.7G
vgg13 | 133.05M | 11.32G | 75.41G | 503.35 | 22.49G
vgg13_bn | 133.05M | 11.37G | 78.87G | 517.65 | 21.97G
vgg16 | 138.36M | 15.49G | 95.54G | 622.3 | 24.88G
vgg16_bn | 138.37M | 15.54G | 99.36G | 642.72 | 24.18G
vgg19 | 143.67M | 19.65G | 115.66G | 739.1 | 26.58G
vgg19_bn | 143.68M | 19.71G | 119.85G | 752.85 | 26.18G
wide_resnet101_2 | 126.89M | 22.98G | 136.68G | 770.79 | 29.82G
wide_resnet50_2 | 68.88M | 11.6G | 72.4G | 430.54 | 26.94G
