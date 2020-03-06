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
