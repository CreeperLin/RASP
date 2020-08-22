import rasp
import torch.nn as nn


def test_model():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
    )
    rasp.stat(model, input_shape=(1, 3, 224, 224))
    rasp.stat(model, input_shape=(1, 3, 224, 224), timing=True)
