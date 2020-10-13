import torchvision.models as models
import torch.nn as nn
from config.value_config import *
def make_model(key):
    model = models.__dict__[key](pretrained=True)
    fc_feature = model.fc.in_features
    model.fc = nn.Linear(fc_feature, NUMCLASS)
    return model



