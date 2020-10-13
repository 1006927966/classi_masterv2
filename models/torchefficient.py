from efficientnet_pytorch.model import EfficientNet 
import torch.nn as nn 

def make_model(arch, nclass):
    num = nclass 
    if arch == 'efficientNetb0':
        model_ft = EfficientNet.from_pretrained('efficientnet-b0')
    elif arch == 'efficientNetb1':
        model_ft = EfficientNet.from_pretrained('efficientnet-b1')
    elif arch == 'efficientNetb2':
        model_ft = EfficientNet.from_pretrained('efficientnet-b2')
    elif arch == 'efficientNetb3':
        model_ft = EfficientNet.from_pretrained('efficientnet-b3')
    elif arch == 'efficientNetb4':
        model_ft = EfficientNet.from_pretrained('efficientnet-b4')
    elif arch == 'efficientNetb5':
        model_ft = EfficientNet.from_pretrained('efficientnet-b5')
    elif arch == 'efficientNetb6':
        model_ft = EfficientNet.from_pretrained('efficientnet-b6')
    elif arch == 'efficientNetb7':
        model_ft = EfficientNet.from_pretrained('efficientnet-b7')
    else:
        model_ft = EfficientNet.from_pretrained('efficientnet-b3')
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, num)
    return model_ft 

    