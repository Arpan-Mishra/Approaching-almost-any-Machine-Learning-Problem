import torch.nn as nn
import pretrainedmodels

def get_model(pretrained, model_name = None):

    if pretrained:
        model = pretrainedmodels.__dict__[str(model_name)](
            pretrained = 'imagenet'
        )
    else:
        model = pretrainedmodels.__dict__[str(model_name)](
            pretrained = None
        )


    model.last_linear = nn.Sequential(nn.BatchNorm1d(512),
    nn.Dropout(0.25),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256, eps = 1e-05, momentum = 0.1),
    nn.Dropout(0.5),
    nn.Linear(256, 1))
    

    return model


get_model(True, 'resnet34')
    