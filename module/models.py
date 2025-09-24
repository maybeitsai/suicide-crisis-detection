import torch
import torch.nn as nn
import torchvision.models as models

def face_expression_model(num_classes, variant="large"):
    if variant == "large":
        model = models.mobilenet_v3_large(pretrained=True)
        in_feats = model.classifier[0].in_features   # 960
    else:
        model = models.mobilenet_v3_small(pretrained=True)
        in_feats = model.classifier[0].in_features   # 576

    model.classifier = nn.Sequential(
        nn.Linear(in_feats, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes)
    )
    return model

def pose_recognition_model(num_classes, variant="large"):
    if variant == "large":
        model = models.mobilenet_v3_large(pretrained=True)
        in_feats = model.classifier[0].in_features   # 960
    else:
        model = models.mobilenet_v3_small(pretrained=True)
        in_feats = model.classifier[0].in_features   # 576

    model.classifier = nn.Sequential(
        nn.Linear(in_feats, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes)
    )
    return model
