import torch
import torchvision
import torch.nn as nn

vgg16_false = torchvision.models.vgg16(weights =False )
# vgg16_true = torchvision.models.vgg16(weights = True)

print(vgg16_false)

# vgg16_false.classifier.add_module("add_linear",nn.Linear(in_features=1000,out_features=10))

# print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(in_features=1000,out_features=10)
print(vgg16_false)