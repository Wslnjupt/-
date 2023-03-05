import torch
import torchvision
import torchvision.transforms as transforms

import  torch.nn as nn
import  torch.nn.functional as F

import torch.optim as optim

train_set  = torchvision.datasets.FashionMNIST(root =r"C:\Users\Administrator\Desktop\机器学习，深度学习参考教材\deeplizard--pytorch神经网络入门\准备数据\data_FashionMNIST",
                                               train= True,download=True,transform= transforms.Compose([transforms.ToTensor()]))

class Network(nn.Module):

    def __init__(self):
        super(Network,self).__init__()

        self.cov1 = nn.Conv2d(in_channels = 1,out_channels=6,kernel_size = 5)
        self.cov2 = nn.Conv2d(in_channels = 6,out_channels=12,kernel_size = 5)


        self.fc1 = nn.Linear(in_features = 12*4*4,out_features = 120)
        self.fc2 = nn.Linear(in_features = 120,out_features = 60)
        self.out = nn.Linear(in_features = 60,out_features = 10)


    def forward(self,t):
        t = F.relu(self.cov1(t))
        t = F.max_pool2d(t,kernel_size = 2,stride = 2)

        t = F.relu(self.cov2(t))
        t = F.max_pool2d(t,kernel_size = 2,stride = 2)

        t =F.relu(self.fc1(t.reshape(-1,12*4*4)))
        t =F.relu(self.fc2(t))
        t  = self.out(t)
        return  t


torch.set_grad_enabled(True)

network = Network()
# sample = next(iter(train_set))
# image,label = sample
# # print(image.shape)
# pred = network(image.unsqueeze(0))
# print(pred)
# print(label)
# print(pred.argmax(dim = 1))
#
# print(F.softmax(pred,dim = 1))
# print(F.softmax(pred,dim =1).sum())

data_loader = torch.utils.data.DataLoader(train_set,batch_size = 10)

image,labels = next(iter(data_loader))
# print(image.shape)
# print(labels.shape)

pred =network(image)
# print(pred.shape)
# print(pred)
# print(pred.argmax(dim =1) == labels)
# print(pred.argmax(dim =1).eq(labels).sum())

loss = F.cross_entropy(pred,labels)
# print(loss)
# print(network.cov1.weight.grad)
loss.backward()
# print(network.cov1.weight.grad)

optimizer = optim.SGD(network.parameters(),lr = 0.01)

print(loss)

optimizer.step()

pred = network(image)
loss = F.cross_entropy(pred,labels)
print(loss)