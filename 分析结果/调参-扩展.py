from collections import OrderedDict
from collections import namedtuple
from itertools import product
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple("Run",params.keys())

        runs = []

        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

# params = OrderedDict(lr = [0.01,0.001],batch_size = [1000,10000])
# print(params)

# runs = RunBuilder.get_runs(params)
# run = runs[0]
#
# print(run)
# print(run.lr,run.batch_size)
#
# for run in runs:
#     print(run,run.lr,run.batch_size)

train_set = torchvision.datasets.FashionMNIST(root = r"C:\Users\74566\Desktop\机器学习，深度学习参考教材\deeplizard--pytorch神经网络入门\准备数据\data_FashionMNIST",
                                              train= True,download = True,transform = transforms.Compose([transforms.ToTensor()]))





torch.set_grad_enabled(True)

class Network(nn.Module):

    def __init__(self):
        super(Network,self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1,out_channels=6,kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels = 12,kernel_size  =5)



        self.fc1 = nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=60)
        self.fc3 = nn.Linear(in_features=60,out_features=10)


    def forward(self,t):

        t = self.conv1(t)
        t = F.relu(t)
        t  = F.max_pool2d(t,kernel_size= 2,stride = 2)


        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size = 2,stride = 2)

        t = self.fc1(t.reshape(-1,12*4*4))
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.fc3(t)

        return t


def get_num_correct(preds,labels):
    return  (preds.argmax(dim = 1) == labels).sum()


parameters = OrderedDict(lr = [0.01,0.001],batch_size = [10,100,1000],shuffle = [True,False])
# parameters_values = [v for v in parameters.values()]
runs = RunBuilder.get_runs(parameters)


for run in runs:
    device = torch.device("cpu")
    network = Network().to(device)
    comment = f"{run}"
    # comment = "batch_size =100,lr= 0.01"
    tb = SummaryWriter(comment=comment)

    updater = optim.SGD(network.parameters(),lr = run.lr)
    # loss = F.cross_entropy(preds,labels)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = run.batch_size,shuffle = run.shuffle)
    for epoch  in range(2):
        train_loss = 0
        train_num_correct = 0
        for images,labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = network(images)
            train_num_correct += get_num_correct(preds,labels)

            loss = F.cross_entropy(preds,labels)
            train_loss += loss*run.batch_size

            updater.zero_grad()
            loss.backward()


            updater.step()
        print("lr = ",run.lr,"batch_size = ",run.batch_size,"shuffle = ",run.shuffle,
              "epoch=  ",epoch,"train_loss = ",train_loss,"correct_% = ",train_num_correct/len(train_set))

        tb.add_scalar("train_loss",train_loss,epoch)
        tb.add_scalar("train_correct_%",train_num_correct/len(train_set),epoch)
    tb.close()
