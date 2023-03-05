import json
import time

import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from itertools import product
from collections import OrderedDict
from collections import namedtuple
import pandas as pd



train_set = torchvision.datasets.FashionMNIST(root=r"C:\Users\Administrator\Desktop\机器学习，深度学习参考教材\deeplizard--pytorch神经网络入门\准备数据\data_FashionMNIST",
                                              train = True,transform = transforms.Compose([transforms.ToTensor()]))

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.fc1(t.reshape(-1, 12 * 4 * 4))
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.fc3(t)

        return t

# def get_num_correct(preds,labels):
#     return  (preds.argmax(dim = 1) == labels).sum()


torch.set_grad_enabled(True)

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple("Run",params.keys())

        run = []
        for v in product(*params.values()):
            run.append(Run(*v))

        return run

class RunManager():
    def __init__(self):

         self.epoch_count = 0
         self.epoch_loss = 0
         self.epoch_num_correct = 0
         self.epoch_start_time = None

         self.run_param = None
         self.run_count = 0
         self.run_data = []
         self.run_start_time = None

    def begin_run(self,run,network,loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment = f"{run}")

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss/len(self.loader.dataset)
        accuracy = self.epoch_num_correct/len(self.loader.dataset)

        self.tb.add_scalar("loss",loss,self.epoch_count)
        self.tb.add_scalar("accuracy",accuracy,self.epoch_count)

        for name,param in self.network.named_parameters():
            self.tb.add_histogram(name,param,self.epoch_count)
            self.tb.add_histogram(f"{name}.grad",param.grad,self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] =self.epoch_count
        results["loss"]  = loss
        results["accuracy"] =  accuracy
        results["epoch duration"] = epoch_duration
        results["run duration "] = run_duration
        for k,v in self.run_params._asdict():
            results[k] = v
        self.run_data.append(results)
        df =  pd.DataFrame.from_dict(self.run_data)
        return df
        # clear_output(wait =True)
        #  display(df)
    def track_loss(self,loss):
        self.epoch_loss = loss.item() *self.loader.batch_size

    def track_num_correct(self,preds,labels):
        self.epoch_num_correct += self._get_num_correct(preds,labels)

    @torch.no_grad()
    def _get_num_correct(self,preds,labels):
        return  (preds.argmax(dim = 1) == labels).sum()
    def save(self,fileName):
        pd.DataFrame.from_dict(self.run_data).to_csv(f"{fileName}.csv")

        with open(f"{fileName}.json","w") as f:
            json.dump(self.run_data,f,ensure_ascii=False,indent=4)









params = OrderedDict(lr = [0.01,0.001],batch_size = [1000,2000])
m = RunManager()

for run in RunBuilder.get_runs(params):
    network = Network()
    loader = torch.utils.data.DataLoader(train_set,batch_size = run.batch_size)
    updater = optim.SGD(network.parameters(),lr = run.lr)

    m.begin_run(run,network,loader)
    for epoch in range(5):
        m.begin_epoch()
        for batch in loader:

            images,labels = batch
            preds = network(images)
            loss = F.cross_entropy(preds,labels)
            updater.zero_grad()
            loss.backward()
            updater.step()

            m.track_loss(loss)
            m.track_num_correct(preds,labels)
        m.end_epoch()
    m.end_run()
m.save("results")





