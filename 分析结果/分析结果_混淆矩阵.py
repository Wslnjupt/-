import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


train_set = torchvision.datasets.FashionMNIST(root = r"C:\Users\Administrator\Desktop\机器学习，深度学习参考教材\deeplizard--pytorch神经网络入门\准备数据\data_FashionMNIST",
                                              train= True,download = True,transform = transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set,batch_size = 100)



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

network = Network()
updater = optim.SGD(network.parameters(),lr = 0.01)
# loss = F.cross_entropy(preds,labels)

def get_num_correct(preds,labels):
    return  (preds.argmax(dim = 1) == labels).sum()



for epoch  in range(10):
    train_loss = 0
    train_num_correct = 0
    for images,labels in train_loader:

        preds = network(images)
        train_num_correct += get_num_correct(preds,labels)
        loss = F.cross_entropy(preds,labels)
        train_loss += loss

        updater.zero_grad()
        loss.backward()


        updater.step()
    print("epoch=  ",{epoch},"train_loss = ",{train_loss},"correct_% = ",{train_num_correct/len(train_set)})

def get_all_preds(network,train_loader):
    all_preds = torch.tensor([])
    for batch in train_loader:
        images,labels = batch
        preds =network(images)

        all_preds = torch.cat((all_preds,preds),dim=0)
    return all_preds

with torch.no_grad():
    train_preds = get_all_preds(network,train_loader)


train_preds = train_preds.argmax(dim = 1)
print("train_preds——shape-----(6000,)?????",train_preds.shape)
stacked = torch.cat((train_preds,train_set.targets),dim =1 )
# stacked = torch.stack((train_preds,train_set.targets),dim =1 )

cmt = torch.zeros((10,10),dtype = torch.int64)
for p in stacked:
    j,k = p
    cmt[k,j] += 1

print(cmt)