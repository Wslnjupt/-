import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_grad_enabled(True)

def get_num_correct(preds,labels):
    return (preds.argmax(dim=1) == labels).sum().item()

class Network(nn.Module):

    def __init__(self):
        super(Network,self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 6,kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=60)
        self.out = nn.Linear(in_features=60,out_features=10)

    def forward(self,t):

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size =2, stride = 2)


        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size = 2,stride =2 )

        t = self.fc1(t.reshape(-1,12*4*4))
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        return t



train_set = torchvision.datasets.FashionMNIST(root=r"C:\Users\Administrator\Desktop\机器学习，深度学习参考教材\deeplizard--pytorch神经网络入门\准备数据\data_FashionMNIST",
                                              train=True,
                                              transform = transforms.Compose([transforms.ToTensor()]),
                                              download = True)


train_loader = torch.utils.data.DataLoader(train_set,batch_size = 100)

network = Network()
# loss = F.cross_entropy()
updater = optim.SGD(network.parameters(),lr= 0.01)
# lr = 0.01

for epoch in range(50):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images,labels =batch

        preds = network(images)

        l = F.cross_entropy(preds,labels)

        updater.zero_grad()
        l.backward()
        updater.step()

        total_loss += l.item()
        total_correct += get_num_correct(preds,labels)

    print("epoch=",0,"total_loss= ",total_loss,"total_correct_%=  ",total_correct/len(train_set))


print(network.conv1.weight)



