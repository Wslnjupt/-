import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(root= r"C:\Users\Administrator\Desktop\机器学习，深度学习参考教材\deeplizard--pytorch神经网络入门\准备数据\data_FashionMNIST",
                                              train=True,download=True,transform = transforms.Compose([transforms.ToTensor()]))
transforms.Resize()
#计算数据集的（按照通道分别计算）均值和方差

# #第一种方式
# loader = torch.utils.data.DataLoader(train_set,batch_size = len(train_set))
# data = next(iter(loader))
# # print(type(data))
# # print(len(data))
# # print(data[1])
# print(data[0].mean())
# print(data[0].std())
#
#

#第二种方式
loader = torch.utils.data.DataLoader(train_set,batch_size = 1000)
num_of_pixels = len(train_set)*28*28

total_sum = 0
for batch in loader:
    total_sum += batch[0].sum()
mean = total_sum/num_of_pixels


sum_of_squared_error = 0
for batch in loader:
    sum_of_squared_error +=(batch[0] - mean).pow(2)

std = torch.sqrt(sum_of_squared_error/num_of_pixels)
print(mean,std)