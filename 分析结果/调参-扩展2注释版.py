# 这个代码是视频32中的完全体,但是在我的电脑上TensorBoard会造成段错误,所以实际运行的时候,我运行的是删除了所有TensorBoard的代码
# 所以本份代码仅供分析

import torch as t
from pandas import DataFrame
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import time
import json

from itertools import product

from collections import (
    OrderedDict,
    namedtuple
)

# 如果想要添加超参数,需要动两个地方,第一处是params的定义处,添上一项 shuffle = [True, False]
# 另外一个地方是for run in RunBuilder.get_runs(params)的代码体里的 loader=DataLoader(train_set,batch_size=run.batch_size)
# 改成 loader=DataLoader(train_set,batch_size=run.batch_size, shuffle=run.shuffle )
# 第二处修改依赖于第一处

params = OrderedDict(
    lr=[.01],
    batch_size=[100, 1000],
)

train_set = torchvision.datasets.FashionMNIST(
    root='/home/arthur/data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        return t


# 这个类里只有get_runs一个静态方法,这个方法返回一个列表,列表里面装满了超参数的排列,它是具名元组
# 比如第一个元素可能会是lr=0.1, batch_size=100
# 这个列表里面的值供实际的模型使用,同时也供TensorBoard使用
class RunBuilder():
    @staticmethod
    def get_runs(params: OrderedDict) -> list:
        Run = namedtuple('Run', params.keys())
        runs = []
        # 这个迭代的作用就是把生成的具名元组们全都填到列表里
        # 每个具名元组里的元素们都代表着一套超参数
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


# run是个大单位,n轮epoch构成一次run; 一次一次从一个dataloader里往外掏数据,掏空一次就是一次epoch
# 然后换一套超参数,再执行这个步骤就得到另外一次run
class RunManager():
    def __init__(self):
        # 周期的一些属性
        self.epoch_count = 0  # 完整覆盖一次训练集就是一个epoch,这个数字应该指的是已经进行了几轮覆盖
        self.epoch_loss = 0  # 每轮epoch的损失
        self.epoch_num_correct = 0  # 每轮正确的数字
        self.epoch_start_time = None  # 每轮开始的时间

        # 所有使用同一批超参数的epochs们,是一次run ! ! !
        # 用另外一批超参数的epochs们,是另外一次run ! ! !
        self.run_params = None  # 这玩意的值来自RunBuilder,各个run们的超参数们就是从这里面拿的
        self.run_count = 0  # 记录到底是第几次run了
        self.run_data: list = []  # ( 追踪每个epoch的参数值和结果 ? ? ? )
        self.run_start_time = None  # 用来计算run们的运行时间

        self.network = None
        self.loader = None
        self.tb: SummaryWriter = None

    # _表示不打算被外部调用
    @t.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()  # 捕获本次run的开始的时间
        self.run_params = run  # 获取本run中的epochs们用的一套超参数
        self.run_count += 1  # 记录当前的run是第几次run

        self.network = network
        self.loader = loader

        # (这行会导致段错误...)
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    # begin_epoch和end_epoch允许我们在整个运行周期中管理这些值
    def begin_epoch(self):
        self.epoch_start_time = time.time()  # 启动一个epoch的第一件事就是保存开始时间
        self.epoch_count += 1  # epoch数+1
        self.epoch_loss = 0  # 避免受上一轮epoch的影响,把一些东西重置为0
        self.epoch_num_correct = 0

    # 结束某个epoch要干的事情最多
    def end_epoch(self):
        # 计算本次epoch花费的时间,也就是本次完成覆盖一次数据集花了多久
        epoch_duration = time.time() - self.epoch_start_time
        # 计算run执行多久了
        run_duration = time.time() - self.run_start_time

        # 相对于训练集大小来计算损失和精确度
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        # 然后把这两个值传递个tensorboard
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuaracy', accuracy, self.epoch_count)

        # 就像之前那样把网络权重和梯度值传递给tensorboard
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()  # 这个字典包含了我们在运行中关心的一些数据
        results['run'] = self.run_count  # 记录当前的epoch属于第几次run里面的
        results['epoch'] = self.epoch_count  # 记录当前的epoch是本次run的第几个epoch
        # 记录当前epoch的loss,精度和一些耗时
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        # 记录这些东西的原因是想要向硬盘保存

        # 上面的代码存完相关的数据之后,继续往里面存其他数据
        # 这次存的是本次epoch使用的超参数们.
        # 假设一个run中有n个epoch,则连续n个epoch用的都是这一套超参数,在csv文件中可以看到
        for k, v in self.run_params._asdict().items():  # _asdict()是namedtuple的方法
            results[k] = v

        # 存完本次epoch的相关数据之后,将结果字典添加到run_data列表里
        # 各个epoch的这些数据都不尽相同,也就是说results都不尽相同,但是run_data是属于RunManager的
        # 整个程序的执行过程都只有一个run_data. 此所谓:流水的results,铁打的run_data
        self.run_data.append(results)
        # 在所有run执行完毕之后,run_data将会被save函数以csv和json保存在硬盘上


        # df: DataFrame = pd.DataFrame.from_dict(self.run_data, orient='columns')
        # 这两行是针对Jupyter的,保证输出被更新而不是追加,复制一个33.ipynb时取消注释
        # clear_output(wait=True)
        # display(df)

    # 训练过程中track_loss和track_num_correct在合适的位置被调用,
    # loss被传递个给track_loss使用,preds和labels被传递给track_num_correct使用
    # 从而每次epochs的损失和预测正确的数目得到了更新,从而能够被end_epoch记录下来
    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    # 这个函数把run_data在硬盘上保存成csv和json格式的文件,使得其他程序也可以使用
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data
            , orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)


m = RunManager()

for run in RunBuilder.get_runs(params):

    # 下面的几行代码对于每次run都会重新执行,保证每次run拿到的都是新的未训练的网络和各种新的属性和新的超参数
    network = Network().cuda()

    loader = DataLoader(train_set, batch_size=run.batch_size)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    # begin_run函数实际上是做一些记录和初始化的操作
    m.begin_run(run, network, loader)
    for epoch in range(5):  # 决定5次epochs为一次run
        m.begin_epoch()  # 进行一些初始化和记录

        # 这里的代码是1次epoch真正训练的过程
        # 根据既定的batch size,把图片们从loader里面一次一次拿出来
        # 每次拿出来一个batch就把它传递给网络进行学习,而学习就会提高网络的精度
        # 导致每次学习其实就会有新的权重等参数,用这些东西来计算损失和预测正确的数字
        for batch in loader:
            images, labels = batch
            images, labels = images.cuda(), labels.cuda()
            preds = network(images)  # pass batch,向前传导
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m.track_loss(loss)
            m.track_num_correct(preds, labels)

        # 当一次epoch完成之后就用end_epoch收尾
        m.end_epoch()
    m.end_run()
    # 同理,当既定轮次的epochs们执行完毕之后,一次run也结束了
    # 用end_run收尾

m.save('results')

