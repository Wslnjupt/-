from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


# writer = SummaryWriter()
# writer.add_image()
# writer.add_scalar()
#
# writer.close()

writer= SummaryWriter("logs")
image_path = r"C:\Users\Administrator\Desktop\机器学习，深度学习参考教材\deeplizard--pytorch神经网络入门\准备数据\hymenoptera_data\train\bees\21399619_3e61e5bb6f.jpg"
image = Image.open(image_path)
imgae_array = np.array(image)
# print(type(imgae_array))
# print(imgae_array.shape)



writer.add_image("test",imgae_array,2,dataformats="HWC")
for i in range(100):
    writer.add_scalar("y = x",2*i , i )

writer.close()