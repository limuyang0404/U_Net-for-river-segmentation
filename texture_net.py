import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import gpu_no_of_var
# def conv3d_block(input_tensor, input_channel, n_filters, kernel_size=3, stride=1, padding=0, batchnorm=True):
#     #first layer
#     x = nn.Conv3d(in_channels=input_channel, out_channels=n_filters, kernel_size=kernel_size, stride=stride, padding=padding)(input_tensor)
#     if batchnorm:
#         x = nn.BatchNorm3d()(x)
#     x = nn.ReLU()(x)
#     #second layer
#     x = nn.Conv3d(in_channel=n_filters, out_channels=n_filters, kernel_size=kernel_size, stride=stride, padding=padding)(x)
#     if batchnorm:
#         x = nn.BatchNorm3d()(x)
#     x = nn.ReLU()(x)
#     return x
def GaussianInitialize(layer):
    layer_size = list(layer.weight.size())
    layer_point = 1
    for i in range(len(layer_size)):
        layer_point = layer_point * layer_size[i]
    initialize = np.random.normal(0, (2 / layer_point) ** 0.5, layer_point)
    initialize = initialize.reshape(layer_size)
    layer.weight = nn.Parameter(torch.Tensor(initialize))
    return


class conv3d_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv3d_block, self).__init__()
        conv_relu = []
        conv_relu.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.BatchNorm3d(out_channels))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.BatchNorm3d(out_channels))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out
class TextureNet(nn.Module):
    def __init__(self,n_classes=2):
        super(TextureNet,self).__init__()
        self.left_conv_1 = conv3d_block(in_channels=1, out_channels=16)
        self.pool_1 = nn.MaxPool3d(2, 2)                       #（16， 16， 16）
        self.drop1 = nn.Dropout(p=0.1)
        self.left_conv_2 = conv3d_block(in_channels=16, out_channels=32)
        self.pool_2 = nn.MaxPool3d(2, 2)                        #（8， 8， 8）
        self.drop2 = nn.Dropout(p=0.1)
        self.left_conv_3 = conv3d_block(in_channels=32, out_channels=64)
        self.pool_3 = nn.MaxPool3d(2, 2)                        #（4， 4， 4）
        self.drop3 = nn.Dropout(p=0.1)
        self.left_conv_4 = conv3d_block(in_channels=64, out_channels=128)
        self.pool_4 = nn.MaxPool3d(2, 2)                        #（2， 2， 2）
        self.drop4 = nn.Dropout(p=0.1)
        self.left_conv_5 = conv3d_block(in_channels=128, out_channels=256)
        self.deconv_1 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.right_conv_1 = conv3d_block(in_channels=256, out_channels=128)
        self.deconv_2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=3, padding=2)
        self.right_conv_2 = conv3d_block(in_channels=128, out_channels=64)
        self.deconv_3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=3, padding=4)
        self.right_conv_3 = conv3d_block(in_channels=64, out_channels=32)
        self.deconv_4 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=3, padding=8)
        self.right_conv_4 = conv3d_block(in_channels=32, out_channels=16)
        self.right_conv_5 = nn.Conv3d(in_channels=16, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def ParameterInitialize(self):
        GaussianInitialize(self.left_conv_1.conv_ReLU[0])
        GaussianInitialize(self.left_conv_1.conv_ReLU[3])
        GaussianInitialize(self.left_conv_2.conv_ReLU[0])
        GaussianInitialize(self.left_conv_2.conv_ReLU[3])
        GaussianInitialize(self.left_conv_3.conv_ReLU[0])
        GaussianInitialize(self.left_conv_3.conv_ReLU[3])
        GaussianInitialize(self.left_conv_4.conv_ReLU[0])
        GaussianInitialize(self.left_conv_4.conv_ReLU[3])
        GaussianInitialize(self.left_conv_5.conv_ReLU[0])
        GaussianInitialize(self.left_conv_5.conv_ReLU[3])
        GaussianInitialize(self.right_conv_1.conv_ReLU[0])
        GaussianInitialize(self.right_conv_1.conv_ReLU[3])
        GaussianInitialize(self.right_conv_2.conv_ReLU[0])
        GaussianInitialize(self.right_conv_2.conv_ReLU[3])
        GaussianInitialize(self.right_conv_4.conv_ReLU[0])
        GaussianInitialize(self.right_conv_4.conv_ReLU[3])
        GaussianInitialize(self.right_conv_5)
        print('ALL convolutional layer have been initialized !')
        print("*" * 100)

        # # Network definition
        # self.net = nn.Sequential(
        #     nn.Conv3d(1, 16, 3, 1, padding=0),
        #     nn.BatchNorm3d(16),
        #     nn.ReLU(),
        #     nn.Conv3d(16, 16, 3, 1, padding=0),
        #     nn.BatchNorm3d(16),
        #     nn.ReLU(),     #c1
        #
        #     nn.MaxPool3d(2, stride=2, padding=0, return_indices=False, ceil_mode=False),#p1
        #
        #     nn.Dropout(p=0.25),
        #
        #     nn.Conv3d(16, 32, 3, 1, padding=0),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(),
        #     nn.Conv3d(32, 32, 3, 1, padding=0),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(),  # c2
        #
        #     nn.MaxPool3d(2, stride=2, padding=0, return_indices=False, ceil_mode=False),#p2
        #
        #     nn.Dropout(p=0.5),
        #
        #     nn.Conv3d(32, 64, 3, 1, padding=0),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 64, 3, 1, padding=0),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),  # c3
        #
        #     nn.MaxPool3d(2, stride=2, padding=0, return_indices=False, ceil_mode=False),  # p3
        #
        #     nn.Dropout(p=0.5),
        #
        #     nn.Conv3d(64, 128, 3, 1, padding=0),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 128, 3, 1, padding=0),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(),  # c4
        #
        #     nn.MaxPool3d(2, stride=2, padding=0, return_indices=False, ceil_mode=False),  # p4
        #
        #     nn.Dropout(p=0.5),
        #
        #     nn.Conv3d(128, 256, 3, 1, padding=0),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(),
        #     nn.Conv3d(256, 256, 3, 1, padding=0),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(),  # c5
        #
        #
        #     nn.ConvTranspose3d(256, 128, 3, 1, padding=0),
        #
        #     nn.Conv3d(8,16,5,1,padding=1, bias=False), #Parameters  #in_channels, #out_channels, filter_size, stride (downsampling factor)
        #     # # #nn.Dropout3d() #Droput can be added like this ...
        #     nn.ReLU(),
        #     # #
        #     nn.Conv3d(16,32,3,1,padding=1, bias=False),
        #     nn.ReLU(),
        #     #
        #     nn.MaxPool3d(2, stride=2, padding=1, return_indices=False, ceil_mode=False),
        #     nn.BatchNorm3d(32),
        #     #
        #     #
        #     # #
        #     # nn.Linear(32, 120, bias=True),
        #     # nn.BatchNorm3d(32),
        #     # # nn.ReLU(),
        #     # #
        #     # nn.Linear(120, 84, bias=True),
        #     # nn.BatchNorm3d(32),
        #     # # # nn.ReLU(),
        #     # # #
        #     # nn.Linear(84, 32, bias=True),
        #     # nn.BatchNorm3d(32),
        #     #
        #     # # nn.Conv3d(32, 32, (8, 8, 2), 32),
        #     # nn.ReLU()
        #     #
        #     # nn.Conv3d(9,2,8,8)
        #     #
        #     #
        #     #
        #     # nn.Conv3d(50,n_classes,1,1), #This is the equivalent of a fully connected layer since input has width/height/depth = 1
        #     # nn.ReLU(),
        #
        # )
        # # self.fc=nn.Sequential(
        # #     nn.Linear(16384, 120, bias=True),
        # #     nn.BatchNorm3d(32),
        # #     nn.ReLU(),
        # #     nn.Linear(120, 84, bias=True),
        # #     # nn.BatchNorm3d(84),
        # #     nn.ReLU(),
        # #     nn.Linear(84, 2, bias=True)
        # # )
        # self.fc1 = nn.Linear(16384,120,bias=True)
        # self.fc2 = nn.Linear(120,84,bias=True)
        # self.fc3 = nn.Linear(84,2,bias=True)
        # #The filter weights are by default initialized by random

    #Is called to compute network output
    def forward(self,x):
        feature_1 = self.left_conv_1(x)
        # print(feature_1.shape)
        feature_1_pool = self.pool_1(feature_1)
        drop_1 = self.drop1(feature_1_pool)
        feature_2 = self.left_conv_2(drop_1)
        # print(feature_2.shape)
        feature_2_pool = self.pool_2(feature_2)
        drop_2 = self.drop2(feature_2_pool)
        feature_3 = self.left_conv_3(drop_2)
        # print(feature_3.shape)
        feature_3_pool = self.pool_3(feature_3)
        drop_3 = self.drop3(feature_3_pool)
        feature_4 = self.left_conv_4(drop_3)
        # print(feature_4.shape)
        feature_4_pool = self.pool_4(feature_4)
        drop_4 = self.drop4(feature_4_pool)
        feature_5 = self.left_conv_5(drop_4)
        # print('5:\n', feature_5.shape)
        de_feature_1 = self.deconv_1(feature_5)
        # print('de_1:\n', de_feature_1.shape)
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        # print('temp:\n', temp.shape)
        de_feature_1_conv = self.right_conv_1(temp)
        # print('de_1_conv:\n', de_feature_1_conv.shape)
        de_feature_2 = self.deconv_2(de_feature_1_conv)
        # print('de_2:\n', de_feature_2.shape)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        # print('temp:\n', temp.shape)
        de_feature_2_conv = self.right_conv_2(temp)
        # print('de_2_conv:\n', de_feature_2_conv.shape)
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        # print('de_3:\n', de_feature_3.shape)
        temp = torch.cat((feature_2, de_feature_3), dim=1)
        # print('temp:\n', temp.shape)
        de_feature_3_conv = self.right_conv_3(temp)
        # print('de_3_conv:\n', de_feature_3_conv.shape)
        de_feature_4 = self.deconv_4(de_feature_3_conv)
        # print('de_4:\n', de_feature_4.shape)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        # print('temp:\n', temp.shape)
        de_feature_4_conv = self.right_conv_4(temp)
        # print('de_4_conv:\n', de_feature_4_conv.shape)
        out = self.right_conv_5(de_feature_4_conv)
        # out = F.softmax(out, dim = 1)#out.shape = (32, 2, 32, 32, 32)
        # out1, class_no = torch.max(out, 1)
        return out



    def classify(self,x):
        # pre = x
        feature_1 = self.left_conv_1(x)
        # print(feature_1.shape)
        feature_1_pool = self.pool_1(feature_1)
        drop_1 = self.drop1(feature_1_pool)
        feature_2 = self.left_conv_2(drop_1)
        # print(feature_2.shape)
        feature_2_pool = self.pool_2(feature_2)
        drop_2 = self.drop2(feature_2_pool)
        feature_3 = self.left_conv_3(drop_2)
        # print(feature_3.shape)
        feature_3_pool = self.pool_3(feature_3)
        drop_3 = self.drop3(feature_3_pool)
        feature_4 = self.left_conv_4(drop_3)
        # print(feature_4.shape)
        feature_4_pool = self.pool_4(feature_4)
        drop_4 = self.drop4(feature_4_pool)
        feature_5 = self.left_conv_5(drop_4)
        # print('5:\n', feature_5.shape)
        de_feature_1 = self.deconv_1(feature_5)
        # print('de_1:\n', de_feature_1.shape)
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        # print('temp:\n', temp.shape)
        de_feature_1_conv = self.right_conv_1(temp)
        # print('de_1_conv:\n', de_feature_1_conv.shape)
        de_feature_2 = self.deconv_2(de_feature_1_conv)
        # print('de_2:\n', de_feature_2.shape)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        # print('temp:\n', temp.shape)
        de_feature_2_conv = self.right_conv_2(temp)
        # print('de_2_conv:\n', de_feature_2_conv.shape)
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        # print('de_3:\n', de_feature_3.shape)
        temp = torch.cat((feature_2, de_feature_3), dim=1)
        # print('temp:\n', temp.shape)
        de_feature_3_conv = self.right_conv_3(temp)
        # print('de_3_conv:\n', de_feature_3_conv.shape)
        de_feature_4 = self.deconv_4(de_feature_3_conv)
        # print('de_4:\n', de_feature_4.shape)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        # print('temp:\n', temp.shape)
        de_feature_4_conv = self.right_conv_4(temp)
        # print('de_4_conv:\n', de_feature_4_conv.shape)
        out = self.right_conv_5(de_feature_4_conv)
        # out = F.softmax(out, dim=1)  # out.shape = (32, 2, 32, 32, 32)
        # output = out[:, 1, :, :, :]
        # # print('softmax result = ', x)
        # class_real, class_no = torch.max(out, 1, keepdim=True)#output the max in every row,keepdim means the output's shape same as input.class_no is a tensor,every element means the max's location in the row
        # print('class_real = ', class_real, 'class_no.shape', class_no.shape)
        # for i in range(len(pre.shape(0))):
        #     for j in range(len(pre.shape(2))):
        #         for k in range(len(pre.shape(3))):
        #             for l in range(len(pre.shape(4))):
        #                 if out[i, 1, j, k, l] < 0.7:
        #                     pre[i, 0, j, k, l] = 0
        #                 else:
        #                     pre[i, 0, j, k, l] = 1

        return out


    # Functions to get output from intermediate feature layers
    def f1(self, x,):
        return self.getFeatures( x, 0)
    def f2(self, x,):
        return self.getFeatures( x, 1)
    def f3(self, x,):
        return self.getFeatures( x, 2)
    def f4(self, x,):
        return self.getFeatures( x, 3)
    def f5(self, x,):
        return self.getFeatures( x, 4)


    def getFeatures(self, x, layer_no):
        layer_indexes = [0, 3, 6, 9, 12]

        #Make new network that has the layers up to the requested output
        tmp_net = nn.Sequential()
        layers = list(self.net.children())[0:layer_indexes[layer_no]+1]
        for i in range(len(layers)):
            tmp_net.add_module(str(i),layers[i])
        if type(gpu_no_of_var(self)) == int:
            tmp_net.cuda(gpu_no_of_var(self))
        return tmp_net(x)

if __name__ == "__main__":
    x = torch.rand(size=(32, 1, 32, 32, 32))
    net = TextureNet()
    out = net(x)
    print(out.size())

