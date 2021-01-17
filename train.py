# Compatability Imports
#coding=utf-8
from __future__ import print_function
from os.path import join
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import tb_logger
from collections import Counter
from training_data import TrainingData
from texture_net import TextureNet
import numpy as np
#This is the network definition proposed in the paper
def training_area(data_cube, label_cube, retention=0.7):
    output_data = data_cube[0:int(data_cube.shape[0] * retention), 0:int(data_cube.shape[1] * retention), 0:data_cube.shape[2]]
    output_label = label_cube[0:int(label_cube.shape[0] * retention), 0:int(label_cube.shape[1] * retention), 0:label_cube.shape[2]]
    return output_data, output_label

def random_batch_generate(data_cube, label_cube, batch_size=32, batch_nember=25):
    counter = 0
    data_output = np.zeros((batch_nember, 1, batch_size, batch_size, batch_size))
    label_output = np.zeros((batch_nember, batch_size, batch_size, batch_size))
    while counter < batch_nember:
        random_index_0 = np.random.randint(0, data_cube.shape[0] - batch_size)  #
        random_index_1 = np.random.randint(0, data_cube.shape[1] - batch_size)
        random_index_2 = np.random.randint(0, data_cube.shape[2] - batch_size)
        data_output[counter, 0, :, :, :] = data_cube[random_index_0:random_index_0 + batch_size, random_index_1:random_index_1 + batch_size,
                      random_index_2:random_index_2 + batch_size]
        label_output[counter, :, :, :] = label_cube[random_index_0:random_index_0 + batch_size, random_index_1:random_index_1 + batch_size,
                      random_index_2:random_index_2 + batch_size]
        counter += 1
    return data_output, label_output

if __name__ == '__main__':
    # model_path = 'saved_model.pt'
    # optimizer_path = 'optimizer.pth'
    torch.cuda.empty_cache()
    network = TextureNet(n_classes=2)
    network.ParameterInitialize()
    cross_entropy = nn.CrossEntropyLoss()  # Softmax function is included
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    # Transfer model to gpu
    if torch.cuda.device_count() > 1:
        network = nn.DataParallel(network)
    network.to(device)
    # network.load_state_dict = (checkpoint)    #pytorch调用先前的模型参数
    network.eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)  # Adam method

    # optimizer.load_state_dict(torch.load(join('F3','optimizer.pth')))
    checkpoint = {'model': network.state_dict(), 'optimizer': optimizer.state_dict()}
    model_path = 'saved_model3.pt'
    optimizer_path = 'optimizer3.pth'
    # network.module.load_state_dict(torch.load(model_path))

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    # optimizer.load_state_dict(torch.load(optimizer_path))
    loss_list = []

    input = TrainingData(['cb27_600-2000ms_final.sgy'], ['CB27_river.dat'])
    input.data()
    input.location()
    input.label_cube()
    # data = np.fromfile('cut.bin', dtype='float32', count=-1, sep='').reshape(356,462,161)
    data_cube = input.data_out
    label_cube = input.label_out
    data_cube, label_cube = training_area(data_cube, label_cube)
    network.train()
    for z in range(10000):
        data, label = random_batch_generate(data_cube, label_cube)
        data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
        label = (torch.autograd.Variable(torch.Tensor(label).long())).to(device)
        output = network(data)
        loss = cross_entropy(output, label)
        print(r"The %d epoch's loss is:" % z, loss)
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if z % 100 == 0 and z > 0:
            torch.save(network.module.state_dict(), 'saved_model.pt')  # 网络保存为saved_model.pt
            torch.save(optimizer.state_dict(), 'optimizer.pth')

    x1 = range(len(loss_list))
    # plt.plot(x1, loss_list)
    # plt.savefig('loss.png')

    loss_list = np.array(loss_list)
    np.savetxt('loss3.csv', loss_list, fmt='%f', delimiter=' ')


