# coding=UTF-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
import random
from os.path import isfile, join, getsize
from texture_net import TextureNet
import torch
from torch import nn
from collections import Counter
import math
from training_data import TrainingData
'''this function is made to get a Two-dimensional dimensions with mirrored edges'''


def predict(net_output):
    '''(1, 2, 32, 32, 32)'''
    net_output = nn.Softmax(dim=1)(net_output)
    net_output = torch.squeeze(net_output)
    net_output = net_output.cpu()
    array = (net_output).detach().numpy()
    # (9, 56, 56, 56)
    # print(array.shape)
    tensor_size0, tensor_size1, tensor_size2, tensor_size3 = net_output.size(0), net_output.size(1), net_output.size(2), net_output.size(3)
    predict_result = np.zeros((tensor_size1, tensor_size2, tensor_size3))
    # print(tensor_size0, tensor_size1, tensor_size2, tensor_size3)
    for i in range(tensor_size1):
        for j in range(tensor_size2):
            for l in range(tensor_size3):
                value = 0
                local_key = 0
                for k in range(tensor_size0):
                    # print(i, j, k, array[k, i, j])
                    # value += array[k, i, j, l]*k       #like a vote process
                    if array[k, i, j, l] > value:
                        value = array[k, i, j, l]
                        local_key = k
                predict_result[i, j, l] = local_key
                # print(value)
                # print(i, j, array[:, i, j], value)
                # print(i, j, value)

    return predict_result

def batch_generate(data_cube, label_cube, batch_size=32):
    data_output = np.zeros((1, 1, batch_size, batch_size, batch_size))
    label_output = np.zeros((1, batch_size, batch_size, batch_size))
    data_output[0, 0, :, :, :] = data_cube
    label_output[0, :, :, :] = label_cube
    return data_output, label_output


if __name__ == '__main__':
    network = TextureNet(n_classes=2)
    batch_size = 32
    cross_entropy = nn.CrossEntropyLoss()  # Softmax function is included
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    # Transfer model to gpu
    if torch.cuda.device_count() > 1:
        network = nn.DataParallel(network)
    network.to(device)
    # network.load_state_dict = (checkpoint)    #pytorch调用先前的模型参数
    network.eval()
    # optimizer.load_state_dict(torch.load(join('F3','optimizer.pth')))
    model_path = 'saved_model.pt'
    network.module.load_state_dict(torch.load(model_path))
    # data_cube = np.zeros((784, 336, 616))
    # label_cube = np.zeros((784, 336, 616))
    # predict_cube = np.zeros((784, 336, 616))
    input = TrainingData(['cb27_600-2000ms_final.sgy'], ['CB27_river.dat'])
    input.data()
    input.location()
    input.label_cube()
    # data = np.fromfile('cut.bin', dtype='float32', count=-1, sep='').reshape(356,462,161)
    data_cube = input.data_out
    label_cube = input.label_out
    # data_cube, label_cube = training_area(data_cube, label_cube)
    size_0 = math.ceil(data_cube.shape[0]/batch_size)
    size_1 = math.ceil(data_cube.shape[1]/batch_size)
    size_2 = math.ceil(data_cube.shape[2]/batch_size)
    print(size_0, size_1, size_2)
    data_cube_huge = np.zeros((size_0 * batch_size, size_1 * batch_size, size_2 * batch_size))
    label_cube_huge = np.zeros((size_0 * batch_size, size_1 * batch_size, size_2 * batch_size))
    predict_cube = np.zeros((size_0 * batch_size, size_1 * batch_size, size_2 * batch_size))
    # data_cube = np.zeros((math.ceil(data_cube.shape[0]/32), math.ceil(data_cube.shape[1]/32), math.ceil(data_cube.shape[2]/32)))
    data_cube_huge[0:data_cube.shape[0], 0:data_cube.shape[1], 0:data_cube.shape[2]] = data_cube
    label_cube_huge[0:data_cube.shape[0], 0:data_cube.shape[1], 0:data_cube.shape[2]] = label_cube
    print(data_cube.shape, predict_cube.shape)
    # data_cube[0:751, 0:320, 0:561] = np.fromfile('cut_seismic_data.bin', dtype='float32', count=-1, sep='').reshape(751, 320, 561)
    # label_cube[0:751, 0:320, 0:561] = np.fromfile('cut_seismic_data_label.bin', dtype='float32', count=-1, sep='').reshape(751, 320, 561)
    for i in range(size_0):
        for j in range(size_1):
            for k in range(size_2):
                data = data_cube_huge[i * batch_size:i * batch_size + batch_size, j * batch_size:j * batch_size + batch_size, k * batch_size:k * batch_size + batch_size]
                label = label_cube_huge[i * batch_size:i * batch_size + batch_size, j * batch_size:j * batch_size + batch_size, k * batch_size:k * batch_size + batch_size]
                data, label = batch_generate(data, label)
                data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
                label = (torch.autograd.Variable(torch.Tensor(label).long())).to(device)
                output = network(data)
                print(output.size())
                loss = cross_entropy(output, label)
                print(loss)
                print(i, j, k)
                predict_result = predict(output)
                predict_cube[i * batch_size:i * batch_size + batch_size, j * batch_size:j * batch_size + batch_size, k * batch_size:k * batch_size + batch_size] = predict_result
            pass
        pass
    pass
    print(data_cube.shape[0], data_cube.shape[1], data_cube.shape[2])
    # predict_cube = predict_cube[0:data_cube.shape[0], 0:data_cube[1], 0:data_cube[2]]
    predict_cube_output = predict_cube[0:data_cube.shape[0], 0:data_cube.shape[1], 0:data_cube.shape[2]]
    river_list = []
    for i in range(predict_cube_output.shape[0]):
        for j in range(predict_cube_output.shape[1]):
            for k in range(predict_cube_output.shape[2]):
                if predict_cube_output[i, j, k] == 1:
                    print(i, j, k)
                    river_list.append([i + input.data_out_ilinestart, j + input.data_out_xlinestart, k * input.location_out['sample_interval'] + input.data_out_timestart])
    np.savetxt('horizon_segmentation.dat', river_list)
    predict_cube_output = predict_cube_output.reshape(data_cube.shape[0]*data_cube.shape[1], data_cube.shape[2])
    # predict_cube.tofile('cut_seismic_data_predict')
    np.savetxt("cut_seismic_data_predict.dat", predict_cube_output)
    # predict_cube.flatten()
    # print(Counter(predict_cube))


