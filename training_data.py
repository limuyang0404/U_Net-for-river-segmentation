# coding=UTF-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
import random
from os.path import isfile, join, getsize
import segyio
import pandas as pd
from collections import Counter
'''this function is made to get a Two-dimensional dimensions with mirrored edges'''

def label_location_read(filename):
    '''filename should be the path of a binary file which have 5 columns representing Inline,Xline,X,Y,time'''
    datain = np.loadtxt(filename)  # read river file
    nan_number = 0
    deleted_row = []

    # delete NaN
    # for i in range(datain.shape[0]):
    #     if datain[i, 2] != datain[i, 2]:
    #         nan_number += 1
    #         deleted_row.append(i)
    # datain = np.delete(datain, deleted_row, axis=0)

    # print('nan_number is :', nan_number)
    # np.savetxt(filename[:-4] + '_edit' + ".txt", datain, delimiter="\n")
    # print(filename)
    print(datain.shape)
    label_location_df = pd.DataFrame(data=datain, columns=['INLINE', 'XLINE', 'LOCAL_X', 'LOCAL_Y', 'TIME'])
    print(label_location_df)
    label_location_df['TIME'] = label_location_df['TIME'].apply(lambda x: round(x))  # make all time rounded
    # label_location_df['TIME'] = label_location_df['TIME'].astype('int32')  # make all time rounded
    label_location_df[['INLINE', 'XLINE']] = label_location_df[['INLINE', 'XLINE']].astype('int32')
    return label_location_df  # a dataframe of inline, xline, source coordination X and Y ,time


def horizon_into_cube(data_cube, local, horizon_list):
    label_iline_start = label_xline_start = label_time_start = 999999
    label_iline_end = label_xline_end = label_time_end = 0
    df = []
    for m in range(len(horizon_list)):
        df.append(label_location_read(horizon_list[m]))
        label_iline_start_new, label_iline_end_new, label_xline_start_new, label_xline_end_new, label_time_start_new, label_time_end_new = \
            label_location(df[m])
        if label_iline_start_new < label_iline_start:
            label_iline_start = label_iline_start_new
        if label_xline_start_new < label_xline_start:
            label_xline_start = label_xline_start_new
        if label_time_start_new < label_time_start:
            label_time_start = label_time_start_new
        if label_iline_end_new > label_iline_end:
            label_iline_end = label_iline_end_new
        if label_xline_end_new > label_xline_end:
            label_xline_end = label_xline_end_new
        if label_time_end_new > label_time_end:
            label_time_end = label_time_end_new
    print('$' * 50)
    label_cube = data_cube*1
    label_cube[:, :, :] = 0
    data_iline_start, data_xline_start, data_time_start, data_sample_interval = local[0], local[1], local[2], local[3]
    data_cube = data_cube[label_iline_start - data_iline_start:label_iline_end - data_iline_start + 1, \
                  label_xline_start - data_xline_start:label_xline_end - data_xline_start + 1,
                  int((label_time_start - data_time_start) / data_sample_interval):int(
                      (label_time_end - data_time_start + 1) / data_sample_interval)]  # cut from segy file
    label_cube = label_cube[label_iline_start - data_iline_start:label_iline_end - data_iline_start + 1, \
                   label_xline_start - data_xline_start:label_xline_end - data_xline_start + 1,
                   int((label_time_start - data_time_start) / data_sample_interval):int(
                       (label_time_end - data_time_start + 1) / data_sample_interval)]  # slice from source data cube
    for m in range(len(horizon_list)):
        array_from_dataframe = df[m].to_numpy()
        for i in range(array_from_dataframe.shape[0]):
            inline = array_from_dataframe[i][0]
            xline = array_from_dataframe[i][1]
            time = array_from_dataframe[i][4]  # the time have been rouned at read_river
            # print(inline, label_iline_start, xline, label_xline_start, time, label_time_start, data_sample_interval)
            label_cube[int(inline - label_iline_start), int(xline - label_xline_start), int((time - label_time_start)/data_sample_interval)] = m+1
    print(label_iline_start, label_iline_end, label_xline_start, label_xline_end, label_time_start, label_time_end)

    return data_cube, label_cube, label_iline_start, label_xline_start, label_time_start

def label_location(river_dataframe):  # get the xyz interval of the river
    source_iline_min = river_dataframe['INLINE'].min()
    source_iline_max = river_dataframe['INLINE'].max()
    source_xline_min = river_dataframe['XLINE'].min()
    source_xline_max = river_dataframe['XLINE'].max()
    time_min = river_dataframe['TIME'].min()
    time_max = river_dataframe['TIME'].max()
    row_number = river_dataframe.shape[0]
    print('source_iline_min, source_iline_max, source_xline_min, source_xline_max, time_min, time_max, row_number=\n',
          source_iline_min, source_iline_max, source_xline_min, source_xline_max, time_min, time_max, row_number)
    return source_iline_min, source_iline_max, source_xline_min, source_xline_max, time_min, time_max


class TrainingData(object):
    def __init__(self, input, horizon):
        self._data = []
        self._location = {}
        self._samples = []
        self._data_type = input
        self._horizon = horizon
        self.cube = []
        self.data_out = []
        self.label_out = []
        self.location_out = []
        self.data_out_ilinestart = []
        self.data_out_xlinestart = []
        self.data_out_timestart = []
    def data(self):
        if self._data_type[0][-3: ] == 'bin':
            self._data = np.fromfile(self._data_type[0], dtype='float32', count=-1, sep='').reshape(self._data_type[1][0],self._data_type[1][1],self._data_type[1][2])
            pass
        elif self._data_type[0][-3: ] == 'sgy':
            self._data = segyio.tools.cube(self._data_type[0])
            pass
        return
    def location(self):
        if self._data_type[0][-3: ] == 'bin':
            self._location['inline_start'] = self._data_type[2][0]
            self._location['xline_start'] = self._data_type[2][1]
            self._location['time_start'] = self._data_type[2][2]
            self._location['sample_interval'] = self._data_type[2][3]
            self.location_out = self._location
            pass
        elif self._data_type[0][-3: ] == 'sgy':
            with segyio.open(self._data_type[0]) as f:
                self._location['inline_start'] = f.ilines[0]
                self._location['xline_start'] = f.xlines[0]
                self._location['time_start'] = int(f.samples[0])
                self._location['sample_interval'] = int(f.samples[1] - f.samples[0])
                self.location_out = self._location
            pass
        return
    def label_cube(self):
        if self._data_type[0][-3: ] == 'bin':
            self.data_out, self.label_out, self.data_out_ilinestart, self.data_out_xlinestart, self.data_out_timestart = horizon_into_cube(self._data, [self._location['inline_start'], self._location['xline_start'], self._location['time_start'], self._location['sample_interval']], self._horizon)
            pass
        elif self._data_type[0][-3: ] == 'sgy':
            self.data_out, self.label_out, self.data_out_ilinestart, self.data_out_xlinestart, self.data_out_timestart = horizon_into_cube(self._data, [self._location['inline_start'], self._location['xline_start'], self._location['time_start'], self._location['sample_interval']], self._horizon)
            pass
    pass

if __name__ == '__main__':
    input = TrainingData(['cb27_600-2000ms_final.sgy'], ['CB27_river.dat'])
    input.data()
    input.location()
    input.label_cube()
    print(input.location_out)
#
# a = TrainingData(['cb27_600-2000ms_final.sgy'], ['CB27_river.dat'])
# a.data()
# a.location()
# a.label_cube()
# print(type(a.data_out), type(a.label_out))
# print(a.data_out.shape, a.label_out.shape)
# print(Counter(a.label_out.flatten()))

# print(a.location())
