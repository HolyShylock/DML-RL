import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TaskData(Dataset):
    def __init__(self, path, loader):
        self.data = self.__read(path)
        self.loader = loader

    def __getitem__(self, item):
        p_s, p_a, p_r, p_s_, p_done, label = self.data[item]
        s = self.loader(p_s)
        a = self.loader(p_a)
        r = self.loader(p_r)
        s_ = self.loader(p_s_)
        done = self.loader(p_done)
        return s, a, r, s_, done, label

    def __len__(self):

        return len(self.data)
    
    def __read(self, path):
        data = []
        labels = os.listdir(path)
        count = 0

        for label in labels:
            pwd = path + label + '/'
            State = pwd + 'State/'
            Action = pwd + 'Action/'
            Reward = pwd + 'Reward/'
            State_Next = pwd + 'State_Next/'
            Done = pwd + 'Done/'
            N = len(os.listdir(State))
            count += 1

            for i in range(1, N + 1):
                tmp_s = State + 's' + str(i) + '.npy'
                tmp_a = Action + 'a' + str(i) + '.npy'
                tmp_r = Reward + 'r' + str(i) + '.npy'
                tmp_s_ = State_Next + 's_' + str(i) + '.npy'
                tmp_d = Done + 'd' + str(i) + '.npy'
                data.append([tmp_s, tmp_a, tmp_r, tmp_s_, tmp_d, count])
        
        return data

def loader(path):

    return np.load(path,  allow_pickle = True)

def TaskDataLoad(path, batch, shuffle = True):
    taskData = TaskData(path, loader)
    taskDataLoader = DataLoader(taskData, batch, shuffle = True, num_workers = 0, pin_memory = True)
    
    return taskDataLoader
