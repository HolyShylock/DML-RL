import gym
from getTaskData import Collecter
from TaskDataLoad import TaskDataLoad
import gym
import gym_minigrid
from gym_minigrid.wrappers import *

# for i in range(2):
#     env = gym.make('MiniGrid-Empty-Random-5x5-v0')
#     env = RGBImgPartialObsWrapper(env) # Get pixel observations
#     env = ImgObsWrapper(env) # Get rid of the 'mission' field
#     collecter = Collecter('task' + str(i))
#     collecter.collect(env, 100, 3)

task_data_loader = TaskDataLoad('DML/dataSet/', 2)
flag = 0
for id, data in enumerate(task_data_loader):
    if flag == 0:
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(data[3].shape)
        print(data[4].shape)
        flag = 1
    else:
        break
# print(s.shape)
# print(a)
# print(r)
# print(done)
