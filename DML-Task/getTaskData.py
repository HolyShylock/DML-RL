import numpy as np
import os

path_root = 'DML/dataSet'
class Collecter:
    def __init__(self, label, path = path_root):

        '''
        目录结构：
        
        data -> label1 -> State : 
                       -> Action
                       -> Reward
                       -> State_Next
                       -> Done
             -> label2 -> State
            
            ...
            ...

             -> labeln -> State
            
        '''
        self.path = path
        self.label = path + '/' + label
        self.state_set = self.label + '/' + 'State'
        self.action_set = self.label + '/' + 'Action'
        self.reward_set = self.label + '/' + 'Reward'
        self.state_next_set = self.label + '/' + 'State_Next'
        self.done_set = self.label + '/' + 'Done'
        self.__mkdir(self.path)
        self.__mkdir(self.label)
        self.__mkdir(self.state_set)
        self.__mkdir(self.action_set)
        self.__mkdir(self.reward_set)
        self.__mkdir(self.state_next_set)
        self.__mkdir(self.done_set)

    def __mkdir(self, path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

    def collect(self, env, size, n_action = 3):
        s = env.reset()
        tmp_s = self.state_set + '/' + 's'
        tmp_a = self.action_set + '/' + 'a'
        tmp_r = self.reward_set + '/' + 'r'
        tmp_s_ = self.state_next_set + '/' + 's_'
        tmp_d = self.done_set + '/' +'d'

        for i in range(1, size + 1):
            np.save(tmp_s + str(i), s)
            a = np.random.choice(n_action)
            np.save(tmp_a + str(i), a)
            s_, r, done, _ = env.step(a)
            np.save(tmp_s_ + str(i), s_)
            np.save(tmp_r + str(i), r)
            np.save(tmp_d + str(i), done)
            if done:
                s = env.reset()
            else:
                s = s_
        
        env.close()