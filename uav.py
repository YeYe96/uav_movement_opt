import simpy
import random
import numpy as np

class UAV():
    action_space = np.array([0,1,2,3,4,5,6])
    action_space_size = 7

    #def __init__(self, env, init_floor, weightLimit, id) 原本有一个weightlimit 可以改为battery？
    def __init__(self,env,coord_x,coord_y,coord_z,battery,id):
        self.env = env
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z
        self.battery = battery
        self.id = id
        self.current_reward = 0
        self.last_decision_epoch = self.env.simenv.now


        self.ACTION_FUNCTION_MAP = {
            0: self._add_x,
            1: self._sub_x,
            2: self._add_y,
            3: self._sub_y,
            4: self._add_z,
            5: self._sub_z,
            6: self._hover,
        }

    def act(self,action):
        yield self.env.simenv.process(self.ACTION_FUNCTION_MAP[action]())
        self.battery -= 1

    def _add_x(self):
        if self.coord_x != 25:
            self.coord_x += 1

    def _sub_x(self):
        if self.coord_x != 0:
            self.coord_x -= 1

    def _add_y(self):
        if self.coord_y != 25:
            self.coord_y += 1

    def _sub_y(self):
        if self.coord_y != 0:
            self.coord_y -= 1

    def _add_z(self):
        if self.coord_z != 25:
            self.coord_z += 1

    def _sub_z(self):
        if self.coord_z != 0:
            self.coord_z -= 1

    def _hover(self):
        self.coord_x += 0
        self.coord_y += 0
        self.coord_z += 0

    def update_reward(self, reward):
        self.current_reward += reward
        return True

    def get_reward(self,decision_epoch):
        output = self.current_reward
        if decision_epoch:
            self.current_reward = 0
        return output

    def get_states(self,decision):
        state_representation = np.concatenate([
            self.coord_x,
            self.coord_y,
            self.coord_z
        ])
        return state_representation

