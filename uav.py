import random
import numpy as np

class UAV():
    action_space = np.array([0,1,2,3,4,5,6])
    action_space_size = 7

    def __init__(self,coord_x,coord_y,coord_z,battery,id):
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z
        self.battery = battery
        self.id = id
        self.current_reward = 0

    def act(self,action):
        if action == 0:
            self._add_x()
        if action == 1:
            self._sub_x()
        if action == 2:
            self._add_y()
        if action == 3:
            self._sub_y()
        if action == 4:
            self._add_z()
        if action == 5:
            self._sub_z()
        if action == 6:
            self._hover()
        self.battery -= 1
        return self.coord_x,self.coord_y,self.coord_z

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
        if self.coord_z != 0 & self.coord_z >1:
            self.coord_z -= 1

    def _hover(self):
        self.coord_x += 0
        self.coord_y += 0
        self.coord_z += 0

    def update_reward(self, reward):
        self.current_reward += reward
        return True

    def get_reward(self):
        output = self.current_reward
        return output

    def get_states(self):
        state_representation = np.hstack([
            self.coord_x,
            self.coord_y,
            self.coord_z,
            self.battery
        ])
        return state_representation

