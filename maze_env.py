'''
ABS MOVEMENT BASED RL(ENV)
YeYe
2020/2/19
'''
from __future__ import division

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import math
import datetime
UNIT = 20
MAZE_H = 26
MAZE_W = 26
NUM_UAV = 10
bs_state = np.array([1,0,1,0])

class Maze(tk.Tk,object):
    def __init__(self):
        super(Maze,self).__init__()
        self.action_space = ['u','d','l','r','s']
        self.n_actions = len(self.action_space)
#        self.n_actions_sum = len(self.action_space) * self.n_actions
        self.n_features = 2
#        self.n_features_sum = self.n_features * NUM_UAV
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H*UNIT,MAZE_W*UNIT))
        self._build_maze()
 #       print(bs_state)

    def _build_maze(self):
        self.canvas = tk.Canvas(self,bg='white',height = MAZE_H*UNIT,width = MAZE_W*UNIT)

        # create grid
        for c in range(0,MAZE_W*UNIT,UNIT):
            x0,y0,x1,y1 = c,0,c,MAZE_H*UNIT
            self.canvas.create_line(x0,y0,x1,y1)
        for r in range(0,MAZE_H*UNIT,UNIT):
            x0,y0,x1,y1 = 0,r,MAZE_W*UNIT,r
            self.canvas.create_line(x0,y0,x1,y1)

        # 在网格图上绘制基站中心点
        origin = np.array([10,10])
        self.coordinate = [10,10]

        self.base_station_0_center = origin + np.array([UNIT * 6, UNIT * 6])
        self.base_station_1_center = origin + np.array([UNIT * 19, UNIT * 6])
        self.base_station_2_center = origin + np.array([UNIT * 6, UNIT * 19])
        self.base_station_3_center = origin + np.array([UNIT * 19, UNIT * 19])

        # 在网格图上绘制基站，预设有四个基站，分别在四个方位的中心
        self.base_station_0_rect = self.canvas.create_rectangle(self.base_station_0_center[0]-8.25, self.base_station_0_center[1]-8.25,
                                                                  self.base_station_0_center[0]+8.25, self.base_station_0_center[1]+8.25,
                                                                  fill = 'blue')
        self.base_station_1_rect = self.canvas.create_rectangle(self.base_station_1_center[0]-8.25, self.base_station_1_center[1]-8.25,
                                                                  self.base_station_1_center[0]+8.25, self.base_station_1_center[1]+8.25,
                                                                  fill = 'blue')
        self.base_station_2_rect = self.canvas.create_rectangle(self.base_station_2_center[0]-8.25, self.base_station_2_center[1]-8.25,
                                                                  self.base_station_2_center[0]+8.25, self.base_station_2_center[1]+8.25,
                                                                  fill = 'blue')
        self.base_station_3_rect = self.canvas.create_rectangle(self.base_station_3_center[0]-8.25, self.base_station_3_center[1]-8.25,
                                                                  self.base_station_3_center[0]+8.25, self.base_station_3_center[1]+8.25,
                                                                  fill = 'blue')

        self.UAV_1_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_2_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_3_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_4_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_5_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_6_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_7_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_8_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_9_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.05)
        rate_old = 0
        self.reward = 0
        self.battery = 150
        self.canvas.delete(self.UAV_0_rect)
        self.canvas.delete(self.UAV_1_rect)
        self.canvas.delete(self.UAV_2_rect)
        self.canvas.delete(self.UAV_3_rect)
        self.canvas.delete(self.UAV_4_rect)
        self.canvas.delete(self.UAV_5_rect)
        self.canvas.delete(self.UAV_6_rect)
        self.canvas.delete(self.UAV_7_rect)
        self.canvas.delete(self.UAV_8_rect)
        self.canvas.delete(self.UAV_9_rect)
        self.final_rate = []
        origin = np.array([10,10])
        self.base_action_list = np.array([[10, 10] for i in range(NUM_UAV)])
#        self.coord_list = np.zeros([10,2])
#        print('')
        self.UAV_0_rect = self.canvas.create_rectangle(origin[0]-8.25, origin[1]-8.25,
                                                       origin[0]+8.25, origin[1]+8.25,
                                                       fill = 'yellow')
        self.UAV_1_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_2_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_3_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_4_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_5_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_6_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_7_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_8_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.UAV_9_rect = self.canvas.create_rectangle(origin[0] - 8.25, origin[1] - 8.25,
                                                       origin[0] + 8.25, origin[1] + 8.25,
                                                       fill='yellow')
        self.coordinate = np.array([[10,10] for i in range(NUM_UAV)])
        #return observation
#        print('action_space:',self.action_space)
#        print('n_actions',self.n_actions)
#        print('reset begins')
        '''
        print(np.array([np.array([self.canvas.coords(self.UAV_0_rect)[0]/(MAZE_W * UNIT),
                                             self.canvas.coords(self.UAV_0_rect)[1]/(MAZE_H * UNIT)]),
                          np.array([self.canvas.coords(self.UAV_1_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_1_rect)[1] / (MAZE_H * UNIT)]),
                          np.array([self.canvas.coords(self.UAV_2_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_2_rect)[1] / (MAZE_H * UNIT)]),
                          np.array([self.canvas.coords(self.UAV_3_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_3_rect)[1] / (MAZE_H * UNIT)]),
                          np.array([self.canvas.coords(self.UAV_4_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_4_rect)[1] / (MAZE_H * UNIT)]),]))
        print(np.hstack((np.array([self.canvas.coords(self.UAV_0_rect)[0]/(MAZE_W * UNIT),
                                             self.canvas.coords(self.UAV_0_rect)[1]/(MAZE_H * UNIT)]),
                          np.array([self.canvas.coords(self.UAV_1_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_1_rect)[1] / (MAZE_H * UNIT)]),
                          np.array([self.canvas.coords(self.UAV_2_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_2_rect)[1] / (MAZE_H * UNIT)]),
                          np.array([self.canvas.coords(self.UAV_3_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_3_rect)[1] / (MAZE_H * UNIT)]),
                          np.array([self.canvas.coords(self.UAV_4_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_4_rect)[1] / (MAZE_H * UNIT)]),
                          )))
        '''
        return np.hstack([self.canvas.coords(self.UAV_0_rect)[0]/(MAZE_W * UNIT),
                                             self.canvas.coords(self.UAV_0_rect)[1]/(MAZE_H * UNIT),self.canvas.coords(self.UAV_1_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_1_rect)[1] / (MAZE_H * UNIT),
                          self.canvas.coords(self.UAV_2_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_2_rect)[1] / (MAZE_H * UNIT),
                          self.canvas.coords(self.UAV_3_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_3_rect)[1] / (MAZE_H * UNIT),
                          self.canvas.coords(self.UAV_4_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_4_rect)[1] / (MAZE_H * UNIT),
                         self.canvas.coords(self.UAV_5_rect)[0] / (MAZE_W * UNIT),
                                   self.canvas.coords(self.UAV_5_rect)[1] / (MAZE_H * UNIT),
                         self.canvas.coords(self.UAV_6_rect)[0] / (MAZE_W * UNIT),
                                   self.canvas.coords(self.UAV_6_rect)[1] / (MAZE_H * UNIT),
                         self.canvas.coords(self.UAV_7_rect)[0] / (MAZE_W * UNIT),
                                   self.canvas.coords(self.UAV_7_rect)[1] / (MAZE_H * UNIT),
                         self.canvas.coords(self.UAV_8_rect)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.UAV_8_rect)[1] / (MAZE_H * UNIT),
                         self.canvas.coords(self.UAV_9_rect)[0] / (MAZE_W * UNIT),
                                   self.canvas.coords(self.UAV_9_rect)[1] / (MAZE_H * UNIT)]),rate_old

    def step(self, action, rate_old, pop_dens_list):
        N0 = 174
        bandwidth = 180000
        freq = 2000000
        trans_power = 4000
        c = 300000000
        mulos = 3
        munlos = 23
        height_uav = 50
        height_bs = 100
        alpha = 0.3
        belta = 500
        gt = 0
        gr = 1
#        print('step begins')
        self.battery -= 1
#        pop_dens_list = np.random.randint(0, 100, size=[26, 26])
        s = self.canvas.coords(self.UAV_0_rect)

        origin = np.array([10, 10])
#        print(self.coord_list)
        # move agent
        for num in range(NUM_UAV):
            if action[num] == 0:  # up
                if self.coordinate[num][1] > 20:
                    self.base_action_list[num][1] -= UNIT
                    self.coordinate[num] -= [0, 20]
            elif action[num] == 1:  # down
                if self.coordinate[num][1] < 495:
                    self.base_action_list[num][1] += UNIT
                    self.coordinate[num] += [0, 20]
            elif action[num] == 2:  # right
                if self.coordinate[num][0] < 495:
                    self.base_action_list[num][0] += UNIT
                    self.coordinate[num] += [20, 0]
            elif action[num] == 3:  # left
                if self.coordinate[num][0] > 20 :
                    self.base_action_list[num][0] -= UNIT
                    self.coordinate[num] -= [20, 0]
            elif action[num] == 4:  # stay
                self.coordinate[num] -= [0, 0]
#        print('coord: ', self.coordinate)
        # move agent
        self.canvas.delete(self.UAV_0_rect)
        self.canvas.delete(self.UAV_1_rect)
        self.canvas.delete(self.UAV_2_rect)
        self.canvas.delete(self.UAV_3_rect)
        self.canvas.delete(self.UAV_4_rect)
        self.canvas.delete(self.UAV_5_rect)
        self.canvas.delete(self.UAV_6_rect)
        self.canvas.delete(self.UAV_7_rect)
        self.canvas.delete(self.UAV_8_rect)
        self.canvas.delete(self.UAV_9_rect)
        self.UAV_0_rect = self.canvas.create_rectangle(self.base_action_list[0][0] - 8.25, self.base_action_list[0][1] - 8.25,
                                                        self.base_action_list[0][0] + 8.25, self.base_action_list[0][1] + 8.25,
                                                       fill = 'yellow')
        self.UAV_1_rect = self.canvas.create_rectangle(self.base_action_list[1][0] - 8.25, self.base_action_list[1][1] - 8.25,
                                                        self.base_action_list[1][0] + 8.25, self.base_action_list[1][1] + 8.25,
                                                       fill = 'yellow')
        self.UAV_2_rect = self.canvas.create_rectangle(self.base_action_list[2][0] - 8.25, self.base_action_list[2][1] - 8.25,
                                                        self.base_action_list[2][0] + 8.25, self.base_action_list[2][1] + 8.25,
                                                       fill = 'yellow')
        self.UAV_3_rect = self.canvas.create_rectangle(self.base_action_list[3][0] - 8.25, self.base_action_list[3][1] - 8.25,
                                                        self.base_action_list[3][0] + 8.25, self.base_action_list[3][1] + 8.25,
                                                       fill = 'yellow')
        self.UAV_4_rect = self.canvas.create_rectangle(self.base_action_list[4][0] - 8.25, self.base_action_list[4][1] - 8.25,
                                                        self.base_action_list[4][0] + 8.25, self.base_action_list[4][1] + 8.25,
                                                       fill = 'yellow')
        self.UAV_5_rect = self.canvas.create_rectangle(self.base_action_list[5][0] - 8.25, self.base_action_list[5][1] - 8.25,
                                                        self.base_action_list[5][0] + 8.25, self.base_action_list[5][1] + 8.25,
                                                       fill = 'yellow')
        self.UAV_6_rect = self.canvas.create_rectangle(self.base_action_list[6][0] - 8.25, self.base_action_list[6][1] - 8.25,
                                                        self.base_action_list[6][0] + 8.25, self.base_action_list[6][1] + 8.25,
                                                       fill = 'yellow')
        self.UAV_7_rect = self.canvas.create_rectangle(self.base_action_list[7][0] - 8.25, self.base_action_list[7][1] - 8.25,
                                                        self.base_action_list[7][0] + 8.25, self.base_action_list[7][1] + 8.25,
                                                       fill = 'yellow')
        self.UAV_8_rect = self.canvas.create_rectangle(self.base_action_list[8][0] - 8.25, self.base_action_list[8][1] - 8.25,
                                                        self.base_action_list[8][0] + 8.25, self.base_action_list[8][1] + 8.25,
                                                       fill = 'yellow')
        self.UAV_9_rect = self.canvas.create_rectangle(self.base_action_list[9][0] - 8.25, self.base_action_list[9][1] - 8.25,
                                                        self.base_action_list[9][0] + 8.25, self.base_action_list[9][1] + 8.25,
                                                       fill = 'yellow')


        next_uav_0_coords = self.canvas.coords(self.UAV_0_rect)
        next_uav_1_coords = self.canvas.coords(self.UAV_1_rect)
        next_uav_2_coords = self.canvas.coords(self.UAV_2_rect)
        next_uav_3_coords = self.canvas.coords(self.UAV_3_rect)
        next_uav_4_coords = self.canvas.coords(self.UAV_4_rect)
        next_uav_5_coords = self.canvas.coords(self.UAV_5_rect)
        next_uav_6_coords = self.canvas.coords(self.UAV_6_rect)
        next_uav_7_coords = self.canvas.coords(self.UAV_7_rect)
        next_uav_8_coords = self.canvas.coords(self.UAV_8_rect)
        next_uav_9_coords = self.canvas.coords(self.UAV_9_rect)#next state
        s_ = np.hstack([np.array([self.canvas.coords(self.UAV_0_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_0_rect)[1] / (MAZE_H * UNIT)]),
                  np.array([self.canvas.coords(self.UAV_1_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_1_rect)[1] / (MAZE_H * UNIT)]),
                  np.array([self.canvas.coords(self.UAV_2_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_2_rect)[1] / (MAZE_H * UNIT)]),
                  np.array([self.canvas.coords(self.UAV_3_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_3_rect)[1] / (MAZE_H * UNIT)]),
                  np.array([self.canvas.coords(self.UAV_4_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_4_rect)[1] / (MAZE_H * UNIT)]),
                  np.array([self.canvas.coords(self.UAV_5_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_5_rect)[1] / (MAZE_H * UNIT)]),
                  np.array([self.canvas.coords(self.UAV_6_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_6_rect)[1] / (MAZE_H * UNIT)]),
                  np.array([self.canvas.coords(self.UAV_7_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_7_rect)[1] / (MAZE_H * UNIT)]),
                  np.array([self.canvas.coords(self.UAV_8_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_8_rect)[1] / (MAZE_H * UNIT)]),
                  np.array([self.canvas.coords(self.UAV_9_rect)[0] / (MAZE_W * UNIT),
                            self.canvas.coords(self.UAV_9_rect)[1] / (MAZE_H * UNIT)])])
        '''
        s_ = np.hstack((np.array([next_uav_0_coords[0] / (MAZE_H * UNIT), next_uav_0_coords[1] / (MAZE_W * UNIT)]),
                        np.array([next_uav_1_coords[0] / (MAZE_H * UNIT), next_uav_1_coords[1] / (MAZE_W * UNIT)]),
                        np.array([next_uav_2_coords[0] / (MAZE_H * UNIT), next_uav_2_coords[1] / (MAZE_W * UNIT)]),
                        np.array([next_uav_3_coords[0] / (MAZE_H * UNIT), next_uav_3_coords[1] / (MAZE_W * UNIT)]),
                        np.array([next_uav_4_coords[0] / (MAZE_H * UNIT), next_uav_4_coords[1] / (MAZE_W * UNIT)]),
                        np.array([next_uav_5_coords[0] / (MAZE_H * UNIT), next_uav_5_coords[1] / (MAZE_W * UNIT)]),
                        np.array([next_uav_6_coords[0] / (MAZE_H * UNIT), next_uav_6_coords[1] / (MAZE_W * UNIT)]),
                        np.array([next_uav_7_coords[0] / (MAZE_H * UNIT), next_uav_7_coords[1] / (MAZE_W * UNIT)]),
                        np.array([next_uav_8_coords[0] / (MAZE_H * UNIT), next_uav_8_coords[1] / (MAZE_W * UNIT)]),
                        np.array([next_uav_9_coords[0] / (MAZE_H * UNIT), next_uav_9_coords[1] / (MAZE_W * UNIT)]),
                        )) 
        '''

        coordinate_ = self.coordinate
        def list_sqadd(a,b):
            c = []
            for i in range(len(a)):
                c.append((a[i]-b[i])**2)
            return c


#        print(pop_dens_list)
        rate_list = []
        for i in range(26):
            pop_dens_list_row = [lis[i] for lis in pop_dens_list]
            for j in range(26):
                user_dens = pop_dens_list_row[j]
                #连接基站策略
                bs_coord = [self.base_station_0_center,self.base_station_1_center,
                            self.base_station_2_center,self.base_station_3_center]
                user_coord = [origin[0] + j * UNIT , origin[1] + i * UNIT]
                distance = []
                bs_dist = []
                bs2user_coord = []
                for k in range(4):
                    if bs_state[k] == 1:
#                        print('k:  ', k)
                        bs_dist.append(math.sqrt((height_bs**2) + np.sum(list_sqadd(bs_coord[k],user_coord))))
                        power_bs = (trans_power * gt * gr * (c / freq) ** 2) / 4 * math.pi * min(bs_dist)
                    else:
                        power_bs = 0
                        '''
                        bs2user_coord.append(bs_coord[k] - user_coord)
                        bs_dist.append(sum(bs2user_coord[k] ** 2))
                        '''
                distance.append(min(bs_dist)) #找到最近的MBS
#                print('distence: ', distance)
#                print('bs2user_dist:  ', bs_dist)
                uav_dist = np.array([0 for num in range(NUM_UAV)])
                '''
                probability_los = []
                probability_nlos = []
                uav_los = []
                uav_nlos = []                
                '''
                power_uav = np.zeros([10,1])
                for num in range(NUM_UAV):
                    uav_dist[num] = math.sqrt((height_uav**2) + np.sum(list_sqadd(self.coordinate[num],user_coord)))
#                    print('coord: ',self.coordinate[num])
#                    print('user_coord: ',user_coord)
#                    print('sum: ',list_sqadd(self.coordinate[num],user_coord))
#                    print('uav2user_dist_%s:  ' %num,uav_dist)
#                    print(num)
                    if uav_dist[num] < 60:
                        uav_cal_dist = uav_dist[num]
#                        print('user cal dist:',uav_dist)
                        #                    print('user_cal_dist: ', uav_cal_dist)
                        theta = math.atan(height_uav / uav_cal_dist)
                        #                    print('theta: ', theta)
                        # probability_los = (1+alpha* math.exp(-belta((180/math.pi)*theta-alpha)))**(-1)
#                        print('theta: ',theta)
                        probability_los = (1 + alpha * math.exp(-belta * ((180 / math.pi) * theta - alpha))) ** (-1)
                        probability_nlos = 1 - probability_los
                        uav_los = (probability_los * mulos)
                        uav_nlos = (probability_nlos * munlos)
                        #                power_uav = trans_power*((4*math.pi*freq/c)**(-2))*(distance[1]**(-1))\
                        #                           *(uav_los + uav_nlos)**(-1)
                        power_uav[num] = trans_power * ((4 * math.pi * freq / c) ** (-2)) * (1 / uav_dist[num]) * (
                                uav_los + uav_nlos) ** (-1)
                        break
#                print('power uav: ', power_uav)
#                print(uav_dist)
#                print('max power: ', max(power_uav))
#                print('sum power: ', sum(power_uav))
                distance.append(uav_dist)

                #计算rate


                power_bs_interference = [0]
                if power_bs > max(power_uav):
                    for l in range(len(bs_dist)):
                        if bs_dist[l] != min(bs_dist):
                            power_bs_interference.append((trans_power * gt * gr * (c / freq) ** 2) / 4 * math.pi * bs_dist[l])
                        else:
                            power_bs_interference.append(0)
                    interference_bs = sum(power_bs_interference)
                    SINR = power_bs / (N0 * bandwidth + interference_bs + sum(power_uav))
#                    print('BS>UAV')
                else:
                    for l in range(len(bs_dist)):
                        power_bs_interference.append((trans_power * gt * gr * (c / freq) ** 2) / 4 * math.pi * bs_dist[l])
                    interference_bs = sum(power_bs_interference)
                    SINR = max(power_uav) / (N0 * bandwidth + interference_bs + sum(power_uav) - max(power_uav))
#                    print('rate: ', (bandwidth * math.log2(1 + SINR))*user_dens)
#                    print('power_uav:  ', max(power_uav))
#                    print('inter_bs:  ', interference_bs)
#                    print('sinr:  ', SINR)
                    rate_list.append((bandwidth * math.log2(1 + SINR))*user_dens)
#                    print('UAV>BS')
#                print('SINR: ',SINR)
#                print('power_bs:  ', power_bs)
#        print('power uav: ', power_uav)
#        print(rate_list)
#        print('rate: ', sum(rate_list))
        #reward function

        self.final_rate.append(sum(rate_list))
        self.reward = sum(self.final_rate)/1500

        '''
        if sum(rate_list)>rate_old:
            self.reward += 1
        elif sum(rate_list) == rate_old:
            self.reward += 0
        else:
            self.reward -= 1        
        '''
#        print('coordinate:',self.coordinate , '   ','rate' , sum(rate_list))
#        print('nows battery: ',self.battery,'  time: ',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if self.battery == 0:
            done = True
        else:
            done = False
#        print('step over')
        return s_ , self.reward, self.battery, sum(rate_list)

    def render(self):
        time.sleep(0.05)
        self.update()

'''
if __name__ == "__main__":
    env = Maze()
    env.mainloop()
'''