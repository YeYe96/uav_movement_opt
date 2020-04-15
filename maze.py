'''
ABS MOVEMENT BASED RL(ENV)
YeYe
2020/2/19
'''
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import math

UNIT = 40
MAZE_H = 26
MAZE_W = 26

class Maze(tk.Tk,object):
    def __init__(self):
        super(Maze,self).__init__()
        self.action_space = ['u','d','l','r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H*UNIT,MAZE_W*UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self,bg='white',height = MAZE_H*UNIT,width = MAZE_W*UNIT)

        # create grid
        for c in range(0,MAZE_W*UNIT):
            x0,y0,x1,y1 = c,0,c,MAZE_H*UNIT
            self.canvas.create_line(x0,y0,x1,y1)
        for r in range(0,MAZE_H*UNIT):
            x0,y0,x1,y1 = 0,r,MAZE_W*UNIT,r
            self.canvas.create_line(x0,y0,x1,y1)

        # 在网格图上绘制基站中心点
        origin = np.array([20,20])
        self.coordinate = [0,0]

        self.base_station_0_center = origin + np.array([UNIT * 6, UNIT * 6])
        self.base_station_1_center = origin + np.array([UNIT * 19, UNIT * 6])
        self.base_station_2_center = origin + np.array([UNIT * 6, UNIT * 19])
        self.base_station_3_center = origin + np.array([UNIT * 19, UNIT * 19])

        # 在网格图上绘制基站，预设有四个基站，分别在四个方位的中心
        self.base_station_0_rect = self.canvas.create_rectangle(self.base_station_0_center[0]-15, self.base_station_0_center[1]-15,
                                                                  self.base_station_0_center[0]+15, self.base_station_0_center[1]+15,
                                                                  fill = 'blue')
        self.base_station_1_rect = self.canvas.create_rectangle(self.base_station_1_center[0]-15, self.base_station_1_center[1]-15,
                                                                  self.base_station_1_center[0]+15, self.base_station_1_center[1]+15,
                                                                  fill = 'blue')
        self.base_station_2_rect = self.canvas.create_rectangle(self.base_station_2_center[0]-15, self.base_station_2_center[1]-15,
                                                                  self.base_station_2_center[0]+15, self.base_station_2_center[1]+15,
                                                                  fill = 'blue')
        self.base_station_3_rect = self.canvas.create_rectangle(self.base_station_3_center[0]-15, self.base_station_3_center[1]-15,
                                                                  self.base_station_3_center[0]+15, self.base_station_3_center[1]+15,
                                                                  fill = 'blue')

        self.UAV_1_rect = self.canvas.create_rectangle(origin[0]-15, origin[1]-15,
                                                       origin[0]+15, origin[1]+15,
                                                       fill = 'yellow')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.UAV_1_rect)
        origin = np.array([20,20])
        self.UAV_1_rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                                                       origin[0] + 15, origin[1] + 15,
                                                       fill='yellow')
        self.coordinate = [0,0]
        #return observation
        return self.canvas.coords(self.UAV_1_rect[:2])/(MAZE_H * UNIT)

    def step(self, action):
        N0 = -174
        bandwidth = 180000
        freq = 2000000
        trans_power = 40
        c = 300000000
        pi = math.pi
        mulos = 3
        munlos = 23
        height = 50
        alpha = 0.3
        belta = 500
        gt = 1
        gr = 1
        battery = 200

        s = self.canvas.coords(self.UAV_1_rect)
        base_action = np.array([0,0])
        origin = np.array([20, 20])
        if action == 0:     #up
            if s[1]>UNIT & self.coordinate(1) != 0:
                battery -= 1
                base_action[1] -= UNIT
                self.coordinate -= [0,1]
        elif action == 1:       #down
            if s[1] < (MAZE_H - 1) * UNIT & self.coordinate(1) != 25:
                battery -= 1
                base_action[1] += UNIT
                self.coordinate += [0,1]
        elif action == 2:       #right
            if s[0] < (MAZE_W-1) * UNIT & self.coordinate(0) != 25:
                battery -= 1
                base_action[0]  += UNIT
                self.coordinate += [1,0]
        elif action == 3:       #left
            if s[0] > UNIT & self.coordinate(0) != 0:
                battery -= 1
                base_action[0] -= UNIT
                self.coordinate -= [1,0]
        self.canvas.move(self.UAV_1_rect, base_action[0], base_action[1])    #move agent

        s_ = self.canvas.coords(self.UAV_1_rect)    #next state
        coordinate_ = self.coordinate

        #user density
        pop_dens_list = np.random.randint(0,100,size=(26,26))
        rate_list = []
        for i in range(26):
            pop_dens_list_row = [lis[i] for lis in pop_dens_list]
            for j in range(26):
                user_dens = pop_dens_list_row[j]
                #连接基站策略
                bs_coord = [self.base_station_0_center,self.base_station_1_center,
                            self.base_station_2_center,self.base_station_3_center]
                user_coord = [origin(0) + j * UNIT , origin(1) + i * UNIT]
                distance = []
                bs_dist = []
                bs_dist_coord = []
                for k in range(4):
                    bs_dist_coord.append(bs_coord(k) - user_coord)
                    bs_dist.append(sum(bs_dist_coord(k)**2))
                distance.append(min(bs_dist))   #找到最近的MBS
                uav_dist = sum((self.coordinate - user_coord)**2)
                distance.append(uav_dist)
                #计算rate
                probability_los = (1+alpha*math.exp(-belta((180/pi)*math.asin(height/distance(1))-alpha)))**(-1)
                probability_nlos = 1-probability_los
                power_uav = trans_power*((4*pi*freq/c)**(-2))*(distance(1)**(-1))\
                            *(probability_los*mulos+probability_nlos*munlos)**(-1)
                power_bs = (trans_power*gt*gr*(c/freq)**2)/4*pi*distance(0)
                power_bs_interference = []
                if power_bs > power_uav:
                    for l in range(4):
                        if bs_dist(l) != min(bs_dist):
                            power_bs_interference.append((trans_power * gt * gr * (c / freq) ** 2) / 4 * pi * distance(l))
                        else:
                            power_bs_interference.append(0)
                    interference_bs = sum(power_bs_interference)
                    SINR = power_bs / (N0 * bandwidth + interference_bs + power_uav)
                else:
                    for l in range(4):
                        power_bs_interference.append((trans_power * gt * gr * (c / freq) ** 2) / 4 * pi * distance(l))
                    interference_bs = sum(power_bs_interference)
                    SINR = power_uav / (N0 * bandwidth + interference_bs + power_uav)
                rate_list.append((bandwidth * math.log2(1 + SINR))*user_dens)
        #reward function
        reward = sum(rate_list)
        print(self.coordinate)
        if battery == 0:
            done = True
        else:
            done = False
        return s_ , reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

'''
if __name__ == "__main__":
    env = Maze()
    env.mainloop()
'''