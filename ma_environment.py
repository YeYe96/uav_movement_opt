import numpy as np
import math
from uav import UAV


#NUM_UAV = 10
np.random.seed(1234)
pop_dens_list = np.random.randint(0,100,size=(26,26))


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

'''
class Uav:
    def __init__(self,position_x,positopn_y,position_z,action,action_size):
        self.position_x = position_x
        self.position_y = positopn_y
        self.position_z = position_z
        self.action = action
        self.action_size = action_size
        self.neighbors = []
'''

class Environment:
    def __init__(self,nUAV):
        self.nUAV = nUAV
        self.action_space = UAV.action_space


    def list_sqadd(self,a,b):
        c = []
        for i in range(len(a)):
            c.append((a[i]-b[i])**2)
        return c

    def step(self,actions):
        self.uavs_act = []
        self.states = []
        for idx,a in enumerate(actions):
            print("states: ", self.uavs[idx].get_states())
            self.uavs_act.append(self.uavs[idx].act(a))
#            uav_coord = np.concatenate(self.uavs[idx].coord_x,
#                                  self.uavs[idx].coord_y,
#                                  self.uavs[idx].coord_z)
#            self.states.append(uav_coord)
            output = {
            "states": self.states,
            "states_":self.get_states(self.nUAV),
            "power_uav": self.get_rewards(self.nUAV),
            }
        reward = self.calculate_reward()
        return output,reward

    def get_states(self,nuav):
        return [self.uavs[idx].get_states() for idx in nuav]

    def get_rewards(self,nuav):
        return  [self.uavs[idx].get_reward() for idx in nuav]

    def reset(self):
        '''
        参数暂时不加
        '''
        self.uavs = [UAV(0,0,0,150,id) for id in range(self.nUAV)]
        self.bs_state = np.array([0, 1, 1, 0])
        self.bs_coord = [[7, 7], [7, 19], [19, 7], [19, 19]]
        self.final_rate = []


    def calculate_reward(self):
        output = 0
        rate_list = []
        for i in range(26):
            pop_dens_list_row = [lis[i] for lis in pop_dens_list]
            for j in range(26):
                user_dens = pop_dens_list_row[j]
                user_coord = [j,i]
                distance = []
                bs_dist = []
                for k in range(4):
                    if self.bs_state[k] == 1:
                        if self.bs_state[k] == 1:
                            bs_dist.append(math.sqrt((height_bs**2) + np.sum(self.list_sqadd(self.bs_coord[k],user_coord))))
                            power_bs = (trans_power * gt * gr *(c / freq) **2 )/ 4 * math.pi * min(bs_dist)
                    else:
                        power_bs = 0
                distance.append(min(bs_dist))   #找到最近的MBS
                uav_dist = np.array([0 for num in range(self.nUAV)])

                power_uav = [0 for i in range(10)]
#                uav_dist = []
                count_idx = -1
                for nuav in self.uavs:
                    count_idx += 1
                    uav_cal_coord = np.hstack(nuav.coord_x,nuav.coord_y)
                    uav2user_dist = math.sqrt((nuav.coord_z**2) + self.list_sqadd(uav_cal_coord,user_coord))*10
#                    uav2uav_coord.append(np.hstack(nuav.coord_x,nuav.coord_y,nuav.coord_z))
                    if uav2user_dist < 60:
                        theta = math.atan(nuav.coord_z / uav2user_dist)
                        probability_los = (1 + alpha * math.exp(-belta * ((180 / math.pi) * theta - alpha))) ** (-1)
                        probability_nlos = 1 - probability_los
                        uav_los = (probability_los * mulos)
                        uav_nlos = (probability_nlos * munlos)
                        power_uav[count_idx] = trans_power * ((4 * math.pi * freq / c) ** (-2)) * (1 / uav2user_dist) * (
                                uav_los + uav_nlos) ** (-1)
                        nuav.update_reward(power_uav[count_idx])
                        break
                #计算uav间距离

                power_bs_interference = [0]
                if power_bs > max(power_uav):
                    for l in range(len(bs_dist)):
                        if bs_dist[l] != min(bs_dist):
                            power_bs_interference.append((trans_power * gt * gr * (c / freq) ** 2) / 4 * math.pi * bs_dist[l]*10)
                        else:
                            power_bs_interference.append(0)
                    interference_bs = sum(power_bs_interference)
                    SINR = power_bs/((N0 * bandwidth + interference_bs + sum(power_uav)))
                else:
                    for l in  range(len(bs_dist)):
                        if bs_dist[l] < 7:
                            power_bs_interference.append(
                                (trans_power * gt * gr * (c / freq) ** 2) * self.bs_state[l] / 4 * math.pi * bs_dist[l]*10)
                        else:
                            power_bs_interference.append(0)
                    interference_bs = sum(power_bs_interference)
                    SINR = max(power_uav) / ((N0 * bandwidth + interference_bs + sum(power_uav) - max(power_uav)))

                rate_list.append((bandwidth * math.log2(1 + SINR))*user_dens)
        self.final_rate.append(sum(rate_list))

        return sum(self.final_rate) / 1500

if __name__=="__main__":
    nUAV = 10
    env = Environment(nUAV)
    env.reset()
    print(env.uavs)
    for i in range(1000):
        actions = np.random.randint(0,6,nUAV)
        print(env.step(actions))
        continue
