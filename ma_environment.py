import numpy as np
import math
from uav import UAV


#NUM_UAV = 10
#np.random.seed(1234)
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
        for idx,a in enumerate(actions):
#            print("states %d: " %(idx), self.uavs[idx].get_states())
            self.states = [self.uavs[idx].coord_x,
                           self.uavs[idx].coord_y,
                           self.uavs[idx].coord_z,
                           self.uavs[idx].battery]
            physic_action = np.argmax(a[idx])
            self.uavs_act.append(self.uavs[idx].act(physic_action))
#            uav_coord = np.concatenate(self.uavs[idx].coord_x,
#                                  self.uavs[idx].coord_y,
#                                  self.uavs[idx].coord_z)
#            self.states.append(uav_coord)
            states_ = self.get_states(self.nUAV)
            if self.uavs[idx].battery == 0:
                print('uav%d is back: ' %idx)
                self.uavs[idx].coord_x = 0
                self.uavs[idx].coord_y = 0
                self.uavs[idx].coord_z = 1
                self.uavs[idx].battery = 150
            output = {
            "states": self.states,
            "states_":self.get_states(self.nUAV),
            "power_uav": self.get_rewards(self.nUAV),
            }
#        print(actions)
#        print('output: ',output)
        reward,final_rate = self.calculate_reward()
        for idx in range(self.nUAV):
            reward[idx] = sum(reward[idx])
        return states_,reward

    def get_states(self,nuav):
        return [self.uavs[idx].get_states() for idx in range(nuav)]

    def get_rewards(self,nuav):
        return  [self.uavs[idx].get_reward() for idx in range(nuav)]

    def reset(self):
        '''
        参数暂时不加
        '''
        self.uavs = [UAV(0,0,1,150,id) for id in range(self.nUAV)]
        self.bs_state = np.array([0, 1, 1, 0])
        self.bs_coord = [[7, 7], [7, 19], [19, 7], [19, 19]]
        self.final_rate = []
        self.count = 0
        obs = []
        for idx in range(self.nUAV):
            obs.append(self.uavs[idx].get_states())
        return obs


    def calculate_reward(self):
        output = 0
        rate_list = []
        power_list = [[0*i] for i in range(self.nUAV)]
#        rate_list = [[0*i] for  i in range(nUAV)]
        for i in range(26):
            pop_dens_list_row = [lis[i] for lis in pop_dens_list]
            for j in range(26):
                power_uav = 0
                selected_uav_id = 0
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

#                power_uav = [0 for i in range(self.nUAV)]
#                uav_dist = []
                count_idx = -1
                for nuav in self.uavs:
                    count_idx += 1
                    uav_cal_coord = np.hstack((nuav.coord_x,nuav.coord_y))
                    uav2user_dist = math.sqrt((nuav.coord_z**2) + sum(self.list_sqadd(uav_cal_coord,user_coord)))*10
#                    uav2uav_coord.append(np.hstack(nuav.coord_x,nuav.coord_y,nuav.coord_z))
                    if uav2user_dist < 100:
                        theta = math.atan(nuav.coord_z / uav2user_dist)
                        probability_los = (1 + alpha * math.exp(-belta * ((180 / math.pi) * theta - alpha))) ** (-1)
                        probability_nlos = 1 - probability_los
                        uav_los = (probability_los * mulos)
                        uav_nlos = (probability_nlos * munlos)
                        '''
                        power_uav[count_idx] = trans_power * ((4 * math.pi * freq / c) ** (-2)) * (1 / uav2user_dist) * (
                                uav_los + uav_nlos) ** (-1)
                        power_list[nuav.id].append(power_uav[count_idx])
                        '''
                        power_uav = trans_power * ((4 * math.pi * freq / c) ** (-2)) * (1 / uav2user_dist) * (
                                uav_los + uav_nlos) ** (-1)
                        selected_uav_id = nuav.id
                        break
                #计算uav间距离

                power_bs_interference = [0]
#                if power_bs > max(power_uav):
                if power_bs > power_uav:
                    for l in range(len(bs_dist)):
                        if bs_dist[l] != min(bs_dist):
                            power_bs_interference.append((trans_power * gt * gr * (c / freq) ** 2) / 4 * math.pi * bs_dist[l]*10)
                        else:
                            power_bs_interference.append(0)
                    interference_bs = sum(power_bs_interference)
#                    SINR = power_bs/((N0 * bandwidth + interference_bs + sum(power_uav)))
                    SINR = power_bs/((N0 * bandwidth + interference_bs + power_uav))
                    rate_list.append((bandwidth * math.log2(1 + SINR)) * user_dens)
                else:
                    for l in  range(len(bs_dist)):
                        if bs_dist[l] < 7:
                            power_bs_interference.append(
                                (trans_power * gt * gr * (c / freq) ** 2) * self.bs_state[l] / 4 * math.pi * bs_dist[l]*10)
                        else:
                            power_bs_interference.append(0)
                    interference_bs = sum(power_bs_interference)
                    SINR = power_uav / ((N0 * bandwidth + interference_bs))
#                    SINR = max(power_uav) / ((N0 * bandwidth + interference_bs + sum(power_uav) - max(power_uav)))
                    rate_list.append((bandwidth * math.log2(1 + SINR)) * user_dens)
                    self.uavs[selected_uav_id].update_reward((bandwidth * math.log2(1 + SINR)) * user_dens)
                    power_list[selected_uav_id].append((bandwidth * math.log2(1 + SINR)) * user_dens)

#        self.count += 1
#        print('sum_rate%d: '%self.count, sum(rate_list)/self.nUAV)
#        uav_rate_list = []
#        for idx in range(self.nUAV):
#            uav_rate_list[idx].append(sum(power_list[idx]))
        if sum(rate_list) == 0:
            states = []
            for idx, a in enumerate(actions):
                #            print("states %d: " %(idx), self.uavs[idx].get_states())
                states.append([self.uavs[idx].coord_x,
                               self.uavs[idx].coord_y,
                               self.uavs[idx].coord_z])
            print(states)
        self.final_rate.append(sum(rate_list))
        return power_list,sum(self.final_rate) / 150*self.nUAV

if __name__=="__main__":
    nUAV = 10
    env = Environment(nUAV)
    env.reset()
    print(env.uavs)
    for i in range(1000):
        actions = np.random.randint(0,7,nUAV)
        act = env.step(actions)
#        print(sum([env.uavs[idx].get_reward() for idx in range(nUAV)]) / 10)
#        print(env.step(actions))
#        continue
'''
reward在约50步之后均保持不变？
'''