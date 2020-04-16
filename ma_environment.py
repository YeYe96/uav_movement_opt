import numpy as np
import simpy
from simpy.events import AnyOf
import math
from uav import UAV


#NUM_UAV = 10
np.random.seed(1234)
pop_dens_list = np.random.randint(0,100,size=(26,26))

bs_state = np.array([0,1,1,0])
bs_coord = [[7, 7], [7, 19], [19, 7], [19, 19]]
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

def make(nUAV):
    simenv = simpy
    return Environment(simenv,nUAV)
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
    def __init__(self,simenv,nUAV):
        self.simenv = simenv
        self.nUAV = nUAV
        self.action_space = UAV.action_space


    def list_sqadd(self,a,b):
        c = []
        for i in range(len(a)):
            c.append((a[i]-b[i])**2)
        return c

    def step(self,actions):
        for idx,a in enumerate(actions):
            self.simenv.process(self.uavs[self.decision_uavs[idx]].act(a))

        while True:
            self.decision_uavs = []
            finished_event = self.simenv.run(until=AnyOf(self.simenv,self.epoch_events.values())).events
            self.update_all_reward()

            for event in finished_event:
                event_type = event.value
                #判断结束状态？
                '''
                if "ElevatorArrival" in event_type:
                    decision = self._process_elevator_arrival(event_type)
                elif event_type == "PassengerArrival":
                    decision = self._process_passenger_arrival()
                elif "LoadingFinished" in event_type:
                    decision = self._process_loading_finished(event_type)
                else:
                    raise ValueError("Unimplemented event type: {}".format(event_type))
                
            if decision:
                break
                '''

        output = {
            "states": self.get_states(),
            "rewards": self.get_rewards(),
            "decision agents": self.decision_uavs
        }
        return output

    def now(self):
        return self.simenv.now

    def get_states(self,uav_idxes,decision_epoch):
        return [self.uavs[idx].get_states(decision_epoch) for idx in uav_idxes]

    def get_rewards(self,uav_idxes,decision_epoch):
        return  [self.uavs[idx].get_reward(decision_epoch) for idx in uav_idxes]

    def reset(self):
        '''
        参数暂时不加
        '''
        self.simenv = simpy.Environment()
        #uavs部分暂时用这个代替，下面为原本代码
        #self.elevators = [Elevator(self, np.random.choice(np.arange(self.nFloor)), self.weightLimit, id) for id in range(self.nElevator)]
        self.uavs = [UAV(self,0,0,0,150,id) for id in range(self.nUAV)]
        self.final_rate = []
        return self.step([-1])

    def trigger_epoch_event(self,event_type):
        '''
        写完UAV部分内容后再编辑
        :param event_type:
        :return:
        '''

    def legal_actions(self,idx):
        return self.uavs[idx].legal_actions()

    def render(self):
        '''
        编辑完UAV部分后再编辑，大概率不需要
        :return:
        '''

    def update_all_reward(self):
        for u in self.uavs:
            u.update_reward(self.calculate_reward(u.last_decision_epoch))
        self.last_reward_time = self.simenv.now
        return True

    def calculate_reward(self,decision):
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
                    if bs_state[k] == 1:
                        if bs_state[k] == 1:
                            bs_dist.append(math.sqrt((height_bs**2) + np.sum(self.list_sqadd(bs_coord[k],user_coord))))
                            power_bs = (trans_power * gt * gr *(c / freq) **2 )/ 4 * math.pi * min(bs_dist)
                    else:
                        power_bs = 0
                distance.append(min(bs_dist))   #找到最近的MBS
                uav_dist = np.array([0 for num in range(self.nUAV)])

                power_uav = [0 for i in range(10)]
#                uav_dist = []
                count_idx = -1
                for nuav in range(self.uavs):
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
                                (trans_power * gt * gr * (c / freq) ** 2) * bs_state[l] / 4 * math.pi * bs_dist[l]*10)
                        else:
                            power_bs_interference.append(0)
                    interference_bs = sum(power_bs_interference)
                    SINR = max(power_uav) / ((N0 * bandwidth + interference_bs + sum(power_uav) - max(power_uav)))

                rate_list.append((bandwidth * math.log2(1 + SINR))*user_dens)
        self.final_rate.append(sum(rate_list))

        return sum(self.final_rate) / 1500
