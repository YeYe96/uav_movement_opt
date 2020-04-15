from maze_env import Maze
from RL_brain_DQN import DeepQNetwork
from numpy.linalg import cholesky

import seaborn as sns
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

n_episode = 3000
Q_value = []
rate_sum = []
pop_dens_list = np.random.randint(0,100,size=(26,26))
print(pop_dens_list)

def run_maze():
    step = 0
#    print('pop_dens: ',pop_dens_list)
    for episode in range(n_episode):
        # initial observation
        observation, rate_old = env.reset()

        while True:
            # fresh env
            env.render()
            #            print('render is ok')
            # RL choose action based on observation
            action = RL.choose_action(observation)
            '''
            print('action is ok')
            print('action: ',action)

            return 
            '''
            # RL take action and get next observation and reward
            #           print('action: ',action)
#            print('rate_old: ', rate_old)
            observation_, reward, battery, rate = env.step(action, rate_old, pop_dens_list)

            rate_old = rate

            RL.store_transition(observation, action, reward, observation_)

#            print('this episode is: ', episode, '    reward of this episode is: ', reward, ' time is: ',
#                 datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            if (step > 450) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if battery == 0:
                print('this episode is: ', episode, '    reward of this episode is: ', reward, ' rate is: ',rate,' time is: ',
                      datetime.datetime.now().strftime('%H:%M:%S'))
                Q_value.append(reward)
                rate_sum.append(rate)
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()



if __name__ == "__main__":
    # maze game
    env = Maze()
    data = pop_dens_list
    heatmap_plot = sns.heatmap(data,center=0,cmap='gist_ncar')
    plt.savefig('heatmap of pop dens.png')
    RL = DeepQNetwork(env.n_features_sum,env.n_actions_sum, env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      #                      epsilon_min=0.1,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    plt.show()
    x1 = []
    y1 = []
    for i in range(len(Q_value)):
        x1.append(i)
        y1.append(Q_value[i])
    plt.title('reward analysis')
    plt.plot(x1, y1,)
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()
    plt.savefig('reward_plt.png')

    x2 = []
    y2 = []
    for j in range(len(rate_sum)):
        x2.append(j)
        y2.append(rate_sum[j])
    plt.title('rate analysis')
    plt.plot(x2,y2)
    plt.ylabel('rate')
    plt.xlabel('episode')
    plt.show()
    plt.savefig('rate_plt.png')
    RL.plot_cost()