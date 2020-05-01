import numpy as np
import tensorflow as tf

import ma_environment as gym

from model_agent_maddpg import MADDPG
from replay_buffer import ReplayBuffer


def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

    return target_init, target_update


agent1_ddpg = MADDPG('agent1')
agent1_ddpg_target = MADDPG('agent1_target')

agent2_ddpg = MADDPG('agent2')
agent2_ddpg_target = MADDPG('agent2_target')

agent3_ddpg = MADDPG('agent3')
agent3_ddpg_target = MADDPG('agent3_target')

saver = tf.train.Saver()

agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')
agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')
agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')

agent3_actor_target_init, agent3_actor_target_update = create_init_update('agent3_actor', 'agent3_target_actor')
agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')


def get_agents_action(o_n, sess, noise_rate=0):
#    print(o_n[1])
#    print(np.shape(np.random.rand(7)))
    agent1_action = agent1_ddpg.action(state=[o_n[0]], sess=sess) + np.random.randn(1) * noise_rate
    agent2_action = agent2_ddpg.action(state=[o_n[1]], sess=sess) + np.random.randn(1) * noise_rate
    agent3_action = agent3_ddpg.action(state=[o_n[2]], sess=sess) + np.random.randn(1) * noise_rate
    return agent1_action, agent2_action, agent3_action

def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(32)

    act_batch = total_act_batch[:, 0, :]
    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :]])

    obs_batch = total_obs_batch[:, 0, :]

    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other_actor1_o = total_next_obs_batch[:, 1, :]
    next_other_actor2_o = total_next_obs_batch[:, 2, :]
    # 获取下一个情况下另外两个agent的行动
    next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess), other_actors[1].action(next_other_actor2_o, sess)])
#    print('noa_shape: ',np.shape(next_other_action))
#    print('nob_shape: ',np.shape(next_obs_batch))
#    print('action_shape: ',np.shape(agent_ddpg.action(next_obs_batch,sess)))
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg.action(next_obs_batch, sess),
                                                                     other_action=next_other_action, sess=sess)
#    print('ob_shape: ',np.shape(obs_batch))
#    print('oab_shape: ', np.shape(other_act_batch))
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])


if __name__ == '__main__':
    env = gym.Environment(3)
#    print(env.nUAV)
    o_n = env.reset()

    agent_reward_v = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_reward_op = [tf.summary.scalar('agent' + str(i) + '_reward', agent_reward_v[i]) for i in range(3)]

    agent_a0 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_a0_op = [tf.summary.scalar('agent' + str(i) + '_action_0', agent_a0[i]) for i in range(3)]

#    agent_a1 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
#    agent_a1_op = [tf.summary.scalar('agent' + str(i) + '_action_1', agent_a1[i]) for i in range(3)]

#    agent_a2 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
#    agent_a2_op = [tf.summary.scalar('agent' + str(i) + '_action_2', agent_a2[i]) for i in range(3)]

#    agent_a3 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
#    agent_a3_op = [tf.summary.scalar('agent' + str(i) + '_action_3', agent_a3[i]) for i in range(3)]

#    agent_a4 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
#    agent_a4_op = [tf.summary.scalar('agent' + str(i) + '_action_4', agent_a4[i]) for i in range(3)]

#    agent_a5 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
#    agent_a5_op = [tf.summary.scalar('agent' + str(i) + '_action_5', agent_a5[i]) for i in range(3)]

#    agent_a6 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
#    agent_a6_op = [tf.summary.scalar('agent' + str(i) + '_action_6', agent_a6[i]) for i in range(3)]


    reward_100 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    reward_100_op = [tf.summary.scalar('agent' + str(i) + '_reward_l100_mean', reward_100[i]) for i in range(3)]

    reward_1000 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    reward_1000_op = [tf.summary.scalar('agent' + str(i) + '_reward_l1000_mean', reward_1000[i]) for i in range(3)]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init,
              agent2_actor_target_init, agent2_critic_target_init,
              agent3_actor_target_init, agent3_critic_target_init])

    summary_writer = tf.summary.FileWriter('./multi_ma_summary', graph=tf.get_default_graph())

    agent1_memory = ReplayBuffer(100000)
    agent2_memory = ReplayBuffer(100000)
    agent3_memory = ReplayBuffer(100000)

    # e = 1

    reward_100_list = [[], [], []]
    for i in range(1000000):
        if i % 1000 == 0:
            o_n = env.reset()
            for agent_index in range(3):
                summary_writer.add_summary(sess.run(reward_1000_op[agent_index],
                                                    {reward_1000[agent_index]: np.mean(reward_100_list[agent_index])}),
                                           i // 1000)

#        print('o_n:',o_n)
        agent1_action, agent2_action, agent3_action = get_agents_action(o_n, sess, noise_rate=0.2)
#        print('a:', agent1_action)
        #三个agent的行动

#        a = [[np.argmax(i)] for  i in [agent1_action,
#                                       agent2_action,
#                                       agent3_action]]
#        a = [[0, i[0][0], 0, i[0][1], 0] for i in [agent1_action, agent2_action, agent3_action]]
        a = [[i[0][0]] for i in [agent1_action,agent2_action,agent3_action]]
#        print('action: ',agent1_action)
#        a = [[i[0][0],i[0][1],
#              i[0][2],i[0][3],
#              i[0][4],i[0][5],
#              i[0][6]] for i in [agent1_action,agent2_action,agent3_action]]

#        print('a',a)
#        print('shape_a',np.shape(a))
#        print('a: ', a)
        o_n_next, r_n = env.step(a)
        print('episode: ',i,' states: ',o_n,' reward: ',r_n)
#        print(r_n)

        for agent_index in range(3):
#            print('r_n',r_n[agent_index])
            reward_100_list[agent_index].append(r_n[agent_index])
            reward_100_list[agent_index] = reward_100_list[agent_index][-1000:]

#        print('shape_on0: ', np.shape(o_n[0]))
#        print('shape_agent_act: ', np.shape(agent1_action[0]))
#        print('shape_reward0: ', np.shape(r_n[0]))
#        print('shape_o_n_next0: ',np.shape(o_n_next[0]))

        agent1_memory.add(np.vstack([o_n[0], o_n[1], o_n[2]]),
                          np.vstack([agent1_action[0], agent2_action[0], agent3_action[0]]),
                          r_n[0], np.vstack([o_n_next[0], o_n_next[1], o_n_next[2]]), False)
#        print('shape_memory: ', np.shape(agent1_memory))

        agent2_memory.add(np.vstack([o_n[1], o_n[2], o_n[0]]),
                          np.vstack([agent2_action[0], agent3_action[0], agent1_action[0]]),
                          r_n[1], np.vstack([o_n_next[1], o_n_next[2], o_n_next[0]]), False)

        agent3_memory.add(np.vstack([o_n[2], o_n[0], o_n[1]]),
                          np.vstack([agent3_action[0], agent1_action[0], agent2_action[0]]),
                          r_n[2], np.vstack([o_n_next[2], o_n_next[0], o_n_next[1]]), False)

        if i > 50000:
            # e *= 0.9999
            # agent1 train
#            print('shape: ',agent1_memory)
            train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,
                        agent1_critic_target_update, sess, [agent2_ddpg_target, agent3_ddpg_target])

            train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,
                        agent2_critic_target_update, sess, [agent3_ddpg_target, agent1_ddpg_target])

            train_agent(agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,
                        agent3_critic_target_update, sess, [agent1_ddpg_target, agent2_ddpg_target])
            if i % 10000 == 0:
                print('train')
#            print('train')
        for agent_index in range(3):
#            print(agent_index)
            summary_writer.add_summary(
                sess.run(agent_reward_op[agent_index], {agent_reward_v[agent_index]: r_n[agent_index]}), i)
            summary_writer.add_summary(sess.run(agent_a0_op[agent_index], {agent_a0[agent_index]: a[agent_index][0]}),
                                       i)
#            summary_writer.add_summary(sess.run(agent_a1_op[agent_index], {agent_a1[agent_index]: a[agent_index][1]}),
#                                       i)
#            summary_writer.add_summary(sess.run(agent_a2_op[agent_index], {agent_a2[agent_index]: a[agent_index][2]}),
#                                       i)
#            summary_writer.add_summary(sess.run(agent_a3_op[agent_index], {agent_a2[agent_index]: a[agent_index][3]}),
#                                       i)
#            summary_writer.add_summary(sess.run(agent_a4_op[agent_index], {agent_a2[agent_index]: a[agent_index][4]}),
#                                       i)
#            summary_writer.add_summary(sess.run(agent_a5_op[agent_index], {agent_a2[agent_index]: a[agent_index][5]}),
#                                       i)
#            summary_writer.add_summary(sess.run(agent_a6_op[agent_index], {agent_a2[agent_index]: a[agent_index][6]}),
#                                       i)
#            summary_writer.add_summary(
#                sess.run(reward_100_op[agent_index], {reward_100[agent_index]: np.mean(reward_100_list[agent_index])}),
#                i)

#        print('i:', i, 'reward: ', r_n, 'state: ', o_n)
        o_n = o_n_next
        if i % 1000 == 0:
            saver.save(sess, './multi_ma_weight/' + str(i) + '.cptk')

    sess.close()