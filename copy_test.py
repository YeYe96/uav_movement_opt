import ma_environment as gym
import numpy as np
import random
import time
import logging
from pprint import pprint

#logger = gym.logger.get_my_logger(__name__)


def timed_function(func):
    def decorated_func(*args, **kwargs):
        s = time.time()
        output = func(*args, **kwargs)
        e = time.time()
        logger.info("{} finished in {} seconds".format(func.__name__,e-s))
        return output
    return decorated_func

if __name__=="__main__":
    logging.disable(logging.NOTSET)
    nUAV = 10


    env = gym.make(nUAV)
    s = env.reset()
    while True:
        decision_agents  = s["decision agents"]
        states           = s["states"]
        rewards          = s["rewards"]


        # A bunch of printing stuff for debugging
        parsed_states = gym.Environment.parse_states(states[0])

        # Picking action here
        actions = [random.sample(env.legal_actions(agent), 1)[0] for agent in decision_agents]# 0 calls for the legal actions of the first elevator

        # Stone Age rendering ftw.
        s = timed_function(env.step)(actions)
        env.render()
        time.sleep(0.3)