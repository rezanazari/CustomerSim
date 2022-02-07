import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tf warnings
import tensorflow as tf

import gym

from src.shared_functions import *
from src.net_designs import *

import pandas as ps
import numpy as np
import copy
from gym import spaces


class CustomerSimEnv(gym.Env):  # this class should inherit from gym.Env
    def __init__(self, seed=47, data_file="../../kdd98_data/kdd1998tuples.csv",
                 model_path="../assets/"):
        # create a random number generator for the env
        self.rnd = self.seed(seed)
        self.data_file = data_file
        self.model_path = model_path

        # load data
        self.state_cols = ['r0', 'f0', 'm0', 'ir0', 'if0', 'gender', 'age', 'income', 'zip_region']
        self.initial_states = self.load_data()

        # load the models for the regressor and classifier
        self.regressor, self.classifier = self.load_emulator_models()

        # create observation_space and action_space
        ## these two variables should be defined
        self.observation_space = spaces.Box(0, 100000, (9,))
        self.action_space = spaces.Discrete(12)

    def load_data(self):
        """
        This function loads the data which is in the form of [s, a, r, ns] and then returns the initial states
        """
        data = ps.read_csv(self.data_file, header=None)
        column_names = ['customer', 'period', 'r0', 'f0', 'm0', 'ir0', 'if0', 'gender', 'age', 'income',
                        'zip_region', 'zip_la', 'zip_lo', 'a', 'rew', 'r1', 'f1', 'm1', 'ir1', 'if1',
                        'gender1', 'age1', 'income1', 'zip_region1', 'zip_la1', 'zip_lo1']
        data.columns = column_names

        initial_states = data[data['period'] == 1][data['age'] != 0][self.state_cols]
        return initial_states

    def load_emulator_models(self):
        # load emulator models
        regressor = KDDRegressor()
        regressor.build(input_shape=[100, 10])
        regressor.load_weights(os.path.join(self.model_path, "kdd98_propagation_regressor_best.h5"))

        classifier = KDDClassifier()
        classifier.build(input_shape=[100, 10])
        classifier.load_weights(os.path.join(self.model_path, "kdd98_propagation_classifier_best.h5"))
        return regressor, classifier

    def reset(self):
        # returns the state of a random customer
        self.t = 0
        customer_id = self.rnd.randint(0, len(self.initial_states.index), 1)
        self.state = self.initial_states.iloc[customer_id[0]].to_numpy()
        # normalize s
        s = self.normalize(copy.copy(self.state))
        return s

    def get_donation(self, s, a):
        # regressor and classifier get the state and action as the input to predict the reward
        input = np.expand_dims(np.concatenate([s, np.array([a])], axis=0),
                               0)  # Dim = [1 x 10], first dimension is the batch

        # finding donation probability
        donation_prob = self.classifier.predict(input, verbose=0)[:, 1]
        threshold = 0.275

        # this formula adds stochasticity in occurance of donation, with highest volatility around the threshold value
        donations_occurred = (self.rnd.binomial(8, threshold, donation_prob.shape[0]) <
                              self.rnd.binomial(8, donation_prob))[0]  # number

        # donation amount
        donation = np.rint(self.regressor.predict(input, verbose=0).squeeze() * donations_occurred)

        return donations_occurred, donation

    def step(self, a):
        # get donation amount according to trained regressor and classifier
        donations_occurred, donation = self.get_donation(self.state, a)
        reward = donation

        # get next state
        # In customer journey, to get the next state, we use the current state, reward and action.
        ns = copy.copy(self.state)

        # Recency
        ns[0] = (self.state[0] + 1) * (donations_occurred == 0)

        # Frequency
        ns[1] = self.state[1] + donations_occurred

        # Avg. Past Donation
        ns[2] = (self.state[2] * self.state[1] + donation) / (
                ns[1] + 1 * (ns[1] == 0))

        # Avg. Interaction Recency
        ns[3] = (self.state[3] + 1) * (a == 0)  # Null action 0

        # Avg. Interaction Frequency
        ns[4] = self.state[4] + (a != 0)

        # we assume that the other variables 'gender', 'age', 'income', 'zip_region', don't change

        # normalize ns
        ns = self.normalize(ns)
        # update the state of env
        self.state = ns
        self.t += 1

        # check for termination condition which is 18 periods
        done = False
        if self.t >= 18:
            done = True

        return copy.copy(ns), reward, done, {}

    def seed(self, seed=None):
        self.rnd = np.random.RandomState(seed)
        return self.rnd

    def normalize(self,
                  s):
        return s


if __name__ == '__main__':
    env = CustomerSimEnv()
    env.seed(52)
    s = env.reset()
    done = False
    i = 0
    cum_rew = 0
    while not done:
        ns, r, done, _ = env.step(5)
        cum_rew += r
        print("iteration: {} -- rew: {}".format(i, r))
        i += 1
    print("The code ran successfully with cum rew of ", cum_rew)
