import copy
import os
import pickle

import pandas as pd
import pyreadstat
import numpy as np
import gym
from gym import spaces


class ChrunEnv(gym.Env):
    def __init__(self, seed=47,
                 data_file="/bigdisk/lax/renaza/env/gym-customersim/churn_data/looking_glass_v5.sas7bdat",
                 model_path="/bigdisk/lax/renaza/env/gym-customersim/gym_customersim/assets/"):
        super(ChrunEnv, self).__init__()

        self.rnd = np.random.RandomState(seed)
        self.xcols = ["count_of_suspensions_6m", "tot_drpd_pr1", "nbr_contacts", "calls_care_acct", "price_mention"]
        self.ycol = ["churn"]
        self.arpucol = ["avg_arpu_3m", "lifetime_value"]
        self.data, _ = pyreadstat.read_sas7bdat(data_file)
        self.data = self.data[self.xcols + self.ycol + self.arpucol].fillna(0)
        self.model = pickle.load(open(os.path.join(model_path, "churn_model.sav"), 'rb'))

        self.observation_space = spaces.Box(0, 100000, (5,))
        self.action_space = spaces.Discrete(4)

        pass

    def reset(self):
        customer_id = self.rnd.randint(0, len(self.data.index))
        self.s = self.data.iloc[[customer_id]][self.xcols].values.squeeze()
        self.arpu = self.data.iloc[[customer_id]]["avg_arpu_3m"].values.squeeze()
        self.lifetime = self.data.iloc[[customer_id]]["lifetime_value"].values.squeeze()
        return copy.copy(self.s)

    def step(self, a):
        """
        churn = self.model.predict_proba(self.s.reshape(1, -1))
        rew = (churn[0]) * self.arpu

        # a = 1: if price_metnion>0, then reduce the price 10 %. This will lead to 20% reduction in churn
        if a == 1 and self.s[-1] > 0:
            rnd = self.rnd.rand()
            if rnd < .2:
                rew = rew * .9 + 1 * self.arpu
            else:
                rew = 0
        """
        churn = self.model.predict(self.s.reshape(1, -1))
        if churn == 0:
            rew = self.arpu + 12 * self.arpu
            if a > 0:
                rew *= -1
        else:
            # a = 0: do nothing
            if a == 0:
                rew = 0
            # a = 1: if price_metnion>0, then reduce the price 10 %. This will lead to 20% reduction in churn
            elif a == 1:
                if self.s[-1] > 0:
                    rnd = self.rnd.rand()
                    if rnd < .2:
                        rew = (self.arpu + 12 * self.arpu) * .9
                    else:
                        rew = 0
                else:
                    rew = 0

            # a = 2: if price_metnion>0, then reduce the price 20 %. This will lead to 40% reduction in churn
            elif a == 2:
                if self.s[-1] > 0:
                    rnd = self.rnd.rand()
                    if rnd < .4:
                        rew = (self.arpu + 12 * self.arpu) * .8
                    else:
                        rew = 0
                else:
                    rew = 0


            # a = 3: if price_metnion>0, then reduce the price 30 %. This will lead to 50% reduction in churn
            elif a == 3:
                if self.s[-1] > 0:
                    rnd = self.rnd.rand()
                    if rnd < .5:
                        rew = (self.arpu + 12 * self.arpu) * .7
                    else:
                        rew = 0
                else:
                    rew = 0



            # a = 4: if count_of_suspensions_6m>0 and price_metnion==0, then one month free. This will lead to 50% reduction in churn.
            elif a == 4:
                if self.s[-1] > 0 and self.s[0] > 0:
                    rnd = self.rnd.rand()
                    if rnd < .5:
                        rew = 12 * self.arpu
                    else:
                        rew = 0
                else:
                    rew = 0

            # a = 5: if nbr_contacts>0 and price_metnion==0 and count_of_suspensions_6m ==0,
            #        then do a followup call. This will lead to 10% reduction in churn with cost of $10
            else:
                raise
        return copy.copy(self.s), rew, True, {}


if __name__ == '__main__':
    env = ChrunEnv()

    for i in range(100):
        s = env.reset()
        ns, r, done, _ = env.step(1)
        print(i, r)
