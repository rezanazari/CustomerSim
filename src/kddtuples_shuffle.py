import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv("/bigdisk/lax/renaza/env/gym-customersim/kdd98_data/kdd1998tuples.csv",
                       header=None)
    # data.columns = ["customer", "period", "r0", "f0", "m0", "ir0", "if0", "gender", "age", "income",
    #                 "zip_region", "zip_la", "zip_lo", "a", "rew", "r1", "f1", "m1", "ir1", "if1", "gender", "age",
    #                 "income", "zip_region", "zip_la", "zip_lo"]
    data.columns = ["customer", "period", "_S0_", "_S1_", "_S2_", "_S3_", "_S4_", "_S5_", "_S6_", "_S7_",
                    "_S8_", "zip_la", "zip_lo", "a", "rew", "r1", "f1", "m1", "ir1", "if1", "gender", "age",
                    "income", "zip_region", "zip_la", "zip_lo"]

    data_sample = data.sample(frac=0.001)
    # data_sample.to_csv("/bigdisk/lax/renaza/env/gym-customersim/kdd98_data/kdd1998tuples_sample.csv", index=None)

    customer_idx = set(data['customer'])
    reward_lst = []
    for i in range(50):
        data_cust = data[data["customer"] == np.random.randint(len(customer_idx))]
        reward_lst.append(data_cust['rew'].sum())

    print("mean reward is : ", np.mean(reward_lst))