from gym.envs.registration import register

register(
   id='CustomerSim-v0',
   entry_point='gym_customersim.envs:CustomerSimEnv',
   max_episode_steps=18,
   kwargs={'data_file':"/bigdisk/lax/renaza/env/gym-customersim/kdd98_data/kdd1998tuples.csv",
                 'model_path':"/bigdisk/lax/renaza/env/gym-customersim/gym_customersim/assets/"},
)

register(
   id='ChurnSim-v0',
   entry_point='gym_customersim.envs:ChrunEnv',
   max_episode_steps=1,
   kwargs={'data_file':"/bigdisk/lax/renaza/env/gym-customersim/churn_data/looking_glass_v5.sas7bdat",
                 'model_path':"/bigdisk/lax/renaza/env/gym-customersim/gym_customersim/assets/"},
)

register(
   id='ChurnSim-v2',
   entry_point='gym_customersim.envs:ChrunEnvV2',
   max_episode_steps=1,
   kwargs={'data_file':"/bigdisk/lax/renaza/env/gym-customersim/churn_data/looking_glass_v5.sas7bdat",
                 'model_path':"/bigdisk/lax/renaza/env/gym-customersim/gym_customersim/assets/"},
)