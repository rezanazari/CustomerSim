from gym.envs.registration import register

register(
   id='CustomerSim-v0',
   entry_point='gym_customersim.envs:CustomerSimEnv',
   max_episode_steps=18,
   kwargs={'data_file':"kdd98_data/kdd1998tuples.csv",
                 'model_path':"gym_customersim/assets/"},
)