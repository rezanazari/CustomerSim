import gym
import gym_customersim
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == '__main__':

    # Create environment
    env = gym.make('ChurnSim-v2')

    # Instantiate the agent
    model = DQN('MlpPolicy', env, verbose=1, learning_starts= 1, gamma=0, train_freq=1, target_update_interval=1, learning_rate=0.001)

    # Train the agent
    model.learn(total_timesteps=int(1e4))

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10000)
    print("Mean rew: ", mean_reward)

    # Enjoy trained agent
    for i in range(100):
        obs = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        if obs[-1] > 0:
            print(obs, action)