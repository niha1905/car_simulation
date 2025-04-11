import gym
from gym import spaces
import pygame
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.car_x, self.car_y = 100, 300
        self.car_angle = 0
        self.car_speed = 0
        return np.array([self.car_x, self.car_y, self.car_speed, self.car_angle], dtype=np.float32)

    def step(self, action):
        throttle, steering = np.clip(action, -1, 1)
        self.car_speed = np.clip(self.car_speed + throttle * 0.5, -2, 3)
        self.car_angle += steering * 5

        self.car_x += self.car_speed * np.cos(np.radians(self.car_angle))
        self.car_y += self.car_speed * np.sin(np.radians(self.car_angle))

        reward = self.car_speed - abs(steering) * 2

        done = self.car_x < 0 or self.car_x > self.width or self.car_y < 0 or self.car_y > self.height

        return np.array([self.car_x, self.car_y, self.car_speed, self.car_angle], dtype=np.float32), reward, done, {}

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.car_x), int(self.car_y)), 10)
        pygame.display.flip()

    def close(self):
        pygame.quit()

env = DummyVecEnv([lambda: CarEnv()])

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models/", name_prefix="ppo_car")

model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, batch_size=64, n_steps=512, tensorboard_log="./ppo_logs/")

model.learn(total_timesteps=10000, callback=checkpoint_callback)

model.save("ppo_car_model")
print("Training complete, model saved as ppo_car_model.zip")
env.close()
