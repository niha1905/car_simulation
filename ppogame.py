import gym
import pygame
import numpy as np
import random
import math
from stable_baselines3 import PPO

# Constants
TURN_ANGLE = 5
MIN_SPEED = -1.5
MAX_SPEED = 2.5
DEFAULT_SPEED = 1.0
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
ROAD_COLOR = (80, 80, 80)
ROAD_BORDER_COLOR = (200, 200, 200)

class VehicleState:
    def __init__(self, position, angle, speed):
        self.position = position
        self.angle = angle
        self.speed = speed
        self.stuck_counter = 0
        self.last_known_safe_position = position
        self.initial_angle = angle
    
    def update_position(self, action):
        self.angle += action[0] * TURN_ANGLE
        self.speed = min(max(self.speed + action[1], MIN_SPEED), MAX_SPEED)
        dx = self.speed * math.cos(math.radians(self.angle))
        dy = self.speed * math.sin(math.radians(self.angle))
        self.position = (self.position[0] + dx, self.position[1] + dy)

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("PPO Car Simulation")
        self.car_img = pygame.image.load("R:/Projects/ppo_car_simulation/car.png")
        self.car_img = pygame.transform.scale(self.car_img, (40, 20))
        self.action_space = gym.spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                                           high=np.array([1, 1], dtype=np.float32), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.full((4,), -np.inf, dtype=np.float32),
                                                high=np.full((4,), np.inf, dtype=np.float32), dtype=np.float32)
        self.road_x_min, self.road_x_max = 150, 650
        self.road_y_min, self.road_y_max = 50, 550
        self.obstacles = [(random.randint(200, 600), random.randint(150, 450)) for _ in range(5)]
        self.lap_checkpoint = (400, 100)
        self.reset()

    def reset(self):
        self.vehicle = VehicleState((400, 300), random.randint(0, 360), 0.0)
        return np.array([*self.vehicle.position, self.vehicle.speed, self.vehicle.angle], dtype=np.float32)

    def check_collision(self):
        return any(abs(self.vehicle.position[0] - obs[0]) < 20 and abs(self.vehicle.position[1] - obs[1]) < 20 for obs in self.obstacles)
    
    def is_off_track(self):
        return not (self.road_x_min <= self.vehicle.position[0] <= self.road_x_max and 
                    self.road_y_min <= self.vehicle.position[1] <= self.road_y_max)
    
    def step(self, action):
        self.vehicle.update_position(action)
        reward = self.vehicle.speed - abs(action[0]) * 2
        done = False

        if self.check_collision():
            reward -= 100
            done = True
        elif self.is_off_track():
            reward -= 50
            self.vehicle.stuck_counter += 1
            if self.vehicle.stuck_counter > 5:
                self.vehicle.angle += 90
                self.vehicle.speed = DEFAULT_SPEED
                self.vehicle.stuck_counter = 0
        else:
            self.vehicle.stuck_counter = 0

        if abs(self.vehicle.position[0] - self.lap_checkpoint[0]) < 20 and abs(self.vehicle.position[1] - self.lap_checkpoint[1]) < 20:
            reward += 100
            done = True

        self.vehicle.speed *= 0.98
        return np.array([*self.vehicle.position, self.vehicle.speed, self.vehicle.angle], dtype=np.float32), reward, done, {}

    def render(self):
        self.screen.fill(WHITE)
        pygame.draw.rect(self.screen, ROAD_BORDER_COLOR, (self.road_x_min-10, self.road_y_min-10, 520, 520))
        pygame.draw.rect(self.screen, ROAD_COLOR, (self.road_x_min, self.road_y_min, 500, 500))
        pygame.draw.circle(self.screen, GREEN, self.lap_checkpoint, 10)
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, RED, (obs[0], obs[1], 20, 20))
        rotated_car = pygame.transform.rotate(self.car_img, -self.vehicle.angle)
        car_rect = rotated_car.get_rect(center=self.vehicle.position)
        self.screen.blit(rotated_car, car_rect)
        pygame.display.flip()

    def close(self):
        pygame.quit()

# Initialize environment and load trained model
env = CarEnv()
model = PPO.load("R:/Projects/ppo_car_simulation/ppo_car_model.zip")
manual_control = False
obs = env.reset()
done = False

while not done:
    env.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                manual_control = not manual_control
            if event.key == pygame.K_q:
                done = True
    if manual_control:
        keys = pygame.key.get_pressed()
        throttle, steering = 0, 0
        if keys[pygame.K_UP]:
            throttle = 1.0
        if keys[pygame.K_DOWN]:
            throttle = -1.0
        if keys[pygame.K_LEFT]:
            steering = -1.0
        if keys[pygame.K_RIGHT]:
            steering = 1.0
        action = np.array([steering, throttle], dtype=np.float32)
    else:
        action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
env.close()
