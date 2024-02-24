import numpy as np
import pygame
import sys

class PongEnvironment:
    def __init__(self, width=400, height=300):
        self.width = width
        self.height = height
        self.ball_radius = 10
        self.paddle_width = 10
        self.paddle_height = 60
        self.paddle_offset = 20
        self.ball_pos = np.array([self.width // 2, self.height // 2], dtype=float)
        self.ball_vel = np.array([0.03, 0.01], dtype=float)  
        self.paddle_pos = self.height // 2
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pong")
        self.rally_length = 0
        self.hit_reward = 0.1  # Small positive reward for hitting the ball

    def reset(self):
        self.ball_pos = np.array([self.width // 2, self.height // 2], dtype=float)
        self.ball_vel = np.array([0.03, 0.01], dtype=float)
        self.paddle_pos = self.height // 2
        self.rally_length = 0

    def step(self, action):
        self.move_paddle(action)
        self.ball_pos += self.ball_vel

        if self.ball_pos[1] <= self.ball_radius or self.ball_pos[1] >= self.height - self.ball_radius:
            self.ball_vel[1] *= -1  

        if self.ball_pos[0] <= self.ball_radius:
            if self.paddle_pos - self.paddle_height / 2 <= self.ball_pos[1] <= self.paddle_pos + self.paddle_height / 2:
                self.ball_vel[0] *= -1  
                self.rally_length += 1  # Increment rally length on successful hit
                reward = self.hit_reward  # Small positive reward for hitting the ball
            else:
                reward = -1  # Negative reward for missing the ball
                return self.get_state(), reward, True  # End episode

        if self.ball_pos[0] >= self.width - self.ball_radius:
            self.ball_vel[0] *= -1  

        return self.get_state(), 0, False

    def move_paddle(self, action):
        self.paddle_pos = np.clip(self.paddle_pos + action, self.paddle_height / 2, self.height - self.paddle_height / 2)

    def get_state(self):
        return np.array([
            self.ball_pos[0] / self.width,       
            self.ball_pos[1] / self.height,      
            self.ball_vel[0],                    
            self.ball_vel[1],                    
            self.paddle_pos / self.height        
        ])

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, self.paddle_pos - self.paddle_height / 2, self.paddle_width, self.paddle_height))
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), self.ball_radius)
        pygame.display.flip()

    def get_rally_length(self):
        return self.rally_length


