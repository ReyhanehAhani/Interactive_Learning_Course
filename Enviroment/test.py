from env import PongEnvironment
import pygame
import sys
env = PongEnvironment()
state = env.reset()
done = False

action = 0.1  
counter = 0
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    counter += 1
    if counter % 1000 == 0:  
        action *= -1

    next_state, reward, done = env.step(action)
    env.render()
    env.clock.tick(10000)  