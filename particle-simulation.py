# Particle Simulation
# Chandler Supple, 5/28/2019

# Libraries
import numpy as np
from numba import jit
import pygame
import time
import matplotlib.pyplot as plt
%matplotlib inline

# Parameters
print("\nCount: The number of particles, Food: The number of food items, Spread: The spread of particles at initialization, Saturation: The number of particles that can exist in a space\nExample: '100, 20, 100, 3'")
parameters = input("Count, Food, Spread, Saturation: ")
count, food_count, spread, saturation = [int(element) for element in parameters.split(',')]

# Initialization
flagged = False
population_points = []
position = np.random.randint(low= 360 - spread, high= 360 + spread + 1, size= [count, 2])
food = np.random.randint(low= 0, high= 720 + 1, size= [food_count, 2])

# Setup PyGame
pygame.init()
master = pygame.display.set_mode((720, 720))
pygame.display.set_caption("Particle Simulation")
clock = pygame.time.Clock()

white = (255, 255, 255)
black = (0, 0, 0)
grey = (208, 216, 219)

# Particle Modifiers
@jit
def update_position(position, count):
    return position + np.random.randint(low= -7.5, high= 7.5 + 1, size= [count, 2])

@jit
def update_population(position, count, food):
        position_pop = []
        F_surrounding = 1
        for iterator, particle in enumerate(position):
            position_occluded = np.delete(position, iterator, 0)
            P_closeness = np.sqrt((position_occluded[:, 0] - particle[0])**2 + (position_occluded[:, 1] - particle[1])**2)
            P_surrounding = len(P_closeness[np.where(P_closeness < 2.0)])
            if food_count > 0:
                F_closeness = np.sqrt((food[:, 0] - particle[0])**2 + (food[:, 1] - particle[1])**2)
                F_surrounding = len(F_closeness[np.where(F_closeness < 4.0)])
            if P_surrounding >= saturation:
                count = count - 1
                position_pop.append(iterator)
            if P_surrounding < saturation and F_surrounding >= 1:
                count = count + 1
                position = np.append(position, position[iterator] + np.random.randint(low= -7.5, high= 7.5 + 1, size= [1, 2]), axis= 0)
        position = np.delete(position, position_pop, 0)
        return position, count

timer = time.time()
    
# Environment Loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    if count > 0 and count < 12500 and time.time() - timer <= 45: # Limits the length of the simulation
        for particle in position:
            master.set_at((particle[0], particle[1]), black)

        for morsel in food:
            pygame.draw.circle(master, grey, morsel, 4)
            
        position = update_position(position, count)
        position, count = update_population(position, count, food)
        
        population_points.append(count)
        
        pygame.display.flip()
        master.fill(white)
    
    else:
        if flagged == False:
            plt.plot(population_points)
            plt.xlabel("Step")
            plt.ylabel("Population")
            plt.show()
        flagged = True
        
    clock.tick(60)
