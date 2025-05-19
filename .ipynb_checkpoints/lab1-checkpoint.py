#!/usr/bin/env python3
"""Lab 1. Import the core libraries used in this class"""

# imports
import pygame; print(pygame.__version__)
import random 
import time
import pyllusion; print(pyllusion.__version__)
from datetime import datetime
import pandas as pd; print(pd.__version__)
import matplotlib.pyplot as plt
import numpy as np; print(np.__version__)
from PIL import Image; print(Image.__version__)
import os
import sys
print("Congrats! You have successfully imported the core libraries used in this class.")

# now let's try opening a pygame window
# This code creates a simple pygame window that displays the text "Welcome" in the center of the screen.
def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Lab 1 Pygame test")
    
    # Set up font
    font = pygame.font.SysFont(None, 32)
    text = font.render("Congrats! Your illusion experiment will go here. Press any key to quit", True, (255, 255, 255))
    text_rect = text.get_rect(center=(400, 300))
    
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running = False
        
        screen.fill((0, 0, 0))
        screen.blit(text, text_rect)
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()



