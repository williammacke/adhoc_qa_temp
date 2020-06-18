import pygame
from pygame.locals import *

import enum
import numpy as np

BLACK = (   0,   0,   0)
WHITE = ( 255, 255, 255)
GREEN = (   0, 255,   0)
RED   = ( 255,   0,   0)
BLUE  = (   0,   0, 255)

PI = 3.141592653

class Input(enum.Enum):
  W = 2
  A = 1
  S = 3
  D = 0
  J = 5
  Exit = -1

class GUI:
    # Constructor
    #  width, height, num_cols, num_rows
    def __init__(self, num_cols, num_rows, stn_pos):
        self.running = True
        self.screen = None
        self.clock = None
        self.size = self.width, self.height = 640, 400
        self.user_x = 40
        self.user_y = 40
        self.prev_user_x = 40
        self.prev_user_y = 40

        # Initiate screen
        if (self.on_init(num_cols, num_rows, stn_pos) == False):
          self.running = False

    # Initiate pygame gui
    def on_init(self, num_cols, num_rows, stn_pos):
        pygame.init()

        # Set screen to windowed size
        self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        pygame.display.set_caption("Pygame Tutorial Caption")

        # Draw initial experiment screen
        self.screen.fill(WHITE)

        box_width = self.width / num_cols
        box_height = self.height / num_rows

        # Grid lines
        for x in range(1, num_cols + 1):
            point1 = pygame.math.Vector2(x * box_width, 0)
            point2 = pygame.math.Vector2(x * box_width, self.height)
            pygame.draw.line(self.screen, BLACK, point1, point2)
        
        for y in range(1, num_rows + 1):
            point1 = pygame.math.Vector2(0, y * box_height)
            point2 = pygame.math.Vector2(self.width, y * box_height)
            pygame.draw.line(self.screen, BLACK, point1, point2)
        
        # Stations
        x_margin = box_width / 10
        y_margin = box_width / 10
        font = pygame.font.SysFont(None, 54)
        num = 0
        
        for stn in stn_pos:
            rect = [box_width * stn[0] + x_margin, 
                box_height * stn[1] + y_margin, 
                box_width - (x_margin * 2), 
                box_height - (y_margin * 2)]
            
            pygame.draw.rect(
                self.screen, 
                RED, 
                rect
            )
            
            text = font.render(str(num), True, WHITE)
            self.screen.blit(text,
            (box_width * stn[0] + x_margin * 3, box_height * stn[1] + y_margin * 3))
            
            num += 1


        # pygame.draw.rect(self.screen, RED, [self.user_x, self.user_y, 20, 20])
        # stn_rect = [0 + x_margin, 0 + y_margin, box_width - (x_margin * 2), box_height - (y_margin * 2)]
        pygame.display.flip()
        self.running = True

    # Events (keyboard / mouse)
    # Returns what input was chosen
    def on_event(self, event):
        
        self.prev_user_x = self.user_x
        self.prev_user_y = self.user_y

        if event.type == pygame.QUIT:
            self.running = False
            return Input.Exit
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                self.user_y -= 20
                return Input.W
            elif event.key == pygame.K_s:
                self.user_y += 20
                return Input.S
            elif event.key == pygame.K_a:
                self.user_x -= 20
                return Input.A
            elif event.key == pygame.K_d:
                self.user_x += 20
                return Input.D
        return None
                
        
    # Render drawing
    def on_render(self):
        if(self.running):
            pygame.draw.rect(self.screen, RED, [self.user_x, self.user_y, 20, 20])
            pygame.draw.rect(self.screen, WHITE, [self.prev_user_x, self.prev_user_y, 20, 20])
            pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()
    
    def on_execute(self):
        action = None
        while self.running:
            for event in pygame.event.get():
                action = self.on_event(event) 
            if action:
                self.on_render()
                return action

# if __name__ == "__main__":
#     gui = PygameGUI()
#     gui.on_execute()