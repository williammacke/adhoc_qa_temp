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
    def __init__(self, num_cols, num_rows, stn_pos, tool_pos, worker_pos, fetcher_pos):
        self.running = True
        self.screen = None
        self.clock = None
        self.size = self.width, self.height = 640, 400
        self.num_rows = num_rows

        self.user_x = worker_pos[0]
        self.user_y = worker_pos[1]
        self.prev_user_x = worker_pos[0]
        self.prev_user_y = worker_pos[1]

        self.robot_x = fetcher_pos[0]
        self.robot_y = fetcher_pos[1]
        self.prev_robot_x = fetcher_pos[0]
        self.prev_robot_y = fetcher_pos[1]
        self.robot_stay = False

        self.box_width = self.width / num_cols
        self.box_height = self.height / num_rows
        self.x_margin = self.box_width / 10
        self.y_margin = self.box_height / 10
        self.radius = self.box_width / 2 - 5

        self.font = ""

        # Initiate screen
        if (self.on_init(num_cols, num_rows, stn_pos, tool_pos, worker_pos, fetcher_pos) == False):
          self.running = False

    
    # Rectangular station
    def render_station(self, color, stn):
        rect = [self.box_width * stn[0] + self.x_margin, 
                self.box_height * (self.num_rows - 1 - stn[1]) + self.y_margin, 
                self.box_width - (self.x_margin * 2), 
                self.box_height - (self.y_margin * 2)]

        pygame.draw.rect(
            self.screen, 
            color, 
            rect
        )

    # Circular agent
    def render_agent(self, circle_x, circle_y, color):
        gui_x = circle_x * self.box_width + (self.box_width / 2)
        gui_y = (self.num_rows - 1 - circle_y) * self.box_height + (self.box_height / 2)
        pygame.draw.circle(self.screen, color, (int(gui_x), int(gui_y)), int(self.radius))

    # Text within station or agent
    def render_text(self, textString, box_x, box_y):
        text_x = box_x * self.box_width + self.x_margin * 3
        text_y = (self.num_rows - 1 - box_y) * self.box_height + self.y_margin * 3

        text = self.font.render(textString, True, WHITE)
        self.screen.blit(text,
            (text_x, text_y)
        )

    # Initiate pygame gui
    def on_init(self, num_cols, num_rows, stn_pos, tool_pos, worker_pos, fetcher_pos):
        pygame.init()
        self.font = pygame.font.SysFont(None, 54)

        # Set screen to windowed size
        self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        # Caption
        pygame.display.set_caption("Communication Ad-Hoc Teamwork")

        # Draw initial experiment screen
        self.screen.fill(WHITE)

        # Grid lines
        for x in range(1, num_cols + 1):
            point1 = pygame.math.Vector2(x * self.box_width, 0)
            point2 = pygame.math.Vector2(x * self.box_width, self.height)
            pygame.draw.line(self.screen, BLACK, point1, point2)
        
        for y in range(1, num_rows + 1):
            point1 = pygame.math.Vector2(0, y * self.box_height)
            point2 = pygame.math.Vector2(self.width, y * self.box_height)
            pygame.draw.line(self.screen, BLACK, point1, point2)
        
        # Stations
        num = 0
        
        # Worker Stations
        for stn in stn_pos:
            self.render_station(RED, stn)
            self.render_text(str(num), stn[0], stn[1])

            num += 1
        
        # Toolbox Stations
        for tool in tool_pos:
            self.render_station(BLUE, tool)
            self.render_text("T", tool[0], tool[1])

        # Agents
        # Worker
        self.render_agent(worker_pos[0], worker_pos[1], BLACK)
        self.render_text("W",  worker_pos[0], worker_pos[1])
        
        # Fetcher
        self.render_agent(fetcher_pos[0], fetcher_pos[1], GREEN)
        self.render_text("F",  fetcher_pos[0], fetcher_pos[1])

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
                self.user_y -= 1
                return Input.W
            elif event.key == pygame.K_s:
                self.user_y += 1
                return Input.S
            elif event.key == pygame.K_a:
                self.user_x -= 1
                return Input.A
            elif event.key == pygame.K_d:
                self.user_x += 1
                return Input.D
        return None
                
        
    # Render drawing
    def on_render(self):
        if(self.running):
            #User
            # self.render_agent(self.user_x, self.user_y, GREEN)
            # self.render_text("W", self.user_x, self.user_y)
            # self.render_agent(self.prev_user_x, self.prev_user_y, WHITE) # Remove old user agent

            # #Robot
            # if self.robot_stay:
            #     self.robot_stay = False
            # else:
            #     self.render_agent(self.robot_x, self.robot_y, GREEN)
            #     self.render_text("F", self.robot_x, self.robot_y)
            #     self.render_agent(self.prev_robot_x, self.prev_robot_y, WHITE) # Remove old robot agent

            # pygame.draw.rect(self.screen, RED, [self.user_x, self.user_y, 20, 20])
            # pygame.draw.rect(self.screen, WHITE, [self.prev_user_x, self.prev_user_y, 20, 20])
            pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()
    
    def _move_worker(self, move):
        self.prev_robot_x = self.robot_x
        self.prev_robot_y = self.robot_y

        if move == 0:
            self.robot_x += 1
        elif move == 1:
            self.robot_x -= 1
        elif move == 2:
            self.robot_y -= 1
        elif move == 3:
            self.robot_y += 1
        else: #query or pickup
            self.robot_stay = True
            

    def on_execute(self, other_agent_move):
        self._move_worker(other_agent_move)
        action = None
        while self.running:
            for event in pygame.event.get():
                action = self.on_event(event) 
            if action:
                self.on_render()
                return action