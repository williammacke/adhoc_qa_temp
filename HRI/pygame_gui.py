import pygame
from pygame.locals import *

import enum
import numpy as np
import ctypes

BLACK = (   0,   0,   0)
WHITE = ( 255, 255, 255)
GREEN = (   0, 255,   0)
RED   = ( 255,   0,   0)
BLUE  = (   0,   0, 255)

LIGHT_STEEL_BLUE = (167, 190, 211)
PRUSSIAN_BLUE = (13, 44, 84)
EMERALD = (111, 208, 140)
WINE = (115, 44, 44)
APRICOT = (255, 202, 175)
ORANGE_YELLOW = (245, 183, 0)

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
    def __init__(self, num_cols, num_rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos):
        self.running = True
        self.screen = None
        self.clock = None
        
        pygame.init()
        ctypes.windll.user32.SetProcessDPIAware()
        infoObject = pygame.display.Info()
        self.size = self.width, self.height = infoObject.current_w, infoObject.current_h # Fullscreen size
        # self.size = self.width, self.height= 500, 300
        self.num_rows = num_rows

        self.stn_pos = stn_pos
        self.tool_pos = tool_pos
        self.goal_stn = goal_stn

        self.user = [worker_pos[0], worker_pos[1]] 
        self.prev_user = [worker_pos[0], worker_pos[1]]
        self.arrived = False

        self.robot = [fetcher_pos[0], fetcher_pos[1]]
        self.prev_robot = [fetcher_pos[0], fetcher_pos[1]]
        self.robot_stay = False

        self.box_width = self.width / num_cols
        self.box_height = self.height / num_rows
        self.x_margin = self.box_width / 10
        self.y_margin = self.box_height / 10
        self.radius = self.box_width / 3
        self.font = pygame.font.SysFont(None, int(120 * self.height / 1080))

        # Initiate screen
        if (self.on_init(num_cols, num_rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos) == False):
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

    def render_all_stations(self):
        num = 0
        # Overlay worker stations (in case agent gets on top)
        for stn in self.stn_pos:
            if num == self.goal_stn:
                self.render_station(EMERALD, stn)
            else:
                self.render_station(WINE, stn)

            self.render_text(str(num), stn[0], stn[1])

            num += 1
        
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
    def on_init(self, num_cols, num_rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos):
        # Set screen to windowed size
        # self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        #Set screen to fullscreen
        self.screen = pygame.display.set_mode(self.size, pygame.FULLSCREEN)
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
        self.render_all_stations()
        
        # Toolbox Stations
        for tool in tool_pos:
            self.render_station(LIGHT_STEEL_BLUE, tool)
            self.render_text("T", tool[0], tool[1])

        # Agents
        # Worker
        self.render_agent(worker_pos[0], worker_pos[1], ORANGE_YELLOW)
        self.render_text("W",  worker_pos[0], worker_pos[1])
        
        # Fetcher
        self.render_agent(fetcher_pos[0], fetcher_pos[1], PRUSSIAN_BLUE)
        self.render_text("F",  fetcher_pos[0], fetcher_pos[1])

        pygame.display.flip()
        self.running = True

    # Events (keyboard / mouse)
    # Returns what input was chosen
    def on_event(self, event):
        
        self.prev_user[0] = self.user[0]
        self.prev_user[1] = self.user[1]
        if event.type == pygame.QUIT:
            self.running = False
            return Input.Exit
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                self.user[1] += 1
                return Input.W
            elif event.key == pygame.K_s:
                self.user[1] -= 1
                return Input.S
            elif event.key == pygame.K_a:
                self.user[0] -= 1
                return Input.A
            elif event.key == pygame.K_d:
                self.user[0] += 1
                return Input.D
            elif event.key == pygame.K_j:
                self.arrived = True
                return Input.J
            elif event.key == pygame.K_ESCAPE:
                self.running = False
                return Input.Exit

                
        # print(event.type)
        return None
                
        
    # Render drawing
    def on_render(self):
        if(self.running):
            #User
            if not self.arrived:
                self.render_agent(self.prev_user[0], self.prev_user[1], WHITE) # Remove old user agent
                self.render_all_stations()
                
                self.render_agent(self.user[0], self.user[1], ORANGE_YELLOW)
                self.render_text("W", self.user[0], self.user[1])

            #Robot
            if self.robot_stay:
                self.robot_stay = False
            else:
                self.render_agent(self.prev_robot[0], self.prev_robot[1], WHITE) # Remove old robot agent
                
                # Overlay fetcher stations (in case agent gets on top)
                for tool in self.tool_pos:
                    self.render_station(LIGHT_STEEL_BLUE, tool)
                    self.render_text("T", tool[0], tool[1])

                self.render_agent(self.robot[0], self.robot[1], PRUSSIAN_BLUE)
                self.render_text("F", self.robot[0], self.robot[1])


            pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()
    
    def _move_worker(self, move):
        self.prev_robot[0] = self.robot[0]
        self.prev_robot[1] = self.robot[1]

        if move == 0:
            self.robot[0] += 1
        elif move == 1:
            self.robot[0] -= 1
        elif move == 2:
            self.robot[1] += 1
        elif move == 3:
            self.robot[1] -= 1
        else: #query or pickup
            self.robot_stay = True

    def on_execute(self, other_agent_move):
        self._move_worker(other_agent_move)
        action = None
        while self.running:
            if not self.arrived:
                for event in pygame.event.get():
                    action = self.on_event(event)
            else:
                action =  Input.J
            if action:
                self.on_render()
                return action