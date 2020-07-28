import pygame
from random import randint

BLACK = (   0,   0,   0)
GRAY  = ( 192, 192, 192)
WHITE = ( 255, 255, 255)
GREEN = (   0, 255,   0)
RED   = ( 255,   0,   0)
BLUE  = (   0,   0, 255)

LIGHT_STEEL_BLUE = ( 167, 190, 211)
PRUSSIAN_BLUE    = (  13,  44,  84)
EMERALD          = ( 111, 208, 140)
WINE             = ( 115,  44,  44)
APRICOT          = ( 255, 202, 175)
ORANGE_YELLOW    = ( 245, 183,   0)

class GUI:
    # Constructor    
    # def __init__(self, num_cols, num_rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos):
        # self.running = True
        # self.screen = None
        # self.clock = None
        # self.pause_screen = True
        
        # pygame.init()
        # ctypes.windll.user32.SetProcessDPIAware()
        # infoObject = pygame.display.Info()
        # self.size = self.width, self.height = infoObject.current_w, infoObject.current_h # Fullscreen size
        # # self.size = self.width, self.height= 500, 300
        # self.num_rows = num_rows
        # self.num_cols = num_cols

        # self.stn_pos = stn_pos
        # self.tool_pos = tool_pos
        # self.goal_stn = goal_stn

        # self.user = [worker_pos[0], worker_pos[1]] 
        # self.prev_user = [worker_pos[0], worker_pos[1]]
        # self.arrived = False

        # self.robot = [fetcher_pos[0], fetcher_pos[1]]
        # self.prev_robot = [fetcher_pos[0], fetcher_pos[1]]
        # self.robot_stay = False

        # self.box_width = self.width / num_cols
        # self.box_height = self.height / num_rows
        # self.x_margin = self.box_width / 10
        # self.y_margin = self.box_height / 10
        # self.radius = self.box_width / 3
        # self.font = pygame.font.SysFont(None, int(120 * self.height / 1080))

        # # Initiate screen
        # if (self.on_init() == False):
        #     self.running = False

    def __init__(self, num_cols, num_rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos):
        pygame.init()

        self.size = self.width, self.height = 30*17, 30*15

        # Dimension
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.box_width = self.width / num_cols
        self.box_height = self.height / num_rows
        self.x_margin = self.box_width / 10
        self.y_margin = self.box_height / 10
        self.radius = self.box_width / 3

        # Stations
        self.stn_pos = stn_pos
        self.tool_pos = tool_pos
        self.goal_stn = goal_stn

        # Worker
        self.user = [worker_pos[0], worker_pos[1]] 
        self.prev_user = [worker_pos[0], worker_pos[1]]
        self.arrived = False

        # Fetcher
        self.robot = [fetcher_pos[0], fetcher_pos[1]]
        self.prev_robot = [fetcher_pos[0], fetcher_pos[1]]
        self.robot_stay = False

        # Font
        self.font = pygame.font.SysFont(None, int(120 * self.height / 1080))

        if (self.on_init() == False):
            self.running = False

    def on_init(self):
        # Set screen to windowed size
        # self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        # Set screen to fullscreen
        self.screen = pygame.display.set_mode(self.size)

        # self.draw_pause_screen()
        self.draw_experiment_screen()
        self.running = True
    
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
        # Worker stations
        num = 0
        for stn in self.stn_pos:
            if num == self.goal_stn:
                self.render_station(EMERALD, stn)
            else:
                self.render_station(WINE, stn)

            self.render_text(str(num + 1), stn[0], stn[1])

            num += 1
        
        # Toolbox Stations
        for tool in self.tool_pos:
            self.render_station(LIGHT_STEEL_BLUE, tool)
            self.render_text("T", tool[0], tool[1])

    # Circular agent
    def render_agent(self, circle_x, circle_y, color):
        gui_x = circle_x * self.box_width + (self.box_width / 2)
        gui_y = (self.num_rows - 1 - circle_y) * self.box_height + (self.box_height / 2)
        pygame.draw.circle(self.screen, color, (int(gui_x), int(gui_y)), int(self.radius))

    # Text within station or agent
    def render_text(self, textString, box_x, box_y):
        text_x = box_x * self.box_width + self.x_margin * 2
        text_y = (self.num_rows - 1 - box_y) * self.box_height + self.y_margin * 2

        text = self.font.render(textString, True, WHITE)
        self.screen.blit(text,
            (text_x, text_y)
        )

    def draw_experiment_screen(self):
        self.font = pygame.font.SysFont(None, int(self.height / self.num_cols))
        self.screen.fill(WHITE)
    
        # Grid lines
        for x in range(1, self.num_cols):
            point1 = (x * self.box_width, 0)
            point2 = (x * self.box_width, self.height)
            pygame.draw.line(self.screen, BLACK, point1, point2)
        
        for y in range(1, self.num_rows):
            point1 = (0, y * self.box_height)
            point2 = (self.width, y * self.box_height)
            pygame.draw.line(self.screen, BLACK, point1, point2)
        
        # # Stations
        self.render_all_stations()
        
        # Agents
        # Worker
        self.render_agent(self.prev_user[0], self.prev_user[1], ORANGE_YELLOW)
        self.render_text("W", self.prev_user[0], self.prev_user[1])
        
        # Fetcher
        self.render_agent(self.prev_robot[0], self.prev_robot[1], PRUSSIAN_BLUE)
        self.render_text("F", self.prev_robot[0], self.prev_robot[1])
        pygame.display.flip()

 
if __name__ == '__main__':
    gui = GUI(10, 6, [[1, 2], [2, 2], [3, 3]], 1, [[4,4],[4,4],[4,4]], [5,5], [6,6])
    # gui = GUI(cols, rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos)
