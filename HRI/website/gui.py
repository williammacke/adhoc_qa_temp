import numpy as np
import random
import copy

def never_query(obs, agent):
    return None

# Returns list of valid actions that brings fetcher closer to all tools
def get_valid_actions(obs, agent):
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    valid_actions = np.array([True] * 4) # NOOP is always valid
    print(valid_actions)
    for stn in range(len(s_pos)):
        if agent.probs[stn] == 0:
            continue
        tool_valid_actions = np.array([True] * 4)
        if f_pos[0] <= t_pos[stn][0]:
            tool_valid_actions[1] = False # Left
        if f_pos[0] >= t_pos[stn][0]:
            tool_valid_actions[0] = False # Right
        if f_pos[1] >= t_pos[stn][1]:
            tool_valid_actions[2] = False # Down
        if f_pos[1] <= t_pos[stn][1]:
            tool_valid_actions[3] = False # Up
        print(valid_actions)
        print(tool_valid_actions)
        valid_actions = np.logical_and(valid_actions, tool_valid_actions)
        print("hi3")
    print("hi2")
    return valid_actions

# Replaces np.max
def maxElement(array):
    maxElem = 0
    for elem in array:
        if elem > maxElem:
            maxElem = elem

    return maxElem

#Replaces np.array_equal
def checkEqual(a1, a2):
    if len(a1) != len(a2):
        return False
    
    for i in range(len(a1)):
        if a1[i] != a2[i]:
            return False
    return True

class FetcherQueryPolicy:
    """
    Basic Fetcher Policy for querying, follows query_policy function argument (defaults to never query)
    Assumes all tools are in same location
    """
    def __init__(self, query_policy=never_query, prior=None, epsilon=0):
        self.query_policy = query_policy
        self._prior = prior
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None
        self._epsilon = epsilon


    def reset(self):
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None


    def make_inference(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.prev_w_pos is None:
            return
        if w_action == 5:
            for i,stn in enumerate(s_pos):
                if not np.array_equal(stn, self.prev_w_pos):
                    self.probs[i] *= self._epsilon
        elif w_action == 0:
            for i,stn in enumerate(s_pos):
                if stn[0] <= self.prev_w_pos[0]:
                    self.probs[i] *= self._epsilon
        elif w_action == 1:
            for i,stn in enumerate(s_pos):
                if stn[0] >= self.prev_w_pos[0]:
                    self.probs[i] *= self._epsilon
        elif w_action == 3:
            for i,stn in enumerate(s_pos):
                if stn[1] >= self.prev_w_pos[1]:
                    self.probs[i] *= self._epsilon
        elif w_action == 2:
            for i,stn in enumerate(s_pos):
                if stn[1] <= self.prev_w_pos[1]:
                    self.probs[i] *= self._epsilon

        self.probs /= np.sum(self.probs)


    def action_to_goal(self, pos, goal):
        actions = []
        if pos[0] < goal[0]:
            actions.append(0)
        elif pos[0] > goal[0]:
            actions.append(1)
        if pos[1] > goal[1]:
            actions.append(3)
        elif pos[1] < goal[1]:
            actions.append(2)
        if len(actions) == 0:
            return 4
        return random.choice(actions)


    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
            else:
                for stn in self.query:
                    self.probs[stn] = 0
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        self.query = self.query_policy(obs, self)
        if self.query is not None:
            return 5, self.query

        if maxElement(self.probs) < (1 - self._epsilon):
            #dealing with only one tool position currently
            if checkEqual(f_pos, t_pos[0]):
                return 4, None
            else:
                return self.action_to_goal(f_pos, t_pos[0]), None
        else:
            if f_tool != np.argmax(self.probs):
                if np.array_equal(f_pos, t_pos[0]):
                    return 6, np.argmax(self.probs)
                else:
                    return self.action_to_goal(f_pos, t_pos[0]), None
            return self.action_to_goal(f_pos, s_pos[np.argmax(self.probs)]), None

class FetcherAltPolicy(FetcherQueryPolicy):
    """
    More Complicated Fetcher Policy, allows for multiple tool locations
    """
    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
            else:
                for stn in self.query:
                    self.probs[stn] = 0
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        # One station already guaranteed. No querying needed.
        if maxElement(self.probs) >= (1 - self._epsilon):
            target = np.argmax(self.probs)
            if f_tool != target:
                if checkEqual(f_pos, t_pos[target]):
                    return 6, target
                else:
                    return self.action_to_goal(f_pos, t_pos[target]), None
            return self.action_to_goal(f_pos, s_pos[target]), None

        self.query = self.query_policy(obs, self)

        if self.query is not None:
            return 5, self.query
       
        valid_actions = get_valid_actions(obs, self)

        if np.any(valid_actions):
            print(valid_actions)
            p = valid_actions / np.sum(valid_actions)
            action_idx = np.random.choice(np.arange(4), p=p)
            return action_idx, None
        else:
            return 4, None


import pygame
import time
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

NOOP_ALLOWED = True

class GUI:

    def __init__(self, num_cols, num_rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos):
        pygame.init()
        self.running = True
        self.clock = pygame.time.Clock()
        self.pause_screen = True

        # Dimensions and sizes
        self.size = self.width, self.height = 30*15, 30*15
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.box_width = self.width / num_cols
        self.box_height = self.height / num_rows
        self.x_margin = self.box_width / 10
        self.y_margin = self.box_height / 10
        self.radius = self.box_width / 2.5

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
        self.font = pygame.font.SysFont("lucidaconsole", int(120 * self.height / 1080))

        if (self.on_init() == False):
            self.running = False

    def on_init(self):
        # Set screen to windowed size
        # self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        # Set screen to fullscreen
        self.screen = pygame.display.set_mode(self.size)

        self.draw_pause_screen()
        # self.draw_experiment_screen()
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

    # Pause screen
    def draw_pause_screen(self):
        self.font = pygame.font.SysFont("lucidaconsole", int(50 * self.height / 1080))
        self.screen.fill(GRAY)

        text = self.font.render("Your goal station is number " + str(self.goal_stn + 1), True, WHITE)
        self.screen.blit(text, (self.width / 2 - 150, 25))
        text = self.font.render("Tab - Pause/Unpause", True, WHITE)
        self.screen.blit(text, (self.width / 2 - 100 , 75))
        text = self.font.render("Up - Move up", True, WHITE)
        self.screen.blit(text, (self.width / 2 - 70, 125))
        text = self.font.render("Left - Move left", True, WHITE)
        self.screen.blit(text, (self.width / 2 - 80, 175))
        text = self.font.render("Down - Move down", True, WHITE)
        self.screen.blit(text, (self.width / 2 - 100, 225))
        text = self.font.render("Right - Move right", True, WHITE)
        self.screen.blit(text, (self.width / 2 - 90, 275))
        text = self.font.render("Space - Done (press when arrived at station)", True, WHITE)
        self.screen.blit(text, (self.width / 2 - 200, 325))
        if NOOP_ALLOWED:
            text = self.font.render("Enter - Stop (don't move)", True, WHITE)
            self.screen.blit(text, (self.width / 2 - 150, 375))
            text = self.font.render("Press Tab to go to the experiment screen", True, BLACK)
            self.screen.blit(text, (self.width / 2 - 175, 425))
        else:
            text = self.font.render("Press Tab to go to the experiment screen", True, BLACK)
            self.screen.blit(text, (self.width / 2 - 175, 375))
        pygame.display.flip()

    def draw_experiment_screen(self):
        self.font = pygame.font.SysFont("lucidaconsole", int(self.height / self.num_cols - 15))
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
    
    # Render drawing
    def on_render(self):
        if self.running and not self.pause_screen :
            
            self.render_station(WHITE, self.prev_user) # Remove old user agent
            self.render_station(WHITE, self.prev_robot) # Remove old robot agent
            # self.render_agent(self.prev_user[0], self.prev_user[1], WHITE) # Remove old user agent
            # self.render_agent(self.prev_robot[0], self.prev_robot[1], WHITE) # Remove old robot agent
            self.render_all_stations()

            #User
            self.render_agent(self.user[0], self.user[1], ORANGE_YELLOW)
            self.render_text("W", self.user[0], self.user[1])

            #Robot
            if self.robot_stay:
                self.robot_stay = False
            self.render_agent(self.robot[0], self.robot[1], PRUSSIAN_BLUE)
            self.render_text("F", self.robot[0], self.robot[1])

            pygame.display.flip()

 
    # Close pygame when finished
    def on_cleanup(self):
        pygame.quit()
    
    # Move fetcher agent (robot)
    def _move_agent(self, move):
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

    # Pygame event (key down)
    def on_event(self, e):

        #Experiment screen
        if not self.pause_screen:
            self.prev_user[0] = self.user[0]
            self.prev_user[1] = self.user[1]
            print(pygame.key.name(e.key))
            # All directions
            if e.key == pygame.K_LEFT:
                self.user[0] -= 1
                return 1
            elif e.key == pygame.K_RIGHT:
                self.user[0] += 1
                return 0
            elif e.key == pygame.K_DOWN:
                self.user[1] -= 1
                return 3
            elif e.key == pygame.K_UP:
                self.user[1] += 1
                return 2
            # Work
            elif e.key == pygame.K_SPACE:
                return 5
            # NOOP 
            elif e.key == pygame.K_RETURN and NOOP_ALLOWED:
                return 4
        
        # Valid input for both pause screen and experiment screen
        if e.key == pygame.K_TAB: # Pause / unpause
            self.pause_screen = not self.pause_screen # Switch pause screen/experiment screen
            if self.pause_screen:
                self.draw_pause_screen()
            else:
                self.draw_experiment_screen()
        elif e.key == pygame.K_BACKSPACE: # End simulation
            return -1

    #Move fetcher and get user action
    def on_execute(self, other_agent_move):
        self._move_agent(other_agent_move)
        action = None
        while self.running:
            self.clock.tick(8)
            #User input
            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN:
                    action = self.on_event(e)
            if action != None:
                self.on_render()
                return action
        
        return action


if __name__ == '__main__':
    # Dimensions, stations, and worker/fetcher values
    cols = 10
    rows = 6
    stn_pos = [[3,0], [7,0], [3,5],[7,4]]
    goal_stn = 3
    tool_pos = [[9,4], [9,4], [9,4],[9,4]]
    worker_pos = [0,2]
    fetcher_pos = [9,2]

    # Set up pygame gui
    gui = GUI(cols, rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos)

    # Set up fetcher robot
    fetcher = FetcherQueryPolicy()
    # fetcher = FetcherAltPolicy(epsilon=0.05)
    # Observation state 
    f_obs = [worker_pos, fetcher_pos, stn_pos, tool_pos, None, None, None, None]
    print(f_obs)
    done = False

    #Loop actions until expreiment is complete
    while not done:
        gui.clock.tick(8)
        #Get fetcher move
        fetcher_move = fetcher(f_obs)
        print(fetcher_move[0])
        #Get user action
        action = gui.on_execute(fetcher_move[0])
        print(action)
        if action == -1:
            done = True

        f_obs[5] = action
        f_obs[6] = fetcher_move[0]
        print(f_obs)

    gui.clock.tick(8)
    gui.screen.fill(pygame.Color("white"))
    pygame.display.update()
    gui.on_cleanup()