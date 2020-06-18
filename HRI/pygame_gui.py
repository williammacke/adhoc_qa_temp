import pygame
from pygame.locals import *

BLACK = (   0,   0,   0)
WHITE = ( 255, 255, 255)
GREEN = (   0, 255,   0)
RED   = ( 255,   0,   0)
BLUE  = (   0,   0, 255)

PI = 3.141592653

class PygameGUI:
    # Constructor
    def __init__(self):
        self.running = True
        self.screen = None
        self.clock = None
        self.size = self.width, self.height = 640, 400
        self.user_x = 40
        self.user_y = 40

        # Initiate screen
        if (self.on_init() == False):
          self.running = False

    # Initiate pygame gui
    def on_init(self):
        pygame.init()

        # Set screen to windowed size
        self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        pygame.display.set_caption("Pygame Tutorial Caption")
        self.running = True

    # Events (keyboard / mouse)
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                self.user_y -= 20
            elif event.key == pygame.K_s:
                self.user_y += 20
            elif event.key == pygame.K_a:
                self.user_x -= 20
            elif event.key == pygame.K_d:
                self.user_x += 20
        

    # Render drawing
    def on_render(self):
        if(self.running):
            self.screen.fill(WHITE)
            pygame.draw.rect(self.screen, RED, [self.user_x, self.user_y, 20, 20])
            pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()
    
    def on_execute(self):
        while( self.running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
            self.clock.tick(20)
        self.on_cleanup()

# if __name__ == "__main__":
#     gui = PygameGUI()
#     gui.on_execute()