"""
SDL2 is a library responsible of visualisation, it's develop on C/C++
and can be use in python to visualize the poses and the map of our SLAM application
"""

import cv2
import sdl2
import sdl2.ext

class Display(object):
    def __init__(self, W, H):
        sdl2.ext.init()
        self.window = sdl2.ext.Window("V_SLAM", size=(W, H))
        self.window.show()
        self.W, self.H = W, H

    def paint(self, img):
        img = cv2.resize(img, (self.W, self.H))

        # retrives a list of SDL2 events
        events = sdl2.ext.get_events()

        for event in events:
            # checking the type of the event (eg. SDL_QUIT close window)
            if event.type == sdl2.SDL_QUIT:
                exit(0)
            
            # Retrieves a 3D numpy array representing the pixel data of the winndow's surface

            surf = sdl2.ext.pixels3d(self.window.get_surface())

            # update the pixel data of the window's surface with the resized image
            # img.swapaxes(0,1) swaps the axes of the image array to match the expected format of the SDL surface
            surf[:,:,0:3] = img.swapaxes(0,1)

            #refresh the window to display the updated surface
            self.window.refresh()
        