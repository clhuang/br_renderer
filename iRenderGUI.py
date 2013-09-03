import numpy as np
import os

import kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle

kivy.require('1.1.2')

class iRenderGui(Widget):
    azimuth = 20
    altitude = 20

    logOffset=18

    def __init__(self, rend, **kwargs):
        super(iRenderGui, self).__init__(**kwargs)
        Config.set('graphics', 'width', rend.projectionXsize)
        Config.set('graphics', 'height', rend.projectionYsize)

        from kivy.core.window import Window

        self.rend = rend
        self.xsize = rend.projectionXsize
        self.ysize = rend.projectionYsize
        self.stepsize = rend.stepsize
        self.distancePerPixel = rend.distancePerPixel
        self.rendersize = self.ysize * self.xsize
        self.texture = Texture.create(size=(self.xsize, self.ysize))
        self.texture.flip_vertical()


        self.xPixelOffset = rend.xPixelOffset;
        self.yPixelOffset = rend.yPixelOffset;

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        Window.bind(on_resize=self._on_resize)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'w':  #view up
            if self.altitude < 90:
                self.altitude += 2
            else:
                return
        elif keycode[1] == 's':  #view down
            if self.altitude > -90:
                self.altitude -= 2
            else:
                return
        elif keycode[1] == 'a':  #view left
            self.azimuth -= 2
        elif keycode[1] == 'd':  #view right
            self.azimuth += 2
        elif keycode[1] == 'j':  #zoom in
            self.rend.distancePerPixel *= 0.95
        elif keycode[1] == 'k':  #zoom out
            self.rend.distancePerPixel /= 0.95
        elif keycode[1] == 'u':  #decrease contrast
            self.logOffset += 1
        elif keycode[1] == 'i':  #increase contrast
            self.logOffset -= 1
        elif keycode[1] == 'up':  #shift view up
            self.rend.yPixelOffset += 5
        elif keycode[1] == 'down':  #shift view down
            self.rend.yPixelOffset -= 5
        elif keycode[1] == 'left':  #shift view left
            self.rend.xPixelOffset -= 5
        elif keycode[1] == 'right':  #shift view right
            self.rend.xPixelOffset += 5
        elif keycode[1] == 'o':  #decreases stepsize, increasing resolution
            self.rend.stepsize *= 0.8
        elif keycode[1] == 'p':  #increases stepsize, decreasing resolution
            self.rend.stepsize /= 0.8
        else:
            return

        self.update()

    def _on_resize(self, window, width, height):
        self.rend.projectionXsize, self.rend.projectionYsize = width, height
        self.xsize = width
        self.ysize = height
        self.texture = Texture.create(size=(width, height))
        self.texture.flip_vertical()
        self.update()

    def update(self):
        data = np.log(self.rend.iRender(self.channel, self.azimuth, self.altitude, False))
        data = (data + self.logOffset) * 255 / (data.max() + self.logOffset)
        data = np.clip(data, 0, 255).astype('uint8')

        buf = np.getbuffer(data)

        # then blit the buffer
        self.texture.blit_buffer(buf[:], colorfmt='luminance', bufferfmt='ubyte')
        with self.canvas:
            self.rect_bg = Rectangle(
                    texture=self.texture, pos=(0, 0),
                    size = (self.texture.size))
        self.canvas.ask_update()


class iRenderApp(App):
    def __init__(self, rend):
        super(iRenderApp, self).__init__()
        self.rend = rend
    def build(self):
        game = iRenderGui(self.rend)
        game.update()
        return game

def setRend(rend, channel):
    iRenderGui.channel = channel
    app = iRenderApp(rend)
    app.run()
