import numpy as np

import kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle

kivy.require('1.1.2')


class RenderGUI(Widget):
    azimuth = 20
    altitude = 20

    log_offset = 18

    def __init__(self, rend, **kwargs):
        super(RenderGUI, self).__init__(**kwargs)
        Config.set('graphics', 'width', rend.projection_x_size)
        Config.set('graphics', 'height', rend.projection_y_size)

        from kivy.core.window import Window

        self.rend = rend
        self.xsize = rend.projection_x_size
        self.ysize = rend.projection_y_size
        self.stepsize = rend.stepsize
        self.distance_per_pixel = rend.distance_per_pixel
        self.rendersize = self.ysize * self.xsize
        self.texture = Texture.create(size=(self.xsize, self.ysize))
        self.texture.flip_vertical()

        self.x_pixel_offset = rend.x_pixel_offset
        self.y_pixel_offset = rend.y_pixel_offset

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        Window.bind(on_resize=self._on_resize)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'w':  # view up
            if self.altitude < 90:
                self.altitude += 2
            else:
                return
        elif keycode[1] == 's':  # view down
            if self.altitude > -90:
                self.altitude -= 2
            else:
                return
        elif keycode[1] == 'a':  # view left
            self.azimuth -= 2
        elif keycode[1] == 'd':  # view right
            self.azimuth += 2
        elif keycode[1] == 'j':  # zoom in
            self.rend.distance_per_pixel *= 0.95
        elif keycode[1] == 'k':  # zoom out
            self.rend.distance_per_pixel /= 0.95
        elif keycode[1] == 'u':  # decrease contrast
            self.log_offset += 1
        elif keycode[1] == 'i':  # increase contrast
            self.log_offset -= 1
        elif keycode[1] == 'up':  # shift view up
            self.rend.y_pixel_offset += 5
        elif keycode[1] == 'down':  # shift view down
            self.rend.y_pixel_offset -= 5
        elif keycode[1] == 'left':  # shift view left
            self.rend.x_pixel_offset -= 5
        elif keycode[1] == 'right':  # shift view right
            self.rend.x_pixel_offset += 5
        elif keycode[1] == 'o':  # decreases stepsize, increasing resolution
            self.rend.stepsize *= 0.8
        elif keycode[1] == 'p':  # increases stepsize, decreasing resolution
            self.rend.stepsize /= 0.8
        else:
            return

        self.update()

    def _on_resize(self, window, width, height):
        self.rend.projection_x_size, self.rend.projection_y_size = width, height
        self.xsize = width
        self.ysize = height
        self.texture = Texture.create(size=(width, height))
        self.texture.flip_vertical()
        self.update()

    def update(self):
        data = np.log(self.rend.i_render(self.channel, self.azimuth, self.altitude, False))
        data = (data + self.log_offset) * 255 / (data.max() + self.log_offset)
        data = np.clip(data, 0, 255).astype('uint8')

        buf = np.getbuffer(data)

        # then blit the buffer
        self.texture.blit_buffer(buf[:], colorfmt='luminance', bufferfmt='ubyte')
        with self.canvas:
            self.rect_bg = Rectangle(
                    texture=self.texture, pos=(0, 0),
                    size=(self.texture.size))
        self.canvas.ask_update()


class RenderApp(App):
    def __init__(self, rend):
        super(RenderApp, self).__init__()
        self.rend = rend

    def build(self):
        game = RenderGUI(self.rend)
        game.update()
        return game


def set_renderer(rend, channel):
    RenderGUI.channel = channel
    app = RenderApp(rend)
    app.run()

