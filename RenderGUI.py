import numpy as np
import os

import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.properties import NumericProperty, BooleanProperty
from kivy.config import ConfigParser
from kivy.uix.settings import SettingsWithSpinner
from Tkinter import Tk
import tkFileDialog

from .. import savearray

kivy.require('1.1.2')

Tk().withdraw()


BUF_DIMENSIONS = (3840, 2160)  # supports up to 4k screens


class RenderGUI(Widget):
    rend = None
    azimuth = NumericProperty(20.0)
    altitude = NumericProperty(20.0)
    distance_per_pixel = NumericProperty(0.0)
    stepsize = NumericProperty(0.0)
    x_pixel_offset = NumericProperty(0)
    y_pixel_offset = NumericProperty(0)
    rend_opacity = BooleanProperty(False)
    channel = NumericProperty(0)
    log_offset = NumericProperty(6)

    helptext = ('Pan l/r: a/d\n'
                'Tilt u/d: w/s\n'
                'Zoom in/out: j/k\n'
                'Shift l/r: [left]/[right]\n'
                'Shift u/d: [up]/[down]\n'
                'Recenter shift: c\n'
                'Dynamic range inc/dec: i/u\n'
                'Stepsize inc/dec: p/o\n')

    def __init__(self, rend, **kwargs):
        self.texture = Texture.create(size=BUF_DIMENSIONS)
        self.texture_size = BUF_DIMENSIONS
        super(RenderGUI, self).__init__(**kwargs)

        self.rend = rend
        self.buffer_array = np.empty(BUF_DIMENSIONS[::-1], dtype='uint8')
        self.distance_per_pixel = self.rend.distance_per_pixel
        self.stepsize = self.rend.stepsize

        self.x_pixel_offset = rend.x_pixel_offset
        self.y_pixel_offset = rend.y_pixel_offset

        self._keyboard_open()
        Window.bind(on_resize=self._on_resize)

# Sketch putting in JSON stuff on the fly
        import json
        with open(os.path.dirname(os.path.abspath(__file__)) + '/settings.json') as fd:
            configdata = json.loads(fd.read())
            configdata[0]['options'] = self.rend.channellist()
        configdatastr = json.JSONEncoder().encode(configdata)

        config = ConfigParser()
        config.read(os.path.dirname(os.path.abspath(__file__)) + '/defaults.ini')
        self.s = SettingsWithSpinner(size=(300, 0))
        self.s.add_json_panel('Renderer Settings', config, data=configdatastr)
        self.s.bind(on_config_change=self._settings_change)
        self.s.bind(on_close=lambda x: setattr(self.s, 'x', -800))
        self.add_widget(self.s)

#initial update
        self._on_resize(Window, Window.size[0], Window.size[1])

    def _settings_change(self, settings, cparser, section, key, value):
        self._keyboard_open()
        if key == 'opacity':
            self.rend_opacity = (value == '1')
        elif key == 'channel':
            self.channel = self.rend.channellist().index(value)
        elif key in ('altitude', 'azimuth', 'distance_per_pixel', 'stepsize', 'log_offset'):
            setattr(self, key, float(value))
        else:
            return
        self.update()

    def _keyboard_open(self):
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

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
            self.distance_per_pixel *= 0.95
        elif keycode[1] == 'k':  # zoom out
            self.distance_per_pixel /= 0.95
        elif keycode[1] == 'u':  # decrease contrast, increasing dyn range
            self.log_offset += 0.4
        elif keycode[1] == 'i':  # increase contrast
            if self.log_offset > 1:
                self.log_offset -= 0.4
        elif keycode[1] == 'up':  # shift view up
            self.y_pixel_offset += 5
        elif keycode[1] == 'down':  # shift view down
            self.y_pixel_offset -= 5
        elif keycode[1] == 'left':  # shift view left
            self.x_pixel_offset -= 5
        elif keycode[1] == 'right':  # shift view right
            self.x_pixel_offset += 5
        elif keycode[1] == 'c':
            self.x_pixel_offset = self.y_pixel_offset = 0
        elif keycode[1] == 'o':  # decreases stepsize, increasing resolution
            self.stepsize *= 0.8
        elif keycode[1] == 'p':  # increases stepsize, decreasing resolution
            self.stepsize /= 0.8
        else:
            return

        self.update()

    def _on_resize(self, window, width, height):
        self.rend.projection_x_size, self.rend.projection_y_size = width, height
        self.s.size = (self.s.size[0], height)
        self.update()

    def update(self):
        if self.rend is None:
            return
        self.rend.distance_per_pixel = self.distance_per_pixel
        self.rend.stepsize = self.stepsize
        self.rend.y_pixel_offset = self.y_pixel_offset
        self.rend.x_pixel_offset = self.x_pixel_offset
        data = self.rend.i_render(self.channel, self.azimuth, -self.altitude,
                                  opacity=self.rend_opacity, verbose=False)
        data = np.log10(data[0] if self.rend_opacity else data)
        data = (data + self.log_offset) * 255 / (data.max() + self.log_offset)
        data = np.clip(data, 0, 255).astype('uint8')
        self.buffer_array[:data.shape[0], :data.shape[1]] = data

        buf = np.getbuffer(self.buffer_array)

        # then blit the buffer
        self.texture.blit_buffer(buf[:], colorfmt='luminance', bufferfmt='ubyte')
        self.canvas.ask_update()

    def save_image(self):
        output_name = tkFileDialog.asksaveasfilename(title='Image Array Filename')
        if output_name is None:
            return
        self.rend.distance_per_pixel = self.distance_per_pixel
        self.rend.stepsize = self.stepsize
        self.rend.y_pixel_offset = self.y_pixel_offset
        self.rend.x_pixel_offset = self.x_pixel_offset
        data = self.rend.i_render(self.channel, self.azimuth, -self.altitude,
                                  opacity=self.rend_opacity, verbose=False)
        savearray(output_name, data[0] if self.rend_opacity else data)

    def save_spectra(self):
        output_name = tkFileDialog.asksaveasfilename(title='Spectra Array Filename')
        if output_name is None:
            return
        self.rend.distance_per_pixel = self.distance_per_pixel
        self.rend.stepsize = self.stepsize
        self.rend.y_pixel_offset = self.y_pixel_offset
        self.rend.x_pixel_offset = self.x_pixel_offset
        data = self.rend.il_render(self.channel, self.azimuth, -self.altitude,
                                   opacity=self.rend_opacity, verbose=False)
        savearray(output_name, data[0])

        output_name = tkFileDialog.asksaveasfilename(title='Frequency Diff Array Filename')
        if output_name is None:
            return
        savearray(output_name, data[1])


class RenderApp(App):
    def __init__(self, rend):
        super(RenderApp, self).__init__()
        self.rend = rend

    def build(self):
        game = RenderGUI(self.rend)
        game.update()
        return game


def show_renderer(rend):
    app = RenderApp(rend)
    app.run()
