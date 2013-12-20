import numpy as np

import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.properties import NumericProperty, BooleanProperty
from kivy.config import ConfigParser
from kivy.uix.settings import SettingOptions, SettingNumeric, SettingBoolean
from Tkinter import Tk
import tkFileDialog

kivy.require('1.8.0')

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
    snap = NumericProperty(0)

    helptext = ('Pan l/r: a/d\n'
                'Tilt u/d: w/s\n'
                'Zoom in/out: j/k\n'
                'Shift l/r: [left]/[right]\n'
                'Shift u/d: [up]/[down]\n'
                'Recenter shift: c\n'
                'Dynamic range inc/dec: i/u\n'
                'Stepsize inc/dec: ./,\n'
                'Toggle opacity: o\n'
                'Change timestep: [/]')
    initialized = False

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
        self.snap = self.rend.snap

        self.config = ConfigParser()
        self.config.setdefaults('renderer', {'channel': self.rend.channellist()[0],
                                             'snap': self.rend.snap,
                                             'opacity': 0,
                                             'altitude': self.altitude,
                                             'azimuth': self.azimuth,
                                             'distance_per_pixel': self.distance_per_pixel,
                                             'log_offset': self.log_offset,
                                             'stepsize': self.stepsize})
        self.remove_widget(self.spanel)
        self.spanel.config = self.config
        self.chan_opt = SettingOptions(title='Channel',
                                       desc='Emissions channel to select',
                                       key='channel',
                                       section='renderer',
                                       options=self.rend.channellist(),
                                       panel=self.spanel)
        self.spanel.add_widget(self.chan_opt)
        self.snap_opt = SettingNumeric(title='Snap',
                                       desc='Snap number to select',
                                       key='snap',
                                       section='renderer',
                                       panel=self.spanel)
        self.spanel.add_widget(self.snap_opt)
        self.opa_opt = SettingBoolean(title='Opacity',
                                      desc='Whether or not to enable opacity in the simulation',
                                      key='opacity',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.opa_opt)
        self.alt_opt = SettingNumeric(title='Altitude',
                                      desc='The POV angle above horizontal',
                                      key='altitude',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.alt_opt)
        self.azi_opt = SettingNumeric(title='Azimuth',
                                      desc='The POV angle lateral to the x-axis',
                                      key='azimuth',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.azi_opt)
        self.dpp_opt = SettingNumeric(title='Distance per Pixel',
                                      desc='Distance in simulation between pixels, specifies zoom',
                                      key='distance_per_pixel',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.dpp_opt)
        self.stp_opt = SettingNumeric(title='Step Size',
                                      desc='Magnitude of the integration stepsize, increase for performance',
                                      key='stepsize',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.stp_opt)
        self.range_opt = SettingNumeric(title='Dynamic Range',
                                        desc='Orders of magnitude to span in display',
                                        key='log_offset',
                                        section='renderer',
                                        panel=self.spanel)
        self.spanel.add_widget(self.range_opt)
        self.s.interface.add_panel(self.spanel, 'Renderer Settings', self.spanel.uid)

        self._keyboard_open()
        Window.bind(on_resize=self._on_resize)
#initial update
        self._on_resize(Window, Window.size[0], Window.size[1])
        self.initialized = True

    def _settings_change(self, section, key, value):
        self._keyboard_open()
        if key == 'opacity':
            self.rend_opacity = (value == '1')
        elif key == 'snap':
            self.snap = int(value)
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
            self.altitude += 2
        elif keycode[1] == 's':  # view down
            self.altitude -= 2
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
        elif keycode[1] == ',':  # decreases stepsize, increasing resolution
            self.stepsize *= 0.8
        elif keycode[1] == '.':  # increases stepsize, decreasing resolution
            self.stepsize /= 0.8
        elif keycode[1] == '[':  # go back 1 snap
            self.snap -= 1
        elif keycode[1] == ']':  # go forward 1 snap
            self.snap += 1
        elif keycode[1] == 'o':  # toggle opacity
            self.rend_opacity = not self.rend_opacity
        else:
            return

        self.update()

    def _on_resize(self, window, width, height):
        self.rend.projection_x_size, self.rend.projection_y_size = width, height
        self.s.size = (self.s.size[0], height)
        self.update()

    def update(self):
        if not self.initialized:
            return
#limit some values
        self.azimuth = self.azimuth % 360
        self.altitude = sorted((-90, self.altitude, 90))[1]
        self.snap = sorted(self.rend.snap_range + (self.snap,))[1]

#set values in renderer, and render
        self.rend.distance_per_pixel = self.distance_per_pixel
        self.rend.stepsize = self.stepsize
        self.rend.y_pixel_offset = self.y_pixel_offset
        self.rend.x_pixel_offset = self.x_pixel_offset
        self.rend.set_snap(self.snap)
        data = self.rend.i_render(self.channel, self.azimuth, -self.altitude,
                                  opacity=self.rend_opacity, verbose=False)
        data = np.log10(data[0] if self.rend_opacity else data)
        data = (data + self.log_offset) * 255 / (data.max() + self.log_offset)
        data = np.clip(data, 0, 255).astype('uint8')
        self.buffer_array[:data.shape[0], :data.shape[1]] = data

        buf = np.getbuffer(self.buffer_array)

        # then blit the buffer
        self.texture.blit_buffer(buf[:], colorfmt='luminance', bufferfmt='ubyte')

#update values in GUI
        self.azi_opt.value = str(self.azimuth)
        self.alt_opt.value = str(self.altitude)
        self.range_opt.value = str(self.log_offset)
        self.dpp_opt.value = str(round(self.distance_per_pixel, 6))
        self.stp_opt.value = str(round(self.stepsize, 6))
        self.opa_opt.value = '1' if self.rend_opacity else '0'
        self.snap_opt.value = str(self.rend.snap)
        self.canvas.ask_update()

    def save_image(self):
        output_name = tkFileDialog.asksaveasfilename(title='Image Array Filename')
        if output_name is None or output_name == '':
            return
        self.rend.distance_per_pixel = self.distance_per_pixel
        self.rend.stepsize = self.stepsize
        self.rend.y_pixel_offset = self.y_pixel_offset
        self.rend.x_pixel_offset = self.x_pixel_offset
        data = self.rend.i_render(self.channel, self.azimuth, -self.altitude,
                                  opacity=self.rend_opacity, verbose=False)
        self.rend.save_irender(output_name, data[0] if self.rend_opacity else data)

    def save_spectra(self):
        output_name = tkFileDialog.asksaveasfilename(title='Spectra Array Filename')
        if output_name is None or output_name == '':
            return
        self.rend.distance_per_pixel = self.distance_per_pixel
        self.rend.stepsize = self.stepsize
        self.rend.y_pixel_offset = self.y_pixel_offset
        self.rend.x_pixel_offset = self.x_pixel_offset
        data = self.rend.il_render(self.channel, self.azimuth, -self.altitude,
                                   opacity=self.rend_opacity, verbose=False)
        self.rend.save_ilrender(output_name, data)


class RenderApp(App):
    def __init__(self, rend):
        super(RenderApp, self).__init__(use_kivy_settings=False)
        self.rend = rend

    def build(self):
        game = RenderGUI(self.rend)
        game.update()
        return game


def show_renderer(rend):
    app = RenderApp(rend)
    app.run()
