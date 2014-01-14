from __future__ import print_function
import numpy as np
from string import Template
from threading import Thread

import kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.properties import NumericProperty, BooleanProperty
from kivy.config import ConfigParser
from kivy.uix.settings import SettingsPanel, SettingOptions, SettingNumeric, SettingBoolean
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from Tkinter import Tk
import tkFileDialog

kivy.require('1.8.0')

Tk().withdraw()


BUF_DIMENSIONS = (3840, 2160)  # supports up to 4k screens


class RenderGUI(Widget):
    rend = None
    display_data = None
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
                'zoom in/out: j/k\n'
                'Shift l/r: [left]/[right]\n'
                'Shift u/d: [up]/[down]\n'
                'Recenter shift: c\n'
                'Dynamic range inc/dec: i/u\n'
                'Stepsize inc/dec: ./,\n'
                'Toggle opacity: o\n'
                'Change timestep: [/]')
    initialized = False
    renderrange_progress = (0, 0)

    def __init__(self, renderer, **kwargs):
        import os.path
        self.texture = Texture.create(size=BUF_DIMENSIONS)
        self.texture_size = BUF_DIMENSIONS
        super(RenderGUI, self).__init__(**kwargs)

        self.rend = renderer
        self.buffer_array = np.empty(BUF_DIMENSIONS[::-1], dtype='uint8')
        self.distance_per_pixel = self.rend.distance_per_pixel
        self.stepsize = self.rend.stepsize

        self.x_pixel_offset = 0
        self.y_pixel_offset = 0
        self.snap = self.rend.snap

        self.config = ConfigParser()
        self.channellist = [os.path.basename(os.path.splitext(a)[0]) for a in self.rend.channellist()]
        self.config.setdefaults('renderer', {'channel': self.channellist[0],
                                             'snap': self.snap,
                                             'opacity': 0,
                                             'altitude': self.altitude,
                                             'azimuth': self.azimuth,
                                             'distance_per_pixel': self.distance_per_pixel,
                                             'log_offset': self.log_offset,
                                             'stepsize': self.stepsize})
        self.spanel = SettingsPanel(settings=self.s, title='Render Settings', config=self.config)
        self.chan_opt = SettingOptions(title='Channel',
                                       desc='Emissions channel to select',
                                       key='channel',
                                       section='renderer',
                                       options=self.channellist,
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
        self.saverangedialog = SaveRangeDialog(self, size_hint=(.8, .8), title="Save Range")
        self.update_display_enabled = True
        Clock.schedule_interval(lambda dt: self.update_display(), 0.1)

        self.initialized = True

    def _rend_setattr(self, key, value):
        self.rendcontroller_conn.send

    def _settings_change(self, section, key, value):
        self._keyboard_open()
        if key == 'opacity':
            self.rend_opacity = (value == '1')
        elif key == 'snap':
            self.snap = int(value)
        elif key == 'channel':
            self.channel = self.channellist.index(value)
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
        if not self.rend.condition.acquire(False):
            return
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

        self.update(False, False)

    def _on_resize(self, window, width, height):
        self.rend.projection_x_size, self.rend.projection_y_size = width, height
        self.s.size = (self.s.size[0], height)
        self.update()

    def update(self, block=True, lock=True):
        if not self.initialized:
            return
        if lock and not self.rend.condition.acquire(block):
            return
        self.rend.output_for_display = True
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

#update values in GUI
        self.azi_opt.value = str(self.azimuth)
        self.alt_opt.value = str(self.altitude)
        self.range_opt.value = str(self.log_offset)
        self.dpp_opt.value = str(round(self.distance_per_pixel, 6))
        self.stp_opt.value = str(round(self.stepsize, 6))
        self.opa_opt.value = '1' if self.rend_opacity else '0'
        self.snap_opt.value = str(self.rend.snap)

        self.rend.output = None
        self.rend.render_command = ('i_render', (self.channel, self.azimuth, -self.altitude),
                                    dict(opacity=self.rend_opacity, verbose=True))
        self.rend.condition.release()

    def update_display(self):
        if self.update_display_enabled and self.rend.output_for_display and \
           self.rend.output is not None and self.rend.output is not self.display_data:
            if self.rend.condition.acquire(False):
                self.display_data = self.rend.output

                data = self.display_data[0] if isinstance(self.display_data, tuple) else self.display_data
                if len(data.shape) != 2:
                    return
                data = np.log10(data)
                data = (data + self.log_offset) * 255 / (data.max() + self.log_offset)
                data = np.clip(data, 0, 255).astype('uint8')
                self.buffer_array[:data.shape[0], :data.shape[1]] = data

                buf = np.getbuffer(self.buffer_array)

                # then blit the buffer
                self.texture.blit_buffer(buf[:], colorfmt='luminance', bufferfmt='ubyte')
                self.canvas.ask_update()
                self.rend.condition.release()

    def save_image(self):
        output_name = tkFileDialog.asksaveasfilename(title='Image Array Filename')
        if not output_name:
            return
        Thread(target=self._irender_save, args=(output_name, )).start()

    def save_spectra(self):
        output_name = tkFileDialog.asksaveasfilename(title='Spectra Array Filename')
        if not output_name:
            return
        Thread(target=self._ilrender_save, args=(output_name, )).start()

    def _irender_save(self, output_name):
        self.rend.condition.acquire()
        self.rend.output = None
        self.rend.output_for_display = False
        self.rend.render_command = ('i_render', (self.channel, self.azimuth, -self.altitude),
                                    dict(opacity=self.rend_opacity, verbose=False))
        self.rend.condition.wait()
        self.rend.save_irender(output_name, self.rend.output[0] if self.rend_opacity else self.rend.output)
        self.rend.condition.release()

    def _ilrender_save(self, output_name):
        self.rend.condition.acquire()
        self.rend.output = None
        self.rend.output_for_display = False
        self.rend.render_command = ('il_render', (self.channel, self.azimuth, -self.altitude),
                                    dict(opacity=self.rend_opacity, verbose=False))
        self.rend.condition.wait()
        self.rend.save_ilrender(output_name, self.rend.output)
        self.rend.condition.release()

    def save_range(self):
        self.saverangedialog.rend_choice = None
        self.saverangedialog.open()

    def _renderrangefromdialog(self, srd, choice):
        snap_bounds = sorted((int(srd.slider_snapmin.value), int(srd.slider_snapmax.value)))
        snap_skip = int(srd.slider_snapskip.value)
        snap_range = range(snap_bounds[0], snap_bounds[1], snap_skip)
        channellist = self.channellist
        channel_ids = [channellist.index(lib.text) for lib in srd.channelselect.adapter.selection]
        save_loc = srd.savefilename.text
        save_loct = Template(save_loc)
        if len(snap_range) > 1 and '${num}' not in save_loc or len(channel_ids) > 1 and '${chan}' not in save_loc:
            ed = ErrorDialog()
            ed.errortext = 'Missing "${num}" or "${chan}" in file descriptor'
            ed.open()
            return

        total_renders = len(snap_range) * len(channel_ids)

        def saveloop():
            finished_count = 0
            for snap in snap_range:
                for channel_id in channel_ids:
                    finished_count += 1
                    self.renderrange_progress = (finished_count, total_renders)
                    print(snap_range, self.channel, self.azimuth, self.altitude, self.rend_opacity)
                    save_file = save_loct.substitute(num=str(snap), chan=channellist[channel_id])

                    self.rend.set_snap(snap)
                    if choice == 'il':
                        self._ilrender_save(save_file)
                    elif choice == 'i':
                        self._irender_save(save_file)
            self.renderrange_progress = (0, 0)

        Thread(target=saveloop).start()


class ProgressBarUpdating(ProgressBar):
    def __init__(self, update_interval=0.03, **kwargs):
        self.update_interval = update_interval
        super(ProgressBarUpdating, self).__init__(**kwargs)
        self.orig_y = self.y
        self.max = 1
        Clock.schedule_interval(self.update, update_interval)

    def update(self, dt):
        value, max = self.check_progress()
        if max == 0:
            self.y = -10000  # hide
        else:
            self.y = self.orig_y
            value *= (0.94 / max)
            self.value = value + 0.06

    def check_progress(self):
        import time
        return (time.time() % 10 * 10, 100)


class SaveRangeDialog(Popup):
    def __init__(self, render_widget, **kwargs):
        self.render_widget = render_widget
        self.channellist = render_widget.channellist
        self.snap_range = render_widget.rend.snap_range
        super(SaveRangeDialog, self).__init__(**kwargs)


class ErrorDialog(Popup):
    pass


class RenderApp(App):
    def __init__(self, rend):
        super(RenderApp, self).__init__(use_kivy_settings=False)
        self.rend = rend

    def build(self):
        game = RenderGUI(self.rend)
        game.update()
        return game

    def on_close(self):
        from RendererController import cleanup
        cleanup()


def show_renderer(rend):
    app = RenderApp(rend)
    app.run()
