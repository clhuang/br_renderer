import math as m
import os

import numpy as np
from scipy.interpolate import interp1d

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

BLOCKSIZE = 256
MAXGRIDSIZE = 10000000


class Renderer(object):
    '''
    Superclass for rendering 3D models.

    Takes care of most useful functions for 3D rendering,
    such as calculating view vectors, calculating start points
    for individual pixels, determining whether or not in box,
    splitting up large datasets along the x-axis, etc.

    Everything else, e.g. iterating through the spaces
    and integrating must be done by the attached CUDA module
    and the parameter spec_render to render().

    To see the variables provided for use in the CUDA module,
    see renderer.h
    '''
    projection_x_size = 640  # size of the output array
    projection_y_size = 640
    stepsize = 0.2  # size of a step along the LOS
    distance_per_pixel = 0.06  # distance (specified by axes) between pixels in the output

    x_pixel_offset = 0  # if 0, the output is centered in the box center--shifts output L/R
    y_pixel_offset = 0  # shifts output U/D

    mod = None  # CUDA kernel code
    xaxis = yaxis = zaxis = ixaxis = iyaxis = izaxis = None
    textures = {}  # stores references to textures to prevent them being cleared

    progress = (0, 0)

    def load_texture(self, name, arr):
        '''
        Loads an array into a texture with a name.

        Address by the name in the kernel code.
        '''
        tex = self.mod.get_texref(name)  # x*y*z
        arr = arr.astype('float32')

        if len(arr.shape) == 3:
            carr = arr.copy('F')
            texarray = numpy3d_to_array(carr, 'F')
            tex.set_array(texarray)
        else:
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, 1)
            tex.set_flags(0)
            cuda.matrix_to_texref(arr, tex, order='F')

        tex.set_address_mode(0, cuda.address_mode.CLAMP)
        tex.set_address_mode(1, cuda.address_mode.CLAMP)
        tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        tex.set_filter_mode(cuda.filter_mode.LINEAR)
        self.textures[name] = tex

    def load_constant(self, name, val):
        '''
        Loads a constant into memory by name in kernel code.

        If val is a float, int, char, etc., it must be wrapped by
        np.float32() or np.int32() or similar.
        '''
        cuda.memcpy_htod(self.mod.get_global(name)[0], val)

    def clear_textures(self):
        '''
        Removes all textures from memory.
        '''
        self.textures = {}

    def render(self, azimuth, altitude,
               consts, tables, split_tables,
               spec_render, verbose=True):
        '''
        Renders with a view of azimuth and altitude.
        Loads constants (list of tuples of name, value),
        tables (list of tuples of name, value).
        It splits up the rendering space along the x-axis,
        loading a corresponding portion of each table in split_tables,
        and rendering a portion at a time by calling spec_render.

        spec_render is func with format
        spec_render(self, blocksize, gridsize)
        and is where the CUDA kernel is actually called.
        '''

        self.clear_textures()
        view_x, view_y, view_vector = view_axes(azimuth, altitude)

        #add axis inverse lookup tables, as well as x axis lookup table
        tables.extend([('xtex', np.expand_dims(self.xaxis, 1)),
                       ('ixtex', np.expand_dims(self.ixaxis, 1)),
                       ('iytex', np.expand_dims(self.iyaxis, 1)),
                       ('iztex', np.expand_dims(self.izaxis, 1))])

        for tup in tables:
            self.load_texture(*tup)
        for tup in consts:
            self.load_constant(*tup)

        if verbose:
            print('Loaded textures, computed emissivities')

        xsplitsize = MAXGRIDSIZE / (self.yaxis.size * self.zaxis.size)
        numsplits = (self.xaxis.size + xsplitsize - 1) / xsplitsize

        self.load_constant('viewVector', view_vector)
        self.load_constant('viewX', view_x)
        self.load_constant('viewY', view_y)
        self.load_constant('ds', np.float32(self.stepsize))
        self.load_constant('projectionXsize', np.int32(self.projection_x_size))
        self.load_constant('projectionYsize', np.int32(self.projection_y_size))
        self.load_constant('distancePerPixel', np.float32(self.distance_per_pixel))
        self.load_constant('xPixelOffset', np.float32(self.x_pixel_offset))
        self.load_constant('yPixelOffset', np.float32(self.y_pixel_offset))

        for i in xrange(numsplits):
            self.progress = (i, numsplits)
            xstart = i * xsplitsize
            if xstart + xsplitsize > self.xaxis.size:
                xsplitsize = self.xaxis.size - xstart

            if verbose:
                print('Rendering x\'-coords ' + str(xstart) + '-' +
                      str(xstart + xsplitsize) + ' of ' + str(self.xaxis.size))

            for tup in split_tables:
                name, table = tup
                self.load_texture(name, table[xstart:xstart + xsplitsize])

            data_size = self.projection_x_size * self.projection_y_size
            grid_size = (data_size + BLOCKSIZE - 1) / BLOCKSIZE

            self.load_constant('xstart', np.int32(xstart))
            self.load_constant('sliceWidth', np.int32(xsplitsize))

            spec_render(self, BLOCKSIZE, grid_size)

        self.progress = (0, 0)

    def __init__(self, cuda_code):
        '''
        Initialize with a cuda kernel.

        CUDA kernel should include renderer.h to utilize this
        '''
        cuda.init()

        from pycuda.tools import make_default_context
        global context
        self.context = make_default_context()

        self.cuda_code = cuda_code

    def set_axes(self, xaxis, yaxis, zaxis):
        '''
        Sets the x, y, and z axes.

        x, y, z axes are lists correlating indices (in textures)
        to locations in space.
        '''
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.zaxis = zaxis
        self.mod = SourceModule(self.cuda_code, no_extern_c=True,
                                include_dirs=['/Developer/NVIDIA/CUDA-5.0/samples/common/inc',
                                              os.path.dirname(os.path.abspath(__file__))])
        self.load_constant('xmin', np.float32(xaxis.min()))
        self.load_constant('xmax', np.float32(xaxis.max()))
        self.load_constant('ymin', np.float32(yaxis.min()))
        self.load_constant('ymax', np.float32(yaxis.max()))
        self.load_constant('zmin', np.float32(zaxis.min()))
        self.load_constant('zmax', np.float32(zaxis.max()))
        self.load_constant('xtotalsize', np.int32(xaxis.size))

        self.ixaxis = norm_inverse_axis(xaxis)
        self.iyaxis = norm_inverse_axis(yaxis)
        self.izaxis = norm_inverse_axis(zaxis)


def numpy3d_to_array(np_array, order=None):
    '''
    Method for copying a numpy array to a CUDA array

    If you get a buffer error, run this method on np_array.copy('F')
    '''
    from pycuda.driver import Array, ArrayDescriptor3D, Memcpy3D, dtype_to_array_format
    if order is None:
        order = 'C' if np_array.strides[0] > np_array.strides[2] else 'F'

    if order.upper() == 'C':
        d, h, w = np_array.shape
    elif order.upper() == "F":
        w, h, d = np_array.shape
    else:
        raise Exception("order must be either F or C")

    descr = ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0

    device_array = Array(descr)

    copy = Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d

    copy()

    return device_array


def view_axes(azimuth, altitude):
    '''
    The vectors that correspond to a POV with azimuth and altitude

    view_vector is the line following the LOS,
    view_x and view_y are the directions you shift
    in space as you move left/right
    '''
    altitude = sorted((altitude, 90, -90))[1]
    altitude = -altitude
    azimuth = azimuth % 360 - 180

    azimuth = m.radians(int(azimuth))
    altitude = m.radians(int(altitude))

    view_vector = np.array((m.cos(altitude) * m.cos(azimuth), m.cos(altitude) * m.sin(azimuth), m.sin(altitude)))
    view_x = np.array((m.sin(azimuth), -m.cos(azimuth), 0))
    view_y = np.cross(view_x, view_vector)

    return (view_x.astype('float32'), view_y.astype('float32'), view_vector.astype('float32'))


def norm_inverse_axis(axis):
    '''
    From an axis (corresponds table indices to real coordinates),
    generate an inverse lookup table that will take real coordinates
    (normalized to (0, 1)) and return a normalized (0, 1) lookup index
    '''
    axinverse = interp1d((axis - axis.min()) / np.ptp(axis),
                         np.linspace(0, 1, axis.size))
    return axinverse(np.linspace(0, 1, axis.size * 6)).astype('float32')
