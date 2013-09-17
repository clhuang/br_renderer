import math as m
import os

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

BLOCKSIZE = 256
MAXGRIDSIZE = 10000000

class Renderer:
    projectionXsize = 640 #size of the output array
    projectionYsize = 640
    stepsize = 0.2 #size of a step along the LOS
    distancePerPixel = 0.06 #distance (specified by axes) between pixels in the output

    xPixelOffset = 0 #if 0, the output is centered in the box center--shifts output L/R
    yPixelOffset = 0 #shifts output U/D

    textures = {}

    def loadTexture(self, name, arr):
        '''
        Loads an array into a texture with a name.
        Address by the name in the kernel code.
        '''
        tex = self.mod.get_texref(name) #x*y*z

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

    def loadConst(self, name, val):
        '''
        Loads a constant into memory by name in kernel code.
        If val is a float or an int, it must be wrapped by
        np.float32() or np.int32() or similar.
        '''
        cuda.memcpy_htod(self.mod.get_global(name)[0], val)

    def clearTextures(self):
        self.textures = {}

    def render(self, azimuth, altitude,
            consts, tables, splitTables,
            spec_render, verbose=True):
        '''
        Renders with a view of azimuth and altitude.
        Loads constants (list of tuples of name, value),
        tables (list of tuples of name, value).
        It splits up the rendering space along the x-axis,
        loading a corresponding portion of each table in splitTables,
        and rendering a portion at a time by calling spec_render.

        spec_render has format spec_render(self, blocksize, gridsize)
        and is where the CUDA kernel is actually called.
        '''
        self.clearTextures()
        viewX, viewY, viewVector = viewAxes(azimuth, altitude)

        for tup in tables:
            self.loadTexture(*tup)
        for tup in consts:
            self.loadConst(*tup)

        if verbose:
            print('Loaded textures, computed emissivities')

        xsplitsize = MAXGRIDSIZE / (self.yaxis.size * self.zaxis.size)
        numSplits = (self.xaxis.size + xsplitsize - 1) / xsplitsize

        self.loadConst('viewVector', viewVector)
        self.loadConst('viewX', viewX)
        self.loadConst('viewY', viewY)
        self.loadConst('ds', np.float32(self.stepsize))
        self.loadConst('projectionXsize', np.int32(self.projectionXsize))
        self.loadConst('projectionYsize', np.int32(self.projectionYsize))
        self.loadConst('distancePerPixel', np.float32(self.distancePerPixel))
        self.loadConst('xPixelOffset', np.float32(self.xPixelOffset))
        self.loadConst('yPixelOffset', np.float32(self.yPixelOffset))

        for i in range(numSplits):
            xstart = i * xsplitsize
            if xstart + xsplitsize > self.xaxis.size:
                xsplitsize = self.xaxis.size - xstart

            if verbose:
                print('Rendering x\'-coords ' + str(xstart) + '-' +
                        str(xstart + xsplitsize) + ' of ' + str(self.xaxis.size))

            for tup in splitTables:
                name, table = tup
                self.loadTexture(name, table[xstart:xstart+xsplitsize])

            dataSize = self.projectionXsize * self.projectionYsize
            gridSize = (dataSize + BLOCKSIZE - 1) / BLOCKSIZE

            self.loadConst('xstart', np.int32(xstart))
            self.loadConst('sliceWidth', np.int32(xsplitsize))

            spec_render(self, BLOCKSIZE, gridSize)

    def __init__(self, cudaCode):
        '''
        Initialize with a cuda kernel.
        '''
        self.cudaCode = cudaCode

    def setAxes(self, xaxis, yaxis, zaxis):
        '''
        Sets the x, y, and z axes.
        '''
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.zaxis = zaxis
        self.xarange = np.ptp(xaxis)
        self.yarange = np.ptp(yaxis)
        self.zarange = np.ptp(zaxis)
        self.mod = SourceModule(self.cudaCode, no_extern_c=True,
                include_dirs=['/Developer/NVIDIA/CUDA-5.0/samples/common/inc',
                    os.path.dirname(os.path.abspath(__file__))])
        self.loadConst('xmin', np.float32(xaxis.min()))
        self.loadConst('xmax', np.float32(xaxis.max()))
        self.loadConst('ymin', np.float32(yaxis.min()))
        self.loadConst('ymax', np.float32(yaxis.max()))
        self.loadConst('zmin', np.float32(zaxis.min()))
        self.loadConst('zmax', np.float32(zaxis.max()))
        self.loadConst('xtotalsize', np.int32(xaxis.size))

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
        raise LogicError, "order must be either F or C"

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

def viewAxes(azimuth, altitude):
    if altitude > 90: altitude = 90
    if altitude < -90: altitude = -90
    altitude = -altitude
    azimuth = azimuth % 360 - 180

    azimuth = m.radians(int(azimuth))
    altitude = m.radians(int(altitude))

    viewVector = np.array((m.cos(altitude) * m.cos(azimuth), m.cos(altitude) * m.sin(azimuth), m.sin(altitude)))
    viewX = np.array((m.sin(azimuth), -m.cos(azimuth), 0))
    if (altitude == m.pi/2):
        viewY = np.array((-m.cos(azimuth), -m.sin(azimuth), 0))
    elif (altitude == -m.pi/2):
        viewY = np.array((m.cos(azimuth), m.sin(azimuth), 0))
    else:
        viewY = np.cross(viewX, viewVector)
        viewY /= np.linalg.norm(viewY)

    return (viewX.astype('float32'), viewY.astype('float32'), viewVector.astype('float32'))
