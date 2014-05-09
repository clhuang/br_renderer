br_renderer
===========

Basic framework for ray tracing with CUDA, especially for applications that require splitting of the render domain. 
Easily calculates view vectors and start points based on pixel locations, determining whether or not in bounds, etc.

Actual integration and iteration is done by an attached CUDA module (written by the user, and likely will include renderer.cuh)
