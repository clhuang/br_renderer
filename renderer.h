#ifndef INCLUDE_RENDERER
#define INCLUDE_RENDERER
#include "helper_math.h"

//range of x, y, z axes
__constant__ float xmin;
__constant__ float xmax;
__constant__ float ymin;
__constant__ float ymax;
__constant__ float zmin;
__constant__ float zmax;

__constant__ int xtotalsize;

__constant__ float3 viewVector; //direction of view as a vector: if any components are negative zero, will not work
__constant__ float3 viewX; //direction to move as we move along the output x-axis
__constant__ float3 viewY; //direction to move as we move along the output y-axis
__constant__ float ds; //size of increment
__constant__ int projectionXsize; //x-dimension of the output array
__constant__ int projectionYsize; //y-dimension of the output array
__constant__ float distancePerPixel; //real distance between adjacent pixels

__constant__ int xstart; //start index of current slice
__constant__ int sliceWidth; //width of current slice

__constant__ float xPixelOffset; //center of box (not slice) is normally denoted by center pixel in output
__constant__ float yPixelOffset; //allows offset of box center by non-integral number of pixels

/*
 * Convert a space vector (from (xmin,ymin,zmin) to (xmax,ymax,zmax))
 * to a normalized vector suitable for texture lookup
 */
__device__ float3 realToNormalized(float3 vector) {
    return make_float3((vector.x - xmin) / (xmax - xmin),
            (vector.y - ymin) / (ymax - ymin),
            (vector.z - zmin) / (zmax - zmin));
}

/*
 * Gets the initial starting point for an output point at xpixel,ypixel
 * Returns (INFINITY, INFINITY, INFINITY) for something that doesn't intersect box
 */
__device__ float3 initialCP(int xpixel, int ypixel) {
    float3 boxCenter = make_float3((xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2);
    float3 rayOrigin = boxCenter -
        ((projectionXsize / 2 - xpixel + xPixelOffset) * distancePerPixel) * viewX +
        ((projectionYsize / 2 - ypixel + yPixelOffset) * distancePerPixel) * viewY;

    float xpmin = xmin + (xmax - xmin) * xstart / xtotalsize;
    float xpmax = xmin + (xmax - xmin) * (xstart + sliceWidth) / xtotalsize;
    float3 tnear, tfar;
    float tn, tf;

    tnear.x = (((viewVector.x >= 0) ? xpmin : xpmax) - rayOrigin.x) / viewVector.x;
    tnear.y = (((viewVector.y >= 0) ? ymin : ymax) - rayOrigin.y) / viewVector.y;
    tnear.z = (((viewVector.z >= 0) ? zmin : zmax) - rayOrigin.z) / viewVector.z;
    tfar.x = (((viewVector.x < 0) ? xpmin : xpmax) - rayOrigin.x) / viewVector.x;
    tfar.y = (((viewVector.y < 0) ? ymin : ymax) - rayOrigin.y) / viewVector.y;
    tfar.z = (((viewVector.z < 0) ? zmin : zmax) - rayOrigin.z) / viewVector.z;

    tn = tnear.x;
    if (tnear.y > tn) tn = tnear.y;
    if (tnear.z > tn) tn = tnear.z;

    tf = tfar.x;
    if (tfar.y < tf) tf = tfar.y;
    if (tfar.z < tf) tf = tfar.z;

    if (tf < tn)
            return make_float3(INFINITY, INFINITY, INFINITY);

    return rayOrigin + viewVector * tn;
}

/**
 * Returns whether or not currentPosition is still within the box
 * currentPosition is actual position in box
 */
__device__ bool isInSlice(float3 currentPosition) {
    return currentPosition.z >= zmin &&
        currentPosition.z <= zmax &&
        currentPosition.y >= ymin &&
        currentPosition.y <= ymax &&
        currentPosition.x >= xmin + (xmax - xmin) * xstart / (float) xtotalsize &&
        currentPosition.x <= xmin + (xmax - xmin) * (xstart + sliceWidth) / (float) xtotalsize;
}

#endif
