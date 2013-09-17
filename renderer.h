#ifndef INCLUDE_RENDERER
#define INCLUDE_RENDERER
#include "helper_math.h"

__constant__ float xmin;
__constant__ float xmax;
__constant__ float ymin;
__constant__ float ymax;
__constant__ float zmin;
__constant__ float zmax;

__constant__ int xtotalsize;

__constant__ float3 viewVector; //direction of view as a vector
__constant__ float3 viewX; //direction to move as we move along the output x-axis
__constant__ float3 viewY; //direction to move as we move along the output y-axis 
__constant__ float ds; //size of increment
__constant__ int projectionXsize;
__constant__ int projectionYsize;
__constant__ float distancePerPixel;

__constant__ int xstart;
__constant__ int sliceWidth;

__constant__ float xPixelOffset;
__constant__ float yPixelOffset;

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
 */
__device__ float3 initialCP(int xpixel, int ypixel) {
    float3 boxCenter = make_float3((xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2);
    float3 rayOrigin = boxCenter -
        ((projectionXsize / 2 - xpixel + xPixelOffset) * distancePerPixel) * viewX +
        ((projectionYsize / 2 - ypixel + yPixelOffset) * distancePerPixel) * viewY;
    float3 intersection1, intersection2;
    intersection1 = make_float3(INFINITY, INFINITY, INFINITY);
    intersection2 = make_float3(INFINITY, INFINITY, INFINITY);
    float3 possibleIntersect;

    float xpmin = xmin + (xmax - xmin) * xstart / xtotalsize;
    float xpmax = xmin + (xmax - xmin) * (xstart + sliceWidth) / xtotalsize;

    possibleIntersect = rayOrigin + viewVector * (zmax - rayOrigin.z) / viewVector.z;
    if (possibleIntersect.x <= xpmax && possibleIntersect.x >= xpmin &&
            possibleIntersect.y <= ymax && possibleIntersect.y >= ymin) {
        if (intersection1.x == INFINITY)
            intersection1 = possibleIntersect;
        else
            intersection2 = possibleIntersect;
    }

    possibleIntersect = rayOrigin + viewVector * (zmin - rayOrigin.z) / viewVector.z;
    if (possibleIntersect.x <= xpmax && possibleIntersect.x >= xpmin &&
            possibleIntersect.y <= ymax && possibleIntersect.y >= ymin) {
        if (intersection1.x == INFINITY)
            intersection1 = possibleIntersect;
        else
            intersection2 = possibleIntersect;
    }

    possibleIntersect = rayOrigin + viewVector * (ymax - rayOrigin.y) / viewVector.y;
    if (possibleIntersect.x <= xpmax && possibleIntersect.x >= xpmin &&
            possibleIntersect.z <= zmax && possibleIntersect.z >= zmin) {
        if (intersection1.x == INFINITY)
            intersection1 = possibleIntersect;
        else
            intersection2 = possibleIntersect;
    }

    possibleIntersect = rayOrigin + viewVector * (ymin - rayOrigin.y) / viewVector.y;
    if (possibleIntersect.x <= xpmax && possibleIntersect.x >= xpmin &&
            possibleIntersect.z <= zmax && possibleIntersect.z >= zmin) {
        if (intersection1.x == INFINITY)
            intersection1 = possibleIntersect;
        else
            intersection2 = possibleIntersect;
    }

    possibleIntersect = rayOrigin + viewVector * (xpmax - rayOrigin.x) / viewVector.x;
    if (possibleIntersect.y <= ymax && possibleIntersect.y >= ymin &&
            possibleIntersect.z <= zmax && possibleIntersect.z >= zmin) {
        if (intersection1.x == INFINITY)
            intersection1 = possibleIntersect;
        else
            intersection2 = possibleIntersect;
    }

    possibleIntersect = rayOrigin + viewVector * (xpmin - rayOrigin.x) / viewVector.x;
    if (possibleIntersect.y <= ymax && possibleIntersect.y >= ymin &&
            possibleIntersect.z <= zmax && possibleIntersect.z >= zmin) {
        if (intersection1.x == INFINITY)
            intersection1 = possibleIntersect;
        else
            intersection2 = possibleIntersect;
    }

    if (intersection2.x == INFINITY) return intersection1;

    return dot(intersection1 - intersection2, viewVector) > 0 ? intersection2 : intersection1;
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
