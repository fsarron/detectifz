#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

__all__ = ["map_dtfe2d", "map_dtfe3d"]

@nb.njit()
def get_densities2d(simplices, points):
    areas = get_areas(simplices, points)
    npoints = len(points)
    true_vol = np.zeros(npoints)
    for inds, simplex in enumerate(simplices):
        true_vol[simplex] += areas[inds]    
    return 3. / true_vol

@nb.njit()
def get_areas(simplices, points):
    x, y = points.T
    a,_=simplices.shape
    i = np.arange(a)
    areas = np.abs (0.5*( (x[simplices[i,1]]-x[simplices[i,0]])*
                         (y[simplices[i,2]]-y[simplices[i,0]]) - \
                             (x[simplices[i,2]]-x[simplices[i,0]])*
                         (y[simplices[i,1]]-y[simplices[i,0]]) ))
    return areas


def how_many(tri, num_points):
    there_are = np.where(tri.simplices == num_points)[0]
    return there_are

def vol_tetrahedron(points):
    return abs(
        np.dot(
            (points[0] - points[3]),
            np.cross(
                (points[1] - points[3]),
                (points[2] - points[3])))) / 6.


def vol_delaunay(inputs):
    tri, num = inputs
    a = vol_tetrahedron(tri.points[tri.simplices[num]])
    return a


def get_volumes(tri, the_pool):
    shape = tri.simplices.shape[0]
    volumes = np.empty(shape)
    all_inputs = zip(repeat(tri), np.arange(shape))
    volumes = the_pool.map(vol_delaunay, all_inputs)
    return np.array(volumes)


def get_densities3d(inputs):
    tri, num, volumes = inputs
    l = how_many(tri, num)
    true_vol = np.sum(volumes[l]) / 4.
    return 1. / true_vol


def densities3d(tri, the_pool, volumes):
    shape = tri.points.shape[0]
    dens = np.empty(shape)
    all_inputs = zip(repeat(tri), np.arange(shape), repeat(volumes))
    dens = the_pool.map(get_densities3d, all_inputs)
    return np.array(dens)


@nb.njit()
def get_one_smooth(d, indices, indptr):
    npoints = len(d)
    dsmooth = np.zeros(npoints)
    for vert in range(npoints):
        dsmooth[vert] = np.mean(
        d[indptr[indices[vert]:indices[vert+1]]])   
    return dsmooth

@nb.njit()
def compute_dsmooth(d, indices, indptr, nsmooth):
    for _ in range(nsmooth):
        dsmooth = get_one_smooth(d, indices, indptr)
        d = dsmooth 
    return dsmooth




def map_dtfe3d(x, y, z, xsize, ysize=None, zsize=None):
    """
    Create a 3d density cube from given x, y and z points in a volume

    Parameters
    ----------
    x : An N-length one dimensional array
    The x coordinates of the point distribution
    y : An N-length one dimensional array
    The y coordinates of the point distribution
    z : An N-length one dimensional array
    The z coordinates of the point distribution
    xsize : Integer
    The x dimension of the cube
    ysize : Integer, optionale
    The y dimension of the cube. If ysize is not given, it assumes that the x, y and z axis share the same dimension
    zsize : Integer, optional
    The z dimension of the cube. If zsize is not given, it assumes that the x, y and z axis share the same dimension

    Returns
    -------
    grid : An (xsize, ysize, zsize)-shaped array
    The density cube in 3d

    """
    tab = np.vstack((x, y, z)).T
    tri = Delaunay(tab)
    the_pool = Pool()
    volumes = get_volumes(tri, the_pool)
    d = densities3d(tri, the_pool, volumes)
    the_pool.close()
    if (ysize is None) & (zsize is None):
        size = xsize
        x_m = np.linspace(np.min(x), np.max(x), size)
        y_m = np.linspace(np.min(y), np.max(y), size)
        z_m = np.linspace(np.min(z), np.max(z), size)
    else:
        x_m = np.linspace(np.min(x), np.max(x), xsize)
        y_m = np.linspace(np.min(y), np.max(y), ysize)
        z_m = np.linspace(np.min(z), np.max(z), zsize)
    x_m, y_m, z_m = np.meshgrid(x_m, y_m, z_m)
    grid = griddata(tab, d, (x_m, y_m, z_m), method='linear')
    return grid

def map_dtfe2d(all_inputs):
    """
    Create a 2d density map from given x and y points in a plan

    Parameters
    ----------
    x : An N-length one dimensional array
    The x coordinates of the point distribution
    y : An N-length one dimensional array
    The y coordinates of the point distribution
    xsize : Integer, optional
    The x dimension of the map. If xsize if not given, the algorithm compute a best resolution in x and y,
    taking statistically a pixel size under ten times the square root of the 5 sigma area. The algorithm
    try to give the same resolution in x and y
    ysize : Integer, optional
    The y dimension of the map. If ysize is not given, ysize take the value of xsize, if given

    Returns
    -------

    grid : An 2 dimensional array
    The density map in 2d

    """
    x, y, xsize, ysize, proba, fixed_grid,xminmax,yminmax,Nsmooth=all_inputs

    if len(x) >= 4:
        tab = np.vstack((x, y)).T
        tri = Delaunay(tab)
        d = get_densities2d(tri.simplices, tri.points)
        if proba is not None:
            d = d*proba
        if Nsmooth > 0:
            indices, indptr = tri.vertex_neighbor_vertices
            d = compute_dsmooth(d, indices, indptr, Nsmooth)
        
        if fixed_grid == False:
            if (xsize is None) & (ysize is None):
                if y.max()-y.min() < x.max()-x.min():
                    xsize = int(1/(np.sqrt(1/(np.median(d)+5*np.std(d)))/(10/((y.max()-y.min())/(x.max()-x.min())))))
                    ysize = int(1/(np.sqrt(1/(np.median(d)+5*np.std(d)))/10))
                else:
                    xsize = int(1/(np.sqrt(1/(np.median(d)+5*np.std(d)))/10))
                    ysize = int(1/(np.sqrt(1/(np.median(d)+5*np.std(d)))/(10*((y.max()-y.min())/(x.max()-x.min())))))
                x_m = np.linspace(np.min(x), np.max(x), xsize)
                y_m = np.linspace(np.min(y), np.max(y), ysize)
            elif (xsize is not None) & (ysize is None):
                size = xsize
                x_m = np.linspace(np.min(x), np.max(x), size)
                y_m = np.linspace(np.min(y), np.max(y), size)
            else:
                x_m = np.linspace(np.min(x), np.max(x), xsize)
                y_m = np.linspace(np.min(y), np.max(y), ysize)
            x_m, y_m = np.meshgrid(x_m, y_m)
        elif fixed_grid == True:
            if (xminmax is None) or (yminmax is None) or (xsize is None) or (ysize is None):
                return 'Error : if fixed_grid is true, you need to prcvide the grid limits and sizes in both directions'
            else:
                x_m = np.linspace(np.min(xminmax), np.max(xminmax), xsize)
                y_m = np.linspace(np.min(yminmax), np.max(yminmax), ysize)
                x_m, y_m = np.meshgrid(x_m, y_m)
        grid = LinearNDInterpolator(tri, d)(x_m,y_m)

    else:
        grid = np.zeros((ysize,xsize))
    return grid

