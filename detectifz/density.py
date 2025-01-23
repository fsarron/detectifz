import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import LinearNDInterpolator
from .fastdtfe import map_dtfe2d
from .fastdtfe_nogrid import dtfe2d

from astropy import wcs
from PIL import Image, ImageDraw
import astropy.convolution
import astropy.stats
from astropy.io import fits
from astropy import units
from astropy.wcs.utils import proj_plane_pixel_area

import ray
import psutil
import time
#import random
from shapely import geometry

from scipy.ndimage.filters import gaussian_filter1d

from pathlib import Path

from reproject import reproject_interp

from joblib import Parallel, delayed

from .utils import nan_helper, angsep_radius, physep_ang

import warnings
from astropy.utils.exceptions import AstropyWarning,AstropyUserWarning
###missing many imports  

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def fill_boundary(ra, dec, nfill, area, boundary_width):
    """Add randomly distributed particles on the conve hull
    boundary of a distribution of points.

    Parameters
    ----------
    ra : np.array, right ascenscion of points
    dec : np.array, declination of points
    nfill: float, density of random points to add in deg-2
    area: float, unmasked area of field
    boundary_width: float, with of boundary region in degrees

    Returns
    -------
    point distribution (ra,dec) with random points on the
    boundary added at the beggining
    """
    np.random.seed(12365)

    points = np.c_[ra, dec]
    hull = ConvexHull(points)

    #pointList = geometry.MultiPoint(points[hull.vertices])
    pointList = [geometry.Point(p) for p in points[hull.vertices]]
    #poly = geometry.Polygon(pointList.coords)
    poly = geometry.Polygon([[p.x, p.y] for p in pointList])
    #polys = [geometry.Polygon([p.x, p.y]) for p in pointList]
    #poly = geometry.MultiPolygon(polygons=ploys)

    poly_buff = poly.buffer(boundary_width)
    hullin_p = np.array(poly_buff.boundary.coords)
    minmax_buff = [
        np.min(hullin_p[:, 0]),
        np.max(hullin_p[:, 0]),
        np.min(hullin_p[:, 1]),
        np.max(hullin_p[:, 1]),
    ]

    ramin, ramax, decmin, decmax = minmax_buff
    
    area_max = (ramax-ramin)*(decmax-decmin) 
    border_area = area_max - area
    Nfill = int(nfill * border_area)
    
    rardn, decrdn = np.random.uniform(ramin, ramax, Nfill), np.random.uniform(
        decmin, decmax, Nfill
    )
    
    ra_corners = np.array([ramin, ramin, ramax, ramax])
    dec_corners = np.array([decmin, decmax, decmax, decmin])
    
    rardn = np.concatenate([rardn, ra_corners])
    decrdn = np.concatenate([decrdn, dec_corners])

    radecrdn = np.c_[rardn, decrdn]
    radecfill = radecrdn[np.logical_not(in_hull(radecrdn, points[hull.vertices]))]

    return radecfill


def convex_hull_toimage(ra, dec, head, im):
    """Make a mask from the Convex Hull of a distribution of points.

    Parameters
    ----------
    ra : np.array, right ascenscion of points
    dec : np.array, declination of points
    head: astropy.io.fits  `Header` object, header used for mask projection
    im: np.array, an image corresponding to header

    Returns
    -------
    Convex Hull mask : np.array, mask of same size as im. True in the region
    where points are located, False outside
    """
    points = np.c_[ra, dec]
    pix = wcs.WCS(head).wcs_world2pix(points, 0)
    hull = ConvexHull(pix)
    verts = [(pix[v, 1], pix[v, 0]) for v in hull.vertices]
    img = Image.new("L", im.shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img,dtype=np.int64)

    return (mask.T).astype(bool)


@ray.remote(max_calls=10)
def get_dtfemc(islice, zslices, galmc_Mlim, Nmc, map_params, masks, headmasks):
    """Make DTFE overdensity map in one redshift slice.

    Parameters
    ----------
    islice: int, slice index
    zslices: np.array of size:3xN, redshift limits of all slices
    galmc_Mlim: np.array, Monte Carlo realization of galaxy catalogue
                with limiting stellar mass enforced
    Nmc: int, number of Monte Carlo realizations
    map_params : np.array or list, list of parameters defining
                the density map resolution and limits defined as
                [xsize,ysize,xminmax,yminmax,pixdeg]
    masks: np.array dtype:bool, mask image
    headmasks: astropy.io.fits `Header` object, mask header

    Returns
    -------
    Convex Hull mask : np.array, mask of same size as im. True in the region
    where points are located, False outside
    """
    xsize, ysize, xminmax, yminmax, pixdeg = map_params
    zslice = zslices[islice, 0]
    print('slice',islice,zslice)
    galmc_Mlim_slice = np.empty(Nmc, dtype="object")
    for imc in range(Nmc):
        galmc_Mlim_slice[imc] = galmc_Mlim[imc][
            np.where(
                (galmc_Mlim[imc][:, 3] >= zslices[islice, 1])
                & (galmc_Mlim[imc][:, 3] < zslices[islice, 2])
            )
        ]
    Ngalslice_mc = np.array([len(galmc_Mlim_slice[imc]) for imc in range(Nmc)])

    galmc_Mlim_slice_master = np.vstack(galmc_Mlim_slice)

    ra_master, dec_master = galmc_Mlim_slice_master[:, 1], galmc_Mlim_slice_master[:, 2]

    random.seed(12365)
    dtfe_mc = np.zeros((ysize, xsize))

    # width of reflected boundary
    boundary_width = min(angsep_radius(zslice, 2).value, 0.1)  # 2Mpc

    # get convex hull mask
    hullmask = convex_hull_toimage(ra_master, dec_master, headmasks, masks)

    for imc in range(Nmc):
        # if Ngalslice_mc > 2:
        ra_mc, dec_mc, mass_mc = (
            galmc_Mlim_slice[imc][:, 1],
            galmc_Mlim_slice[imc][:, 2],
            10 ** galmc_Mlim_slice[imc][:, 4],
        )

        # for each MC realization fill border with random at mean density
        radecfill = fill_boundary(
            ra_master,
            dec_master,
            int(len(galmc_Mlim_slice_master) / Nmc),
            boundary_width,
        )
        massfill = np.repeat(10, len(radecfill))
        ra_mc = np.concatenate([ra_mc, radecfill[:, 0]])
        dec_mc = np.concatenate([dec_mc, radecfill[:, 1]])
        mass_mc = np.concatenate([mass_mc, 10 ** massfill])

        all_inputs = [ra_mc, dec_mc, xsize, ysize, mass_mc, True, xminmax, yminmax, 1]
        #dmap = np.log10(map_dtfe2d(all_inputs))
        #dmap[np.where(np.isnan(dmap))] = -10
        #dmap[np.where(np.isinf(dmap))] = -10
        #dtfe_mc += dmap / Nmc
        dtfe_mc += np.log10(map_dtfe2d(all_inputs)) / Nmc
        # so if less than two galaxies in the slice, count as a zero density contrast everywhere

    kernel = astropy.convolution.Tophat2DKernel(
        radius=0.1 / physep_ang(zslice, pixdeg).value
    )
    dtfe_mc = astropy.convolution.convolve(dtfe_mc, kernel, preserve_nan=True)
    #dd1 = np.copy(dtfe_mc)
    mm = masks * hullmask

    mu, _, sigma = astropy.stats.sigma_clipped_stats(dtfe_mc[mm], sigma=3.0)
    meandens = 10 ** mu * np.exp(2.652 * sigma ** 2)
    dtfe_mc = dtfe_mc - np.log10(meandens)
    dtfe_mc[np.where(np.isinf(dtfe_mc))] = np.nan
    dtfe_mc[~hullmask] = np.nan

    return dtfe_mc



@ray.remote(max_calls=10)
def get_dtfemc_nogrid_ray(islice, zslices, galcat_mc, maskMlim_mc, Nmc, map_params, masks, headmasks):
    """Make DTFE overdensity map in one redshift slice.

    Parameters
    ----------
    islice: int, slice index
    zslices: np.array of size:3xN, redshift limits of all slices
    galmc_Mlim: np.array, Monte Carlo realization of galaxy catalogue
                with limiting stellar mass enforced
    Nmc: int, number of Monte Carlo realizations
    map_params : np.array or list, list of parameters defining
                the density map resolution and limits defined as
                [xsize,ysize,xminmax,yminmax,pixdeg]
    masks: np.array dtype:bool, mask image
    headmasks: astropy.io.fits `Header` object, mask header

    Returns
    -------
    Convex Hull mask : np.array, mask of same size as im. True in the region
    where points are located, False outside
    """
    print('islice', islice)
    xsize, ysize, xminmax, yminmax, pixdeg = map_params
    zslice = zslices[islice, 0]
    
    #galmc_slice = np.empty(Nmc, dtype="object")
    mask_mc = np.empty(Nmc, dtype="object")
    for imc in range(Nmc):
        mask_mc[imc] = ( (galcat_mc[imc][:, 3] >= zslices[islice, 1])
                        & (galcat_mc[imc][:, 3] < zslices[islice, 2]) 
                       & maskMlim_mc[imc])
        #galmc_slice[imc] = galcat_mc[imc][mask_mc[imc]]
    #Ngalslice_mc = np.array([len(galmc_slice[imc]) for imc in range(Nmc)])
    mask_mc_all = np.sum(mask_mc, axis=0)
    #galmc_slice_master = np.vstack(galmc_slice)

    ra_slice, dec_slice = galcat_mc[0, mask_mc_all, 1:3].T
    #points_slice = np.c_[ra_slice, dec_slice]
    
    # get convex hull mask
    hullmask = convex_hull_toimage(ra_slice, dec_slice, headmasks, masks)
        
    #dtfe_mc = np.zeros((ysize, xsize))



    # for points that appear in at least one MC realization fill border with random at mean density
    boundary_width = min(angsep_radius(zslice, 2).value, 0.1)  # width of reflected boundary 2Mpc
    area_unmasked = np.sum(masks) * proj_plane_pixel_area(wcs.WCS(headmasks))
    
    #ngal_fill = int(len(ra_slice) / Nmc)
    nfill = int(np.mean([np.sum(mask_mc[imc][mask_mc_all]) for imc in range(Nmc)]))

    #print('ngal_fill', ngal_fill)
    radecfill = fill_boundary(
        ra_slice,
        dec_slice,
        nfill,
        area_unmasked,
        boundary_width
    )
    #print(len(radecfill))
    #print('ngal_fill', ngal_fill, radecfill)
    ngal_fill = len(radecfill) #-= 1
    massfill = np.repeat(10, ngal_fill)#len(radecfill))
    #ra_dtfe = np.concatenate([galcat_mc[0, :, 1], radecfill[:, 0]])
    #dec_dtfe = np.concatenate([galcat_mc[0, :, 2], radecfill[:, 1]])
    #points_dtfe = np.c_[ra_dtfe,dec_dtfe]
    
    points_dtfe = np.concatenate([galcat_mc[0, :, 1:3][mask_mc_all],
                                  radecfill])
    #points_dtfe = np.concatenate([galcat_mc[0, :, 1:3],
    #                              radecfill])
    
    #densities = np.empty((Nmc,len(points_dtfe)))
    dtfemc = np.zeros(len(points_dtfe))
    for imc in range(Nmc):
        mask_mc_all_imc = mask_mc[imc][mask_mc_all]
        pmask_dtfe = np.concatenate([mask_mc_all_imc,np.repeat(True, ngal_fill)])
        mass_dtfe = np.concatenate([10**galcat_mc[imc,:,4][mask_mc_all][mask_mc_all_imc],10**massfill])
        
        #pmask_dtfe = np.concatenate([mask_mc[imc],np.repeat(True, ngal_fill)])
        #mass_dtfe = np.concatenate([10**galcat_mc[imc,:,4][mask_mc[imc]],10**massfill])
        
        #print('ngal_fill', ngal_fill, len(pmask_dtfe), len(points_dtfe))

        #pmask_dtfe = mask_mc[imc][mask_mc_all]
        #mass_dtfe = 10**galcat_mc[imc,:,4][mask_mc[imc]] #[mask_mc_all]
        
        #print('npoints slice', len(points_dtfe),len(pmask_dtfe),np.sum(pmask_dtfe),len(mass_dtfe))
        #print('islice', islice, 'imc', imc, 'npoints', np.sum(pmask_dtfe))
        
        inputs_dtfe2d = [points_dtfe, pmask_dtfe, mass_dtfe, 1]
        #densities[imc] = dtfe2d(inputs_dtfe2d)
        #densities[imc] = dtfe2d(inputs_dtfe2d)
        dtfemc += np.log10(dtfe2d(inputs_dtfe2d))/Nmc
        
        
    #dtfemc = np.log10(np.nanmedian(densities, axis=0)) 
    #np.savez('dtfemc_'+str(islice)+'.npz', points_dtfe=points_dtfe, 
    #         pmask_dtfe = pmask_dtfe,
    #         mass_dtfe=mass_dtfe,
    #         dtfemc=dtfemc
    #        )
    #dtfemc = np.log10(np.nanmean(densities,axis=0))
    #dtfemc = np.nanmean(np.log10(densities),axis=0)
    #dtfemc = np.nansum(densities,axis=0) #/Nmc
    x_m = np.linspace(np.min(xminmax), np.max(xminmax), xsize)
    y_m = np.linspace(np.min(yminmax), np.max(yminmax), ysize)
    x_m, y_m = np.meshgrid(x_m, y_m)
    if len(points_dtfe[~np.isnan(dtfemc)]) > 3:
        tri = Delaunay(points_dtfe[~np.isnan(dtfemc)])
        grid_dtfemc = LinearNDInterpolator(tri, 
                                dtfemc[~np.isnan(dtfemc)])(x_m,y_m)
    else:
        grid_dtfemc = np.zeros((ysize, xsize))
    kernel = astropy.convolution.Tophat2DKernel(
        radius=0.1 / physep_ang(zslice, pixdeg).value
    )
    grid_dtfemc = astropy.convolution.convolve(grid_dtfemc, kernel, 
                                           preserve_nan=True)
    mm = (masks & 
          hullmask & 
          np.logical_not(np.isnan(grid_dtfemc)))
    #&(grid_dtfemc != 0) )
    #print(grid_dtfemc[mm])
    mu, _, sigma = astropy.stats.sigma_clipped_stats(
        grid_dtfemc[mm], sigma=3.0)
    meandens = 10 ** mu * np.exp(2.652 * sigma ** 2)
    grid_dtfemc = grid_dtfemc - np.log10(meandens)
    grid_dtfemc[np.where(np.isinf(grid_dtfemc))] = np.nan
    grid_dtfemc[~hullmask] = np.nan
    
    return grid_dtfemc




def get_dtfemc_nogrid(islice, zslices, galcat_mc, maskMlim_mc, use_mass_density, Nmc, map_params, masks, headmasks):
    """Make DTFE overdensity map in one redshift slice.

    Parameters
    ----------
    islice: int, slice index
    zslices: np.array of size:3xN, redshift limits of all slices
    galmc_Mlim: np.array, Monte Carlo realization of galaxy catalogue
                with limiting stellar mass enforced
    Nmc: int, number of Monte Carlo realizations
    map_params : np.array or list, list of parameters defining
                the density map resolution and limits defined as
                [xsize,ysize,xminmax,yminmax,pixdeg]
    masks: np.array dtype:bool, mask image
    headmasks: astropy.io.fits `Header` object, mask header

    Returns
    -------
    Convex Hull mask : np.array, mask of same size as im. True in the region
    where points are located, False outside
    """
    #print('islice', islice)
    xsize, ysize, xminmax, yminmax, pixdeg = map_params
    zslice = zslices[islice, 0]
    
    #galmc_slice = np.empty(Nmc, dtype="object")
    mask_mc = np.empty(Nmc, dtype="object")
    for imc in range(Nmc):
        mask_mc[imc] = ( (galcat_mc[imc][:, 3] >= zslices[islice, 1])
                        & (galcat_mc[imc][:, 3] < zslices[islice, 2]) 
                       & maskMlim_mc[imc])
        #galmc_slice[imc] = galcat_mc[imc][mask_mc[imc]]
    #Ngalslice_mc = np.array([len(galmc_slice[imc]) for imc in range(Nmc)])
    mask_mc_all = np.sum(mask_mc, axis=0)
    #galmc_slice_master = np.vstack(galmc_slice)

    ra_slice, dec_slice = galcat_mc[0, mask_mc_all, 1:3].T
    #points_slice = np.c_[ra_slice, dec_slice]
    
    # get convex hull mask
    hullmask = convex_hull_toimage(ra_slice, dec_slice, headmasks, masks)
        
    #dtfe_mc = np.zeros((ysize, xsize))



    # for points that appear in at least one MC realization fill border with random at mean density
    boundary_width = min(angsep_radius(zslice, 2).value, 0.1)  # width of reflected boundary 2Mpc
    area_unmasked = np.sum(masks) * proj_plane_pixel_area(wcs.WCS(headmasks))
    
    #ngal_fill = int(len(ra_slice) / Nmc)
    nfill = int(np.mean([np.sum(mask_mc[imc][mask_mc_all]) for imc in range(Nmc)]))

    #print('ngal_fill', ngal_fill)
    radecfill = fill_boundary(
        ra_slice,
        dec_slice,
        nfill,
        area_unmasked,
        boundary_width
    )
    #print(len(radecfill))
    #print('ngal_fill', ngal_fill, radecfill)
    ngal_fill = len(radecfill) #-= 1
    massfill = np.repeat(10, ngal_fill)#len(radecfill))
    #ra_dtfe = np.concatenate([galcat_mc[0, :, 1], radecfill[:, 0]])
    #dec_dtfe = np.concatenate([galcat_mc[0, :, 2], radecfill[:, 1]])
    #points_dtfe = np.c_[ra_dtfe,dec_dtfe]
    
    points_dtfe = np.concatenate([galcat_mc[0, :, 1:3][mask_mc_all],
                                  radecfill])
    #points_dtfe = np.concatenate([galcat_mc[0, :, 1:3],
    #                              radecfill])
    
    #densities = np.empty((Nmc,len(points_dtfe)))

    dtfemc = np.zeros(len(points_dtfe))
    for imc in range(Nmc):
        mask_mc_all_imc = mask_mc[imc][mask_mc_all]
        pmask_dtfe = np.concatenate([mask_mc_all_imc,np.repeat(True, ngal_fill)])
        if use_mass_density:
            mass_dtfe = np.concatenate([10**galcat_mc[imc,:,4][mask_mc_all][mask_mc_all_imc],10**massfill])
        else:
            mass_dtfe = np.ones(np.sum(pmask_dtfe))

        #pmask_dtfe = np.concatenate([mask_mc[imc],np.repeat(True, ngal_fill)])
        #mass_dtfe = np.concatenate([10**galcat_mc[imc,:,4][mask_mc[imc]],10**massfill])
        
        #print('ngal_fill', ngal_fill, len(pmask_dtfe), len(points_dtfe))

        #pmask_dtfe = mask_mc[imc][mask_mc_all]
        #mass_dtfe = 10**galcat_mc[imc,:,4][mask_mc[imc]] #[mask_mc_all]
        
        #print('npoints slice', len(points_dtfe),len(pmask_dtfe),np.sum(pmask_dtfe),len(mass_dtfe))
        #print('islice', islice, 'imc', imc, 'npoints', np.sum(pmask_dtfe))
        
        inputs_dtfe2d = [points_dtfe, pmask_dtfe, mass_dtfe, 1]
        #densities[imc] = dtfe2d(inputs_dtfe2d)
        #densities[imc] = dtfe2d(inputs_dtfe2d)
        dtfemc += np.log10(dtfe2d(inputs_dtfe2d))/Nmc
        
        
    #dtfemc = np.log10(np.nanmedian(densities, axis=0)) 
    #np.savez('dtfemc_'+str(islice)+'.npz', points_dtfe=points_dtfe, 
    #         pmask_dtfe = pmask_dtfe,
    #         mass_dtfe=mass_dtfe,
    #         dtfemc=dtfemc
    #        )
    #dtfemc = np.log10(np.nanmean(densities,axis=0))
    #dtfemc = np.nanmean(np.log10(densities),axis=0)
    #dtfemc = np.nansum(densities,axis=0) #/Nmc
    x_m = np.linspace(np.min(xminmax), np.max(xminmax), xsize)
    y_m = np.linspace(np.min(yminmax), np.max(yminmax), ysize)
    x_m, y_m = np.meshgrid(x_m, y_m)
    if len(points_dtfe[~np.isnan(dtfemc)]) > 3:
        tri = Delaunay(points_dtfe[~np.isnan(dtfemc)])
        grid_dtfemc = LinearNDInterpolator(tri, 
                                dtfemc[~np.isnan(dtfemc)])(x_m,y_m)
    else:
        grid_dtfemc = np.zeros((ysize, xsize))
    kernel = astropy.convolution.Tophat2DKernel(
        radius = 1.5 * 0.1 / ((physep_ang(zslice, pixdeg) * (1 + zslice))).value
    ) ##250 comoving kpc, correspoding to 100kpc proper at z=0.5
    
    grid_dtfemc = astropy.convolution.convolve(grid_dtfemc, kernel, 
                                           preserve_nan=True)

    mm = (masks & 
          hullmask & 
          np.logical_not(np.isnan(grid_dtfemc)))
    
    #&(grid_dtfemc != 0) )
    #print(grid_dtfemc[mm])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, _, sigma = astropy.stats.sigma_clipped_stats(
        grid_dtfemc[mm], sigma=3.0)

    meandens = 10 ** mu * np.exp(2.652 * sigma ** 2)
    grid_dtfemc = grid_dtfemc - np.log10(meandens)
    grid_dtfemc[np.where(np.isinf(grid_dtfemc))] = np.nan
    grid_dtfemc[~hullmask] = np.nan

    # print(np.where(np.logical_not(np.isnan(grid_dtfemc))))
    
    return grid_dtfemc



def get_dmap(detectifz):
    """Make 3D-DTFE overdensity map from 2D density maps in slices.

    Parameters
    ----------
    detectifz: intance of detectifz.DETECTIFz class
    Returns
    -------
    tuple (im3d_mass, masks, headmap) :
    im3d_mass : np.ndarray, 3D-DTFE of the field
    masks : np.array, masks in the same 2D projection as density maps
    headmap: astropy.io.fits `Header` object, header of the map 2D projection.
    """

    weight, headmasks = detectifz.data.masks.data, detectifz.data.masks.header
    
    imf = ( detectifz.config.rootdir+"/im3D." + detectifz.field + 
           "_Mlim"+str(np.round(detectifz.config.lgmass_lim,2))+".sigz68_z_" + 
           detectifz.config.avg + ".fits" )
    wf = ( detectifz.config.rootdir+"/weights2D." + detectifz.field + 
          "_Mlim"+str(np.round(detectifz.config.lgmass_lim,2))+".sigz68_z_" + 
          detectifz.config.avg + ".fits" )
    
    print(imf)
    print(wf)
    if Path(imf).is_file() and Path(wf).is_file():
        print("already saved, we read it")
        im3d_mass = fits.getdata(imf)
        masks = fits.getdata(wf).astype(bool)
        headmap = fits.getheader(wf)
    else:

        centre, zinf, zsup = detectifz.zslices.T

        ww = wcs.WCS(detectifz.data.masks.header)

        dlim = 0.0

        # first define grid limits and size
        xminmax = detectifz.data.xyminmax[:2]
        yminmax = detectifz.data.xyminmax[2:]
        xsize = (((xminmax[1] + dlim) - (xminmax[0] - dlim)) / detectifz.config.pixdeg).astype(int)
        ysize = (((yminmax[1] + dlim) - (yminmax[0] - dlim)) / detectifz.config.pixdeg).astype(int)

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [xsize//2, ysize//2]#[1, 1]
        w.wcs.cdelt = np.array([detectifz.config.pixdeg, detectifz.config.pixdeg])
        w.wcs.crval = [xminmax[0] - dlim, yminmax[0] - dlim]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        headmap = w.to_header()
        headmap.set("NAXIS", 2)
        headmap.set("NAXIS1", xsize)
        headmap.set("NAXIS2", ysize)
        
        hdu_weights = fits.PrimaryHDU(data=detectifz.data.masks.data.astype(int),
                                      header=detectifz.data.masks.header)
        
        # Cette projection transforme le bon masque en array NaN !!!
        masks = reproject_interp(hdu_weights, headmap, return_footprint=False)


        masks[np.where(masks > 0)] = 1
        masks[np.where(np.isnan(masks))] = 0

        print('-------------')
        print(np.sum(masks),flush=True) # Si cette somme est un entier (environ 100000), c'est bon
        print('-------------')

        masks = masks.astype(bool)
    
        if detectifz.config.use_mass_density:
            print('')
            print('COMPUTING STELLAR-MASS DENSITY')
            print('')
        else:
            print('')
            print('COMPUTING GALAXY DENSITY')
            print('')


        start = time.time()
             
        im3d_mass = np.array(Parallel(n_jobs=int(1 * detectifz.config.nprocs), max_nbytes=1e6)(
            delayed(get_dtfemc_nogrid)(
                islice, 
                detectifz.zslices, 
                detectifz.data.galcat_mc, 
                detectifz.maskMlim_mc, 
                detectifz.config.use_mass_density,
                detectifz.config.Nmc, 
                [xsize, ysize, xminmax, yminmax, detectifz.config.pixdeg], 
                masks, 
                headmap)
            for islice in range(len(detectifz.zslices))))

        end = time.time()
        print("total time DTFE (s)= " + str(end - start))

        # convolution with 1D normal of sigma=dzmap in redshift space (remove high freq variations)
        nans, x = nan_helper(im3d_mass)  # interpolate nans
        im3d_mass[nans] = np.interp(x(nans), x(~nans), im3d_mass[~nans])
        im3d_mass = gaussian_filter1d(im3d_mass, 1, axis=0)

        im3d_mass[nans] = np.nan  # put back nans to nan

        # write 3d image
        centre, zinf, zsup = detectifz.zslices[:, 0], detectifz.zslices[:, 1], detectifz.zslices[:, 2]
        w2d = wcs.WCS(headmap)
        hdu = fits.PrimaryHDU(im3d_mass)
        w3d = wcs.WCS(hdu.header)
        w3d.wcs.ctype = ["RA---TAN", "DEC--TAN", "z"]
        w3d.wcs.crval = [w2d.wcs.crval[0], w2d.wcs.crval[1], centre[0]]
        w3d.wcs.crpix = [1, 1, 1]
        w3d.wcs.cdelt = [w2d.wcs.cdelt[0], w2d.wcs.cdelt[1], 0.01]
        hdu = fits.PrimaryHDU(im3d_mass, header=w3d.to_header())
        hdu.writeto(imf, overwrite=True)
        hdu = fits.PrimaryHDU(masks.astype(int), header=headmap)
        hdu.writeto(wf, overwrite=True)

    return im3d_mass, masks, headmap
