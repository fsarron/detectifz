import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
#import ray
import psutil
import time

from joblib import Parallel, delayed

import astropy.stats
from astropy import wcs
from astropy import units
from astropy.table import Table, Column, vstack
from astropy.coordinates import SkyCoord

from photutils import CircularAperture, CircularAnnulus
from photutils import SkyCircularAnnulus, SkyCircularAperture
from photutils import SourceCatalog, detect_sources, deblend_sources

from .utils import angsep_radius, physep_ang

__all__ = ["det_photutils", "detection"]


#@ray.remote(max_calls=10)
def det_photutils(SNmin, l, centre_id, zinf_id, zsup_id, im3d_id, weights_id, head):
    """Detects regions above a given signal-to-noise in a 2D map.

    Parameters
    ----------
    SNmin: float, minimum signal-to-noise for detection
    l: int, index of slice
    centre_id: np.array, central redshift of slices
    zinf_id: np.array, lower redshift of slices
    zsup_id: np.array, upper redshift of slices
    im3d_id: np.ndarray, 3D DTFE array
    weights_id: np.array, 2D boolean mask array
    head: astropy.io.fits `Header` object, mask header

    Returns
    -------
    tuple (tab,pos)
    tab: numpy array of detection properties including sky positions, SN and size
    pos: numpy array of detections central pixel positions
    """
    # print(l)
    zslice = centre_id[l]
    zinf = zinf_id[l]
    zsup = zsup_id[l]
    log_dgal_s = np.copy(im3d_id[l])
    weights = np.copy(weights_id)

    # define min group area as disk of r=0.25Mpc in number of pixels
    w = wcs.WCS(head)
    rmin = angsep_radius(zslice, 0.25)
    min_area_deg = np.pi * rmin ** 2

    dpix = w.wcs.cdelt[0] * w.wcs.cdelt[1]
    min_area = min_area_deg.value / dpix
    max_area = np.pi * angsep_radius(zslice, 4).value / dpix

    # compute mean and dispersion of wlog_dgal
    # mud = np.nanmean(log_dgal_s[~weights])
    # sigmad = np.nanstd(log_dgal_s[~weights])
    mud, _, sigmad = astropy.stats.sigma_clipped_stats(
        log_dgal_s[weights], sigma=3
    )  # ,cenfunc=np.nanmedian,stdfunc=np.nanstd)
    dthresh = mud + SNmin * sigmad
    print(SNmin, "sigma detection =", dthresh, " density contrast")

    segm = detect_sources(log_dgal_s, dthresh, npixels=int(min_area))
    if len(np.unique(segm)) > 1:
        segm_deblend = deblend_sources(
            log_dgal_s, segm, npixels=int(min_area), nlevels=32, contrast=0.01
        )
        # SEG = segm_deblend.data.astype('float')
        # SEG[weights] = np.nan
        # SEG[np.where(np.isnan(log_dgal_s))] = np.nan

        cat = SourceCatalog(log_dgal_s, segm_deblend, wcs=w)
        det_table = cat.to_table()
        det_table["id"] = np.arange(len(det_table))
        #print(det_table.colnames)
        #det_table.write('det_table_slice'+str(l)+'.fits')
        det_table["maxval_xindex"] = cat.maxval_xindex
        det_table["maxval_yindex"] = cat.maxval_yindex

        det_table["sky_max"] = SkyCoord(
            w.wcs_pix2world(
                np.c_[cat.maxval_xindex,
                      cat.maxval_yindex]
                * units.pix,
                1,
            ),
            unit=units.deg,
        )

        xy = (
            np.c_[cat.maxval_xindex,
                  cat.maxval_yindex]
            * units.pix
        )
        rr = cat.equivalent_radius.value
        #print(rr)

        log_dgal_s[~weights] = np.nan

        apertures = [CircularAperture(xy[i].value, r=rr[i]) for i in range(len(xy))]
        mask = [aper.to_mask(method="center") for aper in apertures]
        mask_data = [m.multiply(log_dgal_s, fill_value=np.nan) for m in mask]

        rminpix = rmin / np.sqrt(dpix) / units.deg
        apertures250 = [
            CircularAperture(xy[i].value, r=rminpix.value) for i in range(len(xy))
        ]
        mask250 = [aper.to_mask(method="center") for aper in apertures250]
        mask_data250 = [m.multiply(log_dgal_s, fill_value=np.nan)
                        for m in mask250]

        r500kpc = angsep_radius(zslice, 0.5)
        r500kpcpix = r500kpc / np.sqrt(dpix) / units.deg
        apertures500kpc = [
            CircularAperture(xy[i].value, r=r500kpcpix.value) for i in range(len(xy))
        ]
        mask500kpc = [aper.to_mask(method="center")
                      for aper in apertures500kpc]
        mask_data500kpc = [
            m.multiply(log_dgal_s, fill_value=np.nan) for m in mask500kpc
        ]

        r1Mpc = angsep_radius(zslice, 1)
        r1Mpcpix = r1Mpc / np.sqrt(dpix) / units.deg
        apertures1Mpc = [
            CircularAperture(xy[i].value, r=r1Mpcpix.value) for i in range(len(xy))
        ]
        mask1Mpc = [aper.to_mask(method="center") for aper in apertures1Mpc]
        mask_data1Mpc = [m.multiply(log_dgal_s, fill_value=np.nan)
                         for m in mask1Mpc]

        galnum = np.array(
            [
                np.mean(
                    mask_data[i][
                        np.where(
                            (mask_data[i] != 0) & (
                                np.isnan(mask_data[i]) == False)
                        )
                    ]
                )
                for i in range(len(mask_data))
            ]
        )
        unmasked_area_rdet = (
            np.array(
                [
                    np.where((mskdat != 0) & (np.isnan(mskdat) == False))[
                        0].shape[0]
                    for mskdat in mask_data
                ]
            )
            * wcs.utils.proj_plane_pixel_area(w)
            * units.deg
            * units.deg
        )

        unmasked_area_rdet250 = (
            np.array(
                [
                    np.where((mskdat != 0) & (np.isnan(mskdat) == False))[
                        0].shape[0]
                    for mskdat in mask_data250
                ]
            )
            * wcs.utils.proj_plane_pixel_area(w)
            * units.deg
            * units.deg
        )
        SN250 = np.array(
            [
                (
                    np.mean(
                        mskdat[np.where((mskdat != 0) & (
                            np.isnan(mskdat) == False))]
                    )
                    - mud
                )
                / sigmad
                for mskdat in mask_data250
            ]
        )
        idx_nan = np.where(np.isnan(SN250))
        SN250[idx_nan] = 0.0

        unmasked_area_r500kpc = (
            np.array(
                [
                    np.where((mskdat != 0) & (np.isnan(mskdat) == False))[
                        0].shape[0]
                    for mskdat in mask_data500kpc
                ]
            )
            * wcs.utils.proj_plane_pixel_area(w)
            * units.deg
            * units.deg
        )
        unmasked_area_r1Mpc = (
            np.array(
                [
                    np.where((mskdat != 0) & (np.isnan(mskdat) == False))[
                        0].shape[0]
                    for mskdat in mask_data1Mpc
                ]
            )
            * wcs.utils.proj_plane_pixel_area(w)
            * units.deg
            * units.deg
        )

        era = np.zeros(len(galnum))
        era[:] = angsep_radius(zslice, 0.5)
        edec = era

        zz = np.zeros(len(galnum))
        zz[:] = zslice
        zzi = np.zeros(len(galnum))
        zzi[:] = zinf
        zzs = np.zeros(len(galnum))
        zzs[:] = zsup

        rdet_sky = rr * w.wcs.cdelt[0] * units.deg / units.pix
        rdet_Mpc = physep_ang(zslice, rdet_sky.value)

        tab = np.c_[
            np.repeat(l, len(det_table)),
            det_table["id"],
            det_table["sky_max"].ra.value,
            era,
            det_table["sky_max"].dec.value,
            edec,
            zz,
            zzi,
            zzs,
            galnum,
            SN250,
            rdet_sky,
            rdet_Mpc,
            unmasked_area_rdet250.value,
            unmasked_area_r500kpc.value,
            unmasked_area_r1Mpc.value,
            unmasked_area_rdet.value,
            det_table["max_value"],
            det_table['segment_flux'],
        ]
        tab = tab[np.where(rr > 0.01)]

        pos = np.c_[
            np.repeat(l, len(det_table)),
            det_table["id"],
            det_table["maxval_xindex"].value,
            det_table["maxval_yindex"].value,
        ]
        pos = pos[np.where(rr > 0.01)]

    else:
        tab = np.array([])
        pos = np.array([])

    return tab, pos


def detection(detectifz):
    
    """Run detection in all slices.
    Parameters
    ----------
    detectifz: intance of detectifz.DETECTIFz class
    
    Returns
    -------
    tuple (det, det_tab, pos) :
    det : list, list of np.array of detections with properties in each slice
    det_tab : astropy.table, Table of all detections in all slices concatenated
    pos: list, list of np.array of detections with position in
               pixel coordinates in each slice
               
    Parameters of detectifz.DETECTIFz used in this function
    ----------
    im3d,weights,head2D,zslices,SNmin,nprocs

    im3d: np.ndarray, 3D-DTFE of the field
    weights: np.array dtype:bool, mask image
    head2D: astropy.io.fits `Header` object, mask header
    zslices: np.array of size:3xN, redshift limits of all slices
    SNmin: float, signal-to-noise ratio for detection, used in file naming convention
    nprocs: int, number of processes for parallelization with ray        
    """

    
    memo = 1.8*1024**3 #1.5 * (im3d.nbytes + weights.nbytes)
    mem_avail = 10*1024**3 #psutil.virtual_memory().available
    '''
    if memo < 0.9 * mem_avail:
        memo_obj = int(0.9 * memo)
        #memo_heap = memo - memo_obj
        ray.init(
            num_cpus=detectifz.config.nprocs,
            object_store_memory=memo_obj,
            ignore_reinit_error=True,
            log_to_driver=False,
        )
        im3d_id = ray.put(detectifz.im3d)
        weights_id = ray.put(detectifz.weights2d)
        centre, zinf, zsup = detectifz.zslices.T 
        centre_id = ray.put(centre)
        zinf_id = ray.put(zinf)
        zsup_id = ray.put(zsup)

        detect_all = ray.get(
            [
                det_photutils.remote(
                    detectifz.config.SNmin, l, centre_id, zinf_id, zsup_id, im3d_id, weights_id, detectifz.head2d
                )
                for l in range(len(centre))
            ]
        )
        ray.shutdown()
    else:
        raise ValueError("Not enough memory available : ",
                         memo, "<", mem_avail)
    '''
    centre, zinf, zsup = detectifz.zslices.T 

    detect_all = np.array(Parallel(n_jobs=int(1 * detectifz.config.nprocs), max_nbytes=1e6)(
        delayed(det_photutils)(detectifz.config.SNmin, 
                               l, 
                                centre,
                                zinf, 
                                zsup, 
                                detectifz.im3d, 
                                detectifz.weights2d, 
                                detectifz.head2d
                                )
                for l in range(len(centre))))


    # remove slices with no det from the list
    ddet = []
    det = []
    # segm_dgal0 = []
    pos = []
    for i in range(len(detect_all)):
        pos.append(detect_all[i][1])
        # segm_dgal0.append(detect_all[i][1])
        det.append(detect_all[i][0])
        if len(detect_all[i][0]) > 0:
            ddet.append(detect_all[i][0])

    det_tab = Table(np.concatenate(ddet))
    new_names = [
        "slice_idx",
        "id_det",
        "ra",
        "era",
        "dec",
        "edec",
        "z",
        "zinf",
        "zsup",
        "log_dgal",
        "SN",
        "rsky",
        "rMpc",
        "area",
        "area_r500kpc",
        "area_r1Mpc",
        "area_rdet",
        "max_value",
        "segment_flux"
    ]
    for i, n in enumerate(det_tab.colnames):
        det_tab.rename_column(n, new_names[i])
    det_tab.add_column(Column(range(len(det_tab)), name="idu"))
    det_tab.sort("SN")
    det_tab.reverse()

    return det, det_tab, pos
