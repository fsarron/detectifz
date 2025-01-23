import numpy as np
from astropy.table import Table, Column, vstack
from astropy import units
from astropy import wcs
from astropy.coordinates import SkyCoord
from photutils import SkyCircularAperture

from scipy.ndimage.filters import gaussian_filter1d
import regions

from .utils import angsep_radius, physep_ang, nan_helper, detectifz2radec


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list).
    dkamins' answer on stackoverflow.com/questions/7352684.
    This is fast enough for our purpose but could be speeded up using numpy.split

    Parameters
    ----------
    vals: list, values to be grouped
    step: int, default=1, expected difference between consecutive memebers of groups

    Returns
    -------
    result: list of list, each sublist is a group of consecutive elements
    """
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def cleaning(detectifz, det):
    """Link detections close to each other and keep only the highest ranked
        one in each group based on SN.
        
    Parameters     
    ----------
    detectifz: intance of detectifz.DETECTIFz class
    det: list, list of all detections as return by .detection.detection()


    Returns
    -------
    tuple (clus, detmul)
    clus: astropy.table.Table, Table of detected groups (cleaned). Sorted by redshift.
    detmul: list of list, list of detections linked to each group in the final catalogue.
            Not redshift sorted.
            
    Parameters of detectifz.DETECTIFz used in this function
    ----------
    SNmin: float, minimum signal-to-noise ratio of detection
    mdist: float, maximum projected distance in Mpc to group individual detections
    zslices: np.array of size:3xN, redshift limits of all slices
    """
    zc, zinf, zsup = detectifz.zslices.T
    # slice_idx = np.array([np.where(zc == det[i]['z'])[0][0] for i in range(len(det))])
    # det.add_column(Column(slice_idx,name='slice_idx'))

    clus = Table(names=det.colnames)

    indc = 0
    detmult = []
    while len(det) > 0:
        clus0 = Table(det[0], copy=True)

        C = SkyCoord(ra=clus0["ra"] * units.degree, dec=clus0["dec"] * units.degree)
        D = SkyCoord(ra=det["ra"] * units.degree, dec=det["dec"] * units.degree)
        sep = D.separation(C)

        if detectifz.config.objects_detected == 'groups':
            idm = np.where(sep < angsep_radius(clus0["z"], detectifz.config.dclean))[0] #in proper coordinates
        elif detectifz.config.objects_detected == 'protoclusters':
            mask_dclean = sep < angsep_radius(clus0["z"], detectifz.config.dclean / (1 + clus0["z"])) #in comoving coordinates.

            #bbox_center_sky = SkyCoord(ra=0.5*(clus0['bbox_ramax']+clus0['bbox_ramin']),
            #                   dec=0.5*(clus0['bbox_decmax']+clus0['bbox_decmin']),
            #                   unit='deg')  
        
            bbox_center_sky = SkyCoord(ra=clus0['ra'],
                               dec=clus0['dec'],
                               unit='deg')
            sky_region = regions.RectangleSkyRegion(center=bbox_center_sky[0], 
                                            width=0.75 * np.abs(clus0['bbox_ramax'][0]-clus0['bbox_ramin'][0]) * units.deg,
                                            height=0.75 * np.abs(clus0['bbox_decmax'][0]-clus0['bbox_decmin'][0]) * units.deg)
            mask_inbox = sky_region.contains(D, wcs=wcs.WCS(detectifz.head2d))
            mask_merge = mask_inbox + mask_dclean
            idm = np.where(mask_merge)[0]  

        
        idc = np.where(zc == clus0["z"])[0][0]
        idx = np.sort(det[idm]["slice_idx"])

        g = group_consecutives(np.unique(idx))
        tmp = [None]*len(g)
        for i in range(len(g)):
            tmp[i] = np.isin(idc, g[i])
        idg = np.where(tmp)[0][0]

        idg2idm = np.where(np.isin(det[idm]["slice_idx"], g[idg]))[0]
        idxclean = np.where(np.isin(det["idu"], det[idm][idg2idm]["idu"]))[0]


        if clus0["SN"] >= detectifz.config.SNmin:
            
            detmult.append(det[idxclean])

            
            clus0["zinf"] = np.min(det[idm][idg2idm]["zinf"])
            clus0["zsup"] = np.max(det[idm][idg2idm]["zsup"])
            clus0["slice_idx_inf"] = np.min(det[idm][idg2idm]["slice_idx"])
            clus0["slice_idx_sup"] = np.max(det[idm][idg2idm]["slice_idx"])
            # clus0.add_column(Column(indc,name='id'))
            clus = vstack([clus, clus0])
            indc += 1
        det.remove_rows(idxclean)

    clus.remove_column("idu")
    # clus.remove_column('log_dgal')
    clus.add_column(Column(np.arange(indc), name="id"))

    idx_sort = np.argsort(clus['z'])
    clus.sort("z")
    detmult = [detmult[x] for x in idx_sort]
    
    ### This is for protoclusters 
    ### -- NEED to put these parms in the config file ! 
    
    if detectifz.config.objects_detected == 'protoclusters':
    
        requiv = np.zeros(len(clus)) * units.Mpc

        for ic in range(len(clus)):
            xmin = np.min(detmult[ic]['bbox_ramin'])
            xmax = np.max(detmult[ic]['bbox_ramax'])
            ymin = np.min(detmult[ic]['bbox_decmin'])
            ymax = np.max(detmult[ic]['bbox_decmax'])

            requiv[ic] = np.sqrt(physep_ang(clus['z'][ic], (ymax-ymin)) * 
                                 physep_ang(clus['z'][ic], (xmax-xmin)) / np.pi) * (1 + clus['z'][ic])

            clus[ic]['bbox_ramin'] = xmin
            clus[ic]['bbox_ramax'] = xmax
            clus[ic]['bbox_decmin'] = ymin
            clus[ic]['bbox_decmax'] = ymax

        ndets = np.array([len(detmult[ic]) for ic in range(len(clus))])

        clus['rMpc_subdets_cMpc'] = requiv
        clus['ndets'] = ndets

        
    if detectifz.config.objects_detected == 'protoclusters':
        ### JWST Protoclusters r_equiv_min_cMpc = 0, n_subdets_min = 3
        ### Euclid Protoclusters tests : r_equiv_min_cMpc = 0.5, n_subdets_min = 3
        mask_proto_keep = ((requiv > detectifz.config.r_equiv_min_cMpc * units.Mpc ) & 
                           (ndets > detectifz.config.n_subdets_min))

        clus = clus[mask_proto_keep]
        detmult = [detmult[x] for x in mask_proto_keep]#detmult[mask_proto_keep]
        
    #TO DO revert to ra, dec when coord_change !
    #clus.rename_columns(['ra', 'dec'], ['ra_detectifz', 'dec_detectifz'])
    #ra_original, dec_original = detectifz2radec(detectifz.data.skycoords_center, 
    #                                            clus['ra_detectifz', 'dec_detectifz'].to_pandas().to_numpy().T)
    #clus.add_column(Column(ra_original, name='ra'))
    #clus.add_column(Column(dec_original, name='dec'))

    return clus, detmult


def clus_pdz_im3d(detectifz, smooth):
    """Compute the PDF(z) for each group.

    Parameters     
    ----------
    detectifz: intance of detectifz.DETECTIFz class
    
    
    Returns
    -------
    pdzclus: np.array Ngroups x len(zz), PDF(z) for each group
                sampled at zz. Normalized as np.sum(pdzclus[iclus] = 1).
                
                
    Parameters of detectifz.DETECTIFz used in this function
    ----------
    zz: np.array, array of redshift sampling
    im3d: np.ndarray, 3D-DTFE of the field
    maskim: np.array dtype:bool, mask image
    headmasks: astropy.io.fits `Header` object, mask header
    zslices: np.array of size:3xN, redshift limits of all slices
    clus: astropy.table.Table, Table of detected groups
    smooth: float, width of gaussian smoothing applied
            in redshift space to the PDF(z)
    """
    nclus = len(detectifz.clus)
    detectifz.im3d[:, ~detectifz.weights2d] = np.nan
    w2d = wcs.WCS(detectifz.head2d)

    pdzclus = np.zeros((nclus, len(detectifz.data.zz)))
    zz = detectifz.data.zz
    dz = zz[1]-zz[0]
    zzbin = np.linspace(zz[0]-dz/2, zz[-1]+dz/2, len(zz)+1)
    clussky = SkyCoord(detectifz.clus["ra"], detectifz.clus["dec"], unit="deg", frame="fk5")

    izi_im3d, izs_im3d = np.digitize(np.min(detectifz.zslices[:, 0]), zzbin) - 1, np.digitize(
        np.max(detectifz.zslices[:, 0]), zzbin
    )
    slice_idx_shift = np.digitize(np.min(detectifz.zslices[:, 0]), zzbin) - 1
    izi_cut, izs_cut = (detectifz.clus['slice_idx_inf'].astype(int)+slice_idx_shift, 
                        detectifz.clus['slice_idx_sup'].astype(int)+slice_idx_shift
                       )
    #izi_cut, izs_cut = (
    #    np.digitize(detectifz.clus["zinf"], zzbin) - 1,
    #    np.digitize(detectifz.clus["zsup"], zzbin) - 1,
    #)
    
    

    for indc in range(nclus):
        # print(indc)
        
        if detectifz.config.objects_detected == 'groups': 
            aper = SkyCircularAperture(
                clussky[indc], r=angsep_radius(detectifz.clus["z"][indc], 0.25)
            )
        elif detectifz.config.objects_detected == 'protoclusters':
            aper = SkyCircularAperture(
                clussky[indc], r=angsep_radius(detectifz.clus["z"][indc], 2 / (1 + detectifz.clus["z"][indc]))
            )##comoving
        
        aper_pix = aper.to_pixel(w2d)
        aper_masks = aper_pix.to_mask(method="center")
        aper_data = np.array(
            [aper_masks.multiply(detectifz.im3d[i], fill_value=np.nan) for i in range(len(detectifz.im3d))]
        )

        aper_data[:, ~aper_masks.data.astype(bool)] = np.nan
        
        #print('SHAPES', pdzclus[indc, izi_im3d:izs_im3d].shape, len(detectifz.im3d))
        
        pdzclus[indc, izi_im3d:izs_im3d] = (
            10 ** np.nanmean(aper_data, axis=(1, 2)) - 1
        )  # dgal

        pdzclus[indc, : izi_cut[indc] + 1] = 0
        pdzclus[indc, izs_cut[indc] :] = 0

    pdzclus[pdzclus < 0] = 0

    nans, x = nan_helper(pdzclus)  # interpolate nans
    pdzclus[nans] = np.interp(x(nans), x(~nans), pdzclus[~nans])
    pdzclus = gaussian_filter1d(pdzclus, smooth, axis=1)
    detectifz.clus.add_column(Column(np.max(pdzclus, axis=1), name='peak_ampl'))
    detectifz.clus.add_column(Column(detectifz.data.zz[np.argmax(pdzclus, axis=1)], name='z_peak'))

    pdzclus = pdzclus / np.sum(pdzclus, axis=1)[:, None]
    pdzclus[nans] = 0

    return pdzclus
