from astropy.io import fits
from astropy.table import Table, Column
import numpy as np

from astropy import wcs
from astropy import units

from matplotlib.colors import LogNorm

import regions

from scipy.ndimage.filters import gaussian_filter1d
from astropy.coordinates import SkyCoord
from photutils.aperture import (SkyRectangularAperture, 
                                SkyCircularAperture, 
                                SkyCircularAnnulus,
                                aperture_photometry)
from detectifz.utils import physep_ang, angsep_radius


from astropy.wcs.utils import proj_plane_pixel_area,proj_plane_pixel_scales
import scipy.stats

def gal_in_box(clus, galcat, head2d):
    
    nclus = len(clus)
    

    bbox_center_sky = SkyCoord(ra=0.5*(clus['bbox_ramax']+clus['bbox_ramin']),
                           dec=0.5*(clus['bbox_decmax']+clus['bbox_decmin']),
                           unit='deg')  
    galsky = SkyCoord(ra=galcat['ra'], dec=galcat['dec'], unit='deg')


    gal_in_clus = np.empty(nclus, dtype='object')
    mask_inclus = np.zeros((nclus, len(galcat)), dtype=bool)
    
    for indc in range(nclus):   
        sky_region = regions.RectangleSkyRegion(center=bbox_center_sky[indc], 
                                        width=np.abs(clus['bbox_ramax']-clus['bbox_ramin'])[indc] * units.deg,
                                        height=np.abs(clus['bbox_decmax']-clus['bbox_decmin'])[indc] * units.deg)
        
        
        mask_inclus[indc] = sky_region.contains(galsky, wcs=wcs.WCS(head2d))
        #mask_inclus[indc] = ((galcat['ra'] > clus['bbox_ramin'][indc]) &
        #            (galcat['ra'] < clus['bbox_ramax'][indc]) & 
        #            (galcat['dec'] > clus['bbox_decmin'][indc]) & 
        #            (galcat['dec'] < clus['bbox_decmax'][indc])
        #           )
        gal_in_clus[indc] = galcat[mask_inclus[indc]]
    
    return mask_inclus, gal_in_clus
    
    

## convolve p(z)

def get_Pcc(clus, zzclus, pdzclus, sigz_z):
    nclus = len(clus)
    Pcc = np.zeros((nclus,len(zzclus)))
    dz = zzclus[1]-zzclus[0]
    zzbin = np.linspace(zzclus[0]-dz/2., zzclus[-1]+dz/2., len(zzclus)+1)
    for indc in range(nclus):
        jc = np.digitize(clus['z'][indc],zzbin)-1
        sigz = np.copy(sigz_z[jc])
        pclus = np.copy(pdzclus[indc]) / np.trapz(pdzclus[indc], zzclus)
        dzpix68_z = (sigz / 0.01)#.astype(int)
        Pcc[indc] = gaussian_filter1d(pclus,dzpix68_z) #max(2,dzpix68_z)) 
    return Pcc

def advindexing_roll(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]    
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]
    return A[rows, column_indices]


def Ntot_pos_imR200(wimR200,w,galsky):
    ngal = len(galsky.ra)
    galpix = galsky.to_pixel(w)

    Ntot_pos = np.empty(ngal)
    for l in range(ngal):
        ii = int(galpix[1][l])
        if ii < 0:
            ii = 0
        if ii >= wimR200.shape[0]:
            ii = wimR200.shape[0]-1
        
        jj = int(galpix[0][l])
        if jj < 0:
            jj = 0
        if jj >= wimR200.shape[1]:
            jj = wimR200.shape[1]-1

        Ntot_pos[l] = 10**wimR200[ii,jj]

    return Ntot_pos

def field_weights_noclus(iz,clus,pTH,weights2d,head2d):
    w2d = wcs.WCS(head2d)
    idxclus = np.where(pTH[:,iz] > 0)[0]
    clussky = SkyCoord(clus['ra'],clus['dec'],unit='deg',frame='fk5')
    #aper = np.array([
    #    SkyCircularAperture(clussky[i],r=angsep_radius(clus['z'][i],
    #                                                             clus['rMpc_subdets_cMpc'][i] / 
    #                                                             (1 + clus['z'][i]))) for i in idxclus])
    aper = np.array([SkyRectangularAperture(positions= clussky[i], 
                              w=(clus['bbox_ramax']-clus['bbox_ramin'])[i] * units.deg,
                              h=(clus['bbox_decmax']-clus['bbox_decmin'])[i] * units.deg) for i in idxclus])
    aper_pix = [ap.to_pixel(w2d) for ap in aper]
    aper_masks = [apix.to_mask(method='center') for apix in aper_pix]
    wwe = np.copy(weights2d)
    for mask in aper_masks:
        slices_large, slices_small = mask.get_overlap_slices(wwe.shape)
        wwe[slices_large] *= (~((mask.data[slices_small]).astype(bool))) #.astype(int)
    weights2d = np.copy(weights2d)
    wwe = wwe * weights2d
    weights3d_noclus = wwe.astype(bool)
    area_F_z = np.sum(weights3d_noclus)*proj_plane_pixel_area(w2d)

    return weights3d_noclus, area_F_z


def N_from_map(iz,zslices,clus,pdzclus,zzclus,im3d,weights2d,head2d,sigz_z):
    w2d = wcs.WCS(head2d)
    
    
    maskz = ((iz > clus['slice_idx_inf']) & (iz <= clus['slice_idx_sup']) & 
        (clus['SN'] > 1))
    idxclus_clus = np.where(maskz)[0]
    nclus = len(clus)
    #idxclus_clus = np.arange(nclus)
    clussky = SkyCoord(clus['ra'],clus['dec'],unit='deg',frame='icrs')

    
    aper_clus = np.array([SkyRectangularAperture(positions= clussky[i], 
                          w=(clus['bbox_ramax']-clus['bbox_ramin'])[i] * units.deg,
                          h=(clus['bbox_decmax']-clus['bbox_decmin'])[i] * units.deg) for i in idxclus_clus])
    aper_clus_pix = [ap.to_pixel(w2d) for ap in aper_clus]
    aper_clus_masks = [apix.to_mask(method='center') for apix in aper_clus_pix]
    
    
    
    dz = zzclus[1]-zzclus[0]
    zzbin = np.linspace(zzclus[0]-dz/2., zzclus[-1]+dz/2., len(zzclus)+1)
    iz_pdz = np.digitize(zslices[iz, 0], zzbin) -1
    
    pTH = np.zeros((nclus,len(zzclus)))
    Pcc = np.zeros((nclus,len(zzclus)))
    for indc in range(nclus):
        jc = np.digitize(clus['z'][indc],zzbin)-1
        sigz = np.copy(sigz_z[jc])
        pclus = np.copy(pdzclus[indc]) / np.trapz(pdzclus[indc], zzclus)
        dzpix68_z = (sigz / 0.01)#.astype(int)
        Pcc[indc] = gaussian_filter1d(pclus,dzpix68_z) #max(2,dzpix68_z)) 
        rv = scipy.stats.rv_histogram((Pcc[indc],zzbin))
        zinf, zsup = rv.interval(0.95)
        izinf, izsup = np.digitize(zinf,zzbin)-1, np.digitize(zsup,zzbin)-1
        pTH[indc][izinf:izsup] = 1 
    
    wnoclus, area_F_z = field_weights_noclus(iz_pdz,clus,pTH,weights2d,head2d)
    Nbkg = np.nanmedian(10 ** im3d[iz][wnoclus])
    NtotR = np.zeros(nclus)
    ic = 0
    for indc in range(nclus):
        if indc in idxclus_clus: 
            NtotR[indc] = np.nanmedian(10**aper_clus_masks[ic].multiply(im3d[iz]))
            ic += 1
        else:
            NtotR[indc] = np.nan
    
    return NtotR, Nbkg, wnoclus, area_F_z

def Prior_map(zslices,clus,pdzclus,zzclus,im3d,weights2d,head2d,galcat,sigz_z):
    n_im3d = len(im3d)
    nclus = len(clus)
    NtotR = np.zeros((n_im3d, len(clus)))
    Nbkg = np.zeros(n_im3d)
    wnoclus = np.zeros((n_im3d, weights2d.shape[0], weights2d.shape[1]))
    area_F_z = np.zeros(n_im3d)
    for iz in range(n_im3d):
        NtotR[iz], Nbkg[iz], wnoclus[iz], area_F_z[iz] = N_from_map(iz,zslices,clus,pdzclus,zzclus,
                                                                    im3d,weights2d,head2d, sigz_z)
    
    mask_inclus, gal_in_clus = gal_in_box(clus, galcat, head2d)
    
    galsky = SkyCoord(ra=galcat['ra'], 
                      dec=galcat['dec'], unit='deg')
    Npos = np.empty((len(clus), n_im3d), dtype='object')

    for iclus in range(len(clus)): 
        Npos[iclus] = [Ntot_pos_imR200(im3d[iz],wcs.WCS(head2d),galsky[mask_inclus[iclus]]) for iz in range(n_im3d)]
    
    prior_z = np.empty((nclus, len(im3d)), dtype='object')
    for iclus in range(nclus):
        for iz in range(len(im3d)):
            nn_pos = np.array(list(Npos[iclus]))
            prior_z[iclus, iz] = np.minimum(np.maximum(1 - Nbkg[iz] / nn_pos[iz], 0), 0.999)
            prior_z[iclus, iz][np.isnan(prior_z[iclus, iz])] = 0

    
    return prior_z, Npos, NtotR, Nbkg, wnoclus



def get_pmem(detectifz, zzgal, pdzgal):
    
    clus, zzclus, pdzclus, galcat = (detectifz.clus,
                                     detectifz.data.zz,
                                     detectifz.pdzclus,
                                     detectifz.data.galcat)
    
    lgMM = np.linspace(5, 13, 161)
    dM = lgMM[1] - lgMM[0]
    MMbin = np.linspace(lgMM[0]-dM/2., lgMM[-1]+dM/2., len(lgMM)+1)


    
    mask_inclus, gal_in_clus = gal_in_box(clus, galcat, detectifz.head2d)
    
    prior_z, Npos, NtotR, Nbkg, wnoclus = Prior_map(detectifz.zslices,
                        clus,
                        pdzclus,
                        zzclus,
                        detectifz.im3d,
                        detectifz.weights2d,
                        detectifz.head2d,
                        galcat,
                        detectifz.data.sigs.sigz68_z)
    
    dz = zzclus[1]-zzclus[0]
    zzbin = np.linspace(zzclus[0]-dz/2., zzclus[-1]+dz/2., len(zzclus)+1)
    izinf, izsup = np.digitize(detectifz.zslices[[0, -1], 0], zzbin)-1
    
    #Pcc = get_Pcc(clus, zzclus, pdzclus, detectifz.data.sigs.sigz68_z)
    #Pcc_Mz = get_Pcc_Mz(clus, zzclus, pdzclus, detectifz.data.sigs.sigz68_Mz)

    nclus = len(clus)
    pclus = np.array([pdzclus[i] / np.trapz(pdzclus[i], zzclus) for i in range(len(clus))])
    
    prior_clus = np.empty(nclus, dtype='object')
    pconv_norm_z = np.empty(nclus, dtype='object')
    pconv_norm_Mz = np.empty(nclus, dtype='object')
    pmem_norm21_z = np.empty(nclus, dtype='object')
    pmem_norm21_Mz = np.empty(nclus, dtype='object')

    for iclus in range(nclus):
    
        pzg_interp = np.array([np.interp(zzclus, zzgal[mask_inclus[iclus]][ig], pdzgal[mask_inclus[iclus]][ig])
                       for ig in range(np.sum(mask_inclus[iclus]))])
        
        pzg_interp /= np.trapz(pzg_interp, zzclus)[:, None]
        
        Mass = galcat['Mass_median'][mask_inclus[iclus]]

        pprior = np.array(list(prior_z[iclus])).T
        

        igM = np.digitize(np.minimum(12,Mass),MMbin)-1
        jc = np.digitize(clus['z'][iclus],zzbin)-1
        sigz_Mz = detectifz.data.sigs.sigz68_Mz[jc,igM]
        dzpix68_Mz = sigz_Mz / 0.01 #.astype(int)

        Pcc_Mz = np.array([gaussian_filter1d(pclus[iclus],dzpix68_Mz[indg]) for indg in range(len(Mass))])
                
        pmem_raw_Mz = np.array([np.trapz( pprior[ig] * Pcc_Mz[ig, izinf:izsup+1] * pzg_interp[ig][izinf:izsup+1], 
                              zzclus[izinf:izsup+1]) for ig in range(np.sum(mask_inclus[iclus]))])
        
        sigz_z = detectifz.data.sigs.sigz68_z[jc]
        dzpix68_z = (sigz_z / 0.01)#.astype(int)
        Pcc_z = gaussian_filter1d(pclus[iclus],dzpix68_z)
        
        prior_clus[iclus] = np.trapz(pprior * Pcc_z[izinf:izsup+1],zzclus[izinf:izsup+1], axis=1)

        
        pmem_raw_z = np.array([np.trapz( pprior[ig] * Pcc_z[izinf:izsup+1] * pzg_interp[ig][izinf:izsup+1], 
                              zzclus[izinf:izsup+1]) for ig in range(np.sum(mask_inclus[iclus]))])
        
        Pcclus = np.repeat(gaussian_filter1d(pclus[iclus],1),
                   np.sum(mask_inclus[iclus])).reshape(len(zzclus),np.sum(mask_inclus[iclus])).T 
        zclus_Pccmax = zzclus[np.argmax(Pcclus,axis=1)]
        jcc = np.digitize(zclus_Pccmax,zzbin)-1
        shift = jcc - np.argmax(pzg_interp,axis=1)
        pnorm = np.trapz(advindexing_roll(pzg_interp,shift)*Pcclus, zzclus, axis=1)

        pmem_norm21_z[iclus] = np.minimum(pmem_raw_z / pnorm, 0.999)
        pmem_norm21_Mz[iclus] = np.minimum(pmem_raw_Mz / pnorm, 0.999)

        pconv_raw_z = np.array([np.trapz(Pcc_z[izinf:izsup+1] * pzg_interp[ig][izinf:izsup+1], 
                              zzclus[izinf:izsup+1],) for ig in range(np.sum(mask_inclus[iclus]))])
        pconv_norm_z[iclus] = np.minimum(pconv_raw_z / pnorm, 0.999)
        
        pconv_raw_Mz = np.array([np.trapz(Pcc_Mz[ig, izinf:izsup+1] * pzg_interp[ig][izinf:izsup+1], 
                              zzclus[izinf:izsup+1],) for ig in range(np.sum(mask_inclus[iclus]))])
        pconv_norm_Mz[iclus] = np.minimum(pconv_raw_Mz / pnorm, 0.999)
        
        
        #pmem_norm[iclus] = prior_clus[iclus] * pconv_raw / pnorm #* pconv_norm
    pmem_norm24 = prior_clus * pconv_norm_z

    return pmem_norm24, pmem_norm21_z, pmem_norm21_Mz, pconv_norm_z, pconv_norm_Mz, prior_clus, mask_inclus, prior_z, Npos, NtotR, Nbkg, wnoclus
        
        