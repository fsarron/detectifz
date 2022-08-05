import os
import sys
import numpy as np
#import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.cosmology
from astropy.table import Table,Column,vstack
from astropy.io import fits
from photutils import SkyCircularAnnulus,SkyCircularAperture
from photutils import aperture_photometry
from astropy.wcs.utils import proj_plane_pixel_area
from astropy import wcs
from astropy.stats import sigma_clipped_stats
from scipy.interpolate import CubicSpline
from pathlib import Path
from scipy.stats import norm,lognorm
#import img_scale
from matplotlib.patches import Circle
from scipy.signal import find_peaks, peak_prominences
#from aplpy import make_rgb_cube
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
import scipy.stats
import time
import ray
import pandas as pd

import warnings
from astropy.utils.exceptions import AstropyWarning,AstropyUserWarning

#sys.setrecursionlimit(1000000)


def angsep_radius(z,radius): 
    cosmo=astropy.cosmology.FlatLambdaCDM(H0=70.,Om0=0.3) 
    return (((radius*1000)*u.kpc) * cosmo.arcsec_per_kpc_proper(z)).to(u.deg)

def physep_ang(z,radius):  
    cosmo=astropy.cosmology.FlatLambdaCDM(H0=70.,Om0=0.3)  
    return (((radius)*u.deg) / cosmo.arcsec_per_kpc_proper(z)).to(u.Mpc) 

def Ntotclus_mz(masks_im,w,pdzr,pdmr,Nm,Nz,raclus,decclus,rsup):
    skycoords = SkyCoord(raclus, decclus, unit='deg') 
    aper = SkyCircularAperture(skycoords, r=rsup)
    aper_pix = aper.to_pixel(w) 
    aper_masks = aper_pix.to_mask(method='center') 
    aper_data = aper_masks.multiply(masks_im)
    mask1 = aper_masks.data
    aper_data_1d = aper_data[mask1> 0]
    Npix_nm = np.where(aper_data_1d > 0)[0].shape[0]
    A_apertures_nm = Npix_nm * proj_plane_pixel_area(w) * u.deg * u.deg
    Ntotclus_mz = np.array([[(1./A_apertures_nm.value)*np.sum(pdzr[:,i]*pdmr[:,j]) for i in range(Nz)] for j in range(Nm)])

    return Ntotclus_mz

def mN(N_mz,ig,jc,dm,dz,Nm,Nz):
    if dz == 0:
        mN = np.nanmean(N_mz[max(ig-dm,0):min(ig+dm,Nm-1),jc].value)
    else:
        mN = np.nanmean(N_mz[max(ig-dm,0):min(ig+dm,Nm-1),max(jc-dz,0):min(jc+dz,Nz-1)].value)
    return mN

def mN_infsup(N_mz,ig,jc,dm,dz,Nm,Nz):
    izinf,izsup = dz
    if izsup == izinf:
        mN = np.nanmean(N_mz[max(ig-dm,0):min(ig+dm,Nm-1),jc].value)
    else:
        mN = np.nanmean(N_mz[max(ig-dm,0):min(ig+dm,Nm-1),max(izinf,0):min(izsup,Nz-1)].value)
    return mN


def mN_z(N_mz,ig,dm,Nm):
    mN_z = np.nanmean(N_mz[max(ig-dm,0):min(ig+dm,Nm-1),:].value,axis=0)
    return mN_z

def floc(masks_im,w,Nf_zm,pdzr,pdmr,Nm,Nz,raclus,decclus,rinf,rsup,minf,msup,dzpix,jc):
    skycoords = SkyCoord(raclus, decclus, unit='deg')
    annulus_aperture = SkyCircularAnnulus(skycoords, r_in=rinf, r_out=rsup)
    annulus_aperture_pix = annulus_aperture.to_pixel(w)  
    annulus_masks = annulus_aperture_pix.to_mask(method='center')
    annulus_data = annulus_masks.multiply(masks_im)
    mask1 = annulus_masks.data  
    annulus_data_1d = annulus_data[mask1 > 0]
    Npix_nm = np.where(annulus_data_1d > 0)[0].shape[0]
    A_annulus_nm = Npix_nm * proj_plane_pixel_area(w) * u.deg * u.deg

    Nfield_loc_mz = (1./A_annulus_nm) * np.asarray([[np.sum(pdzr[:,i]*pdmr[:,j]) for i in range(Nz)] for j in range(Nm)])

    mNbkg_loc = np.nanmean(np.hstack(np.asarray([Nfield_loc_mz[minf:msup,max(jc-dzpix[i,jc],0):min(jc+dzpix[i,jc],Nz-1)].value.flatten() if dzpix[i,jc] != 0 else Nfield_loc_mz[minf:msup,jc].value for i in range(Nm)])))

    mNbkg_glob = np.nanmean(np.hstack(np.asarray([Nf_zm[minf:msup,max(jc-dzpix[i,jc],0):min(jc+dzpix[i,jc],Nz-1)].value.flatten() if dzpix[i,jc] != 0 else Nf_zm[minf:msup,jc].value for i in range(Nm)])))

    f = mNbkg_loc / mNbkg_glob

    return f,mNbkg_loc

def mN_tot(N_mzr,Nr,Nm,Nz,minf,msup,dzpix,jc):
    mN = np.array([np.nanmean(np.hstack(np.array([N_mzr[k,minf:msup,max(jc-dzpix[i,jc],0):min(jc+dzpix[i,jc],Nz-1)].value.flatten() if dzpix[i,jc] != 0 else N_mzr[k,minf:msup,jc].value for i in range(Nm)]))) for k in range(Nr)])
    return mN


def fgeo(Npos,Nclus_r,rpix):
    fgeo = 10**(Npos - Nclus_r[rpix])
    return fgeo

def fgeo_Ntot(dens_im,w,raclus,decclus,rinf,rsup,Nr):
    skycoords = SkyCoord(raclus, decclus, unit='deg') 
    apertures = [SkyCircularAnnulus(skycoords, r_in=rinf[i], r_out=rsup[i]) if rinf[i] > 0 else SkyCircularAperture(skycoords, r=rsup[i]) for i in range(Nr)]
    apertures_pix = [aper.to_pixel(w) for aper in apertures]
    apertures_masks = [aper_pix.to_mask(method='center') for aper_pix in apertures_pix]
    apertures_data = [aper_masks.multiply(dens_im) for aper_masks in apertures_masks] 

    Ntot_r = np.array([sigma_clipped_stats(aper_data[np.where(aper_data != 0)],sigma=3)[0] for aper_data in apertures_data])

    return Ntot_r


def Nloc_oneclus(im,w2d,raclus,decclus,rinf,rsup):
    skycoords = SkyCoord(raclus, decclus, unit='deg') 
    apertures = SkyCircularAnnulus(skycoords, r_in=rinf, r_out=rsup) if rinf > 0 else SkyCircularAperture(skycoords, r=rsup)
    apertures_pix = apertures.to_pixel(w2d)
    apertures_masks = apertures_pix.to_mask(method='center')
    apertures_data = apertures_masks.multiply(im)

    Ntot_r = sigma_clipped_stats(apertures_data[np.where(apertures_data != 0)],sigma=3)[0]
    
    return Ntot_r


def log_dgal_r(dens_im,w,raclus,decclus,rsup):
    skycoords = SkyCoord(raclus, decclus, unit='deg') 
    aper = SkyCircularAperture(skycoords, r=rsup)
    aper_pix = aper.to_pixel(w)
    aper_masks = aper_pix.to_mask(method='center')
    aper_data = aper_masks.multiply(dens_im)
    
    log_dgal_r = np.nanmean(aper_data[np.where(aper_data != 0)])

    return log_dgal_r


def unmasked_area_r(weights,w,raclus,decclus,rsup):
    
    skycoords = SkyCoord(raclus, decclus, unit='deg') 
    aper = SkyCircularAperture(skycoords, r=rsup)
    aper_pix = aper.to_pixel(w)
    aper_masks = aper_pix.to_mask(method='center')
    
    aper_data_masks = aper_masks.multiply(weights)
    
    unmasked_area = np.sum(aper_data_masks) * proj_plane_pixel_area(w)

    return unmasked_area

def dgal_center(dens_im,w,clus,indc,pos_slices):
    
    dgal = dens_im[pos_slices[clus[indc]['slice_idx'].astype(int)][(clus[indc]['id_det']-1).astype(int),3].astype(int), pos_slices[clus[indc]['slice_idx'].astype(int)][(clus[indc]['id_det']-1).astype(int),2].astype(int)]
        
    return dgal



def Ntot_pos(wim3d,w,galsky):
    ngal = len(galsky.ra)
    galpix = galsky.to_pixel(w)

    Ntot_pos = np.empty(ngal,dtype='object')
    for l in range(ngal):
        ii = int(galpix[1][l])
        if ii < 0:
            ii = 0
        if ii >= wim3d.shape[1]:
            ii = wim3d.shape[1]-1
        
        jj = int(galpix[0][l])
        if jj < 0:
            jj = 0
        if jj >= wim3d.shape[2]:
            jj = wim3d.shape[2]-1

        Ntot_pos[l] = wim3d[:,ii,jj]

    return Ntot_pos



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

        Ntot_pos[l] = wimR200[ii,jj]

    return Ntot_pos



def Ntot_pos_mean(wim2d,w2d,weights,galsky,zslice,radius_Mpc):
    radius_deg = angsep_radius(zslice,radius_Mpc)
    apertures_sky = SkyCircularAperture(galsky, r=radius_deg)
    phot_table = aperture_photometry(wim2d, apertures_sky,mask=weights,wcs=w2d) 
    num_table = aperture_photometry(np.logical_not(weights).astype(int), apertures_sky,wcs=w2d)
    Ntot_pos = np.array(phot_table['aperture_sum'] / num_table['aperture_sum'])

    return Ntot_pos


    

def zlim_pdzclus_gal(Pcc,conflim):
    if conflim=='95':
        cl = 0.9545
    zzbin = np.linspace(0.005,5.005,501)
    zlims = np.array([scipy.stats.rv_histogram((Pcc[indg],zzbin)).interval(cl) for indg in range(len(Pcc))])
    zlims[:,0] = np.maximum(zlims[:,0],0.01)
    zlims[:,1] = np.minimum(zlims[:,1],5)
    return zlims


@ray.remote
def membership_oneclus_mass_pdzlim(indc, gal_id, gal_colnames, mem_info, pdz, zz, pdM, MM, pdzclus, masks_im, headmasks, dzpix_68, Nfield):
    if not sys.warnoptions:
        warnings.simplefilter('ignore', category=AstropyWarning)
        warnings.simplefilter('ignore', category=AstropyUserWarning)
        warnings.simplefilter("ignore")
        
    if not indc % 50:    
        print('clus ',indc)
        
    wmasks = wcs.WCS(headmasks)

    Nfield_glob_Mz = Nfield / (u.deg*u.deg)

    Nz = len(zz)
    NM = len(MM)
        
    mmem_info = np.copy(mem_info[indc])
    
    ## extract info from mem_info
    raclus = mmem_info[1]
    decclus = mmem_info[2]
    zclus = mmem_info[3] #zpeak (NOT zMST)
    r200Mpc = mmem_info[4]
    rmaxMpc = mmem_info[5]
    rmax = angsep_radius(zclus,rmaxMpc)
    r200 = angsep_radius(zclus,r200Mpc)
    clus_imR200_r200 = mmem_info[6]
    ind_gal2Mpc = mmem_info[7]
    ind_galR200 = mmem_info[8]
    gal_imR200 = mmem_info[9]
    f_loc_DETECTIFz = mmem_info[10]
    f_profile_imR200 = 10**(gal_imR200-clus_imR200_r200)


    gal = Table(gal_id,names=gal_colnames)
    
    ###Ntotclus_mzr (so true for a given cluster whatever galaxy mag,z,rgc)
    #####      
    pdzr = pdz[ind_galR200]
    pdMr = pdM[ind_galR200]
    N_totclus_Mzr200 = Ntotclus_mz(masks_im,wmasks,pdzr,pdMr,NM,Nz,raclus,decclus,r200) / (u.deg*u.deg)
    
    
    
    #### Now work for each galaxy (of given mag)
    Pmem = []

    gal_2Mpc =  gal[ind_gal2Mpc]
    ngal = len(gal_2Mpc)
    pdz_2Mpc = pdz[ind_gal2Mpc]
    
    cluscat = SkyCoord(ra=raclus*u.degree, dec=decclus*u.degree, frame='fk5')
    galcat_2Mpc = SkyCoord(ra=gal_2Mpc['ra']*u.degree, dec=gal_2Mpc['dec']*u.degree, frame='fk5')
    rgc = cluscat.separation(galcat_2Mpc)
    rgcMpc = physep_ang(zclus,rgc.value)
    Mgal = np.array(gal_2Mpc['Bestfit_Mass'])
    

    zzbin = np.linspace(0.005,5.005,501)
    jc = np.digitize(zclus,zzbin)-1
    MMbin = np.linspace(3.95,14.15,103)
    ig = np.digitize(Mgal,MMbin)-1
    dM = 5
    
    
    win = np.array([signal.gaussian(M=501,std=max(2,dzpix_68[ig[indg],jc])) for indg in range(ngal)]) 
    Pcc = np.array([signal.convolve(pdzclus[indc],win[indg], mode='same')/sum(win[indg]) for indg in range(ngal)]) 
    conflim='95'
    zlims = zlim_pdzclus_gal(Pcc,conflim)
    izlims = np.digitize(zlims,zzbin)-1
    #dz = np.maximum(2,dzpix[ig,jc])

    ###mNlocbkg
    ###########
    mNbkg_glob = np.asarray([mN_infsup(Nfield_glob_Mz,ig[indg],jc,dM,izlims[indg],NM,Nz) for indg in range(ngal)])

    ###mNtotclus
    ##################
    ##corretion to take into account cluster geometry
    mNtotclus = np.asarray([mN_infsup(N_totclus_Mzr200,ig[indg],jc,dM,izlims[indg],NM,Nz) for indg in range(ngal)])

    
    ###Need to renormlaize before doing that
    dgal_DETECTIFz = ((mNtotclus * f_profile_imR200) / (f_loc_DETECTIFz * mNbkg_glob)) - 1
    ind_dgal = np.where(dgal_DETECTIFz > 0)[0]
    #cf Cucciati et al. (2018), even though the 2.652 factor is not explained - this is something related exp or ln(10)
    rv_lognorm = lognorm(s=1,loc=0.0,scale=np.sqrt(2.652))
    
    #f_beta = 1 - beta in CB16 formalism. But computed using dgal and the proba of a lognormal distribution for galaxu overdensity.
    fbeta_DETECTIFz_lognorm = rv_lognorm.cdf(dgal_DETECTIFz) 
    fbeta_DETECTIFz_Nexcess = dgal_DETECTIFz/(1.+dgal_DETECTIFz)
    
    Pcc = Pcc[ind_dgal] 
    pdzshift = np.asarray([np.roll(pdz_2Mpc[indg],(jc-np.argmax(pdz_2Mpc[indg]))) for indg in ind_dgal])
    pnorm = np.asarray([np.sum(Pcc[i_dgal]*pdzshift[i_dgal]) if  np.sum(Pcc[i_dgal]*pdzshift[i_dgal]) > np.sum(Pcc[i_dgal]*pdz_2Mpc[indg]) else np.sum(Pcc[i_dgal]*pdz_2Mpc[indg]) for i_dgal,indg in enumerate(ind_dgal)])
    
    Pmem = np.array([(fbeta_DETECTIFz_Nexcess[indg]) * np.sum(pdz_2Mpc[indg]*Pcc[i_dgal]) for i_dgal,indg in enumerate(ind_dgal)])
    Pmem = Pmem / pnorm
    
    Pmem_lognorm = np.array([(fbeta_DETECTIFz_lognorm[indg]) * np.sum(pdz_2Mpc[indg]*Pcc[i_dgal]) for i_dgal,indg in enumerate(ind_dgal)])
    Pmem_lognorm = Pmem_lognorm / pnorm
    
    
    memcat=pd.DataFrame()
    
    memcat['id'] = gal_2Mpc[ind_dgal]['id'][np.argsort(Pmem)]
    memcat['ra'] = gal_2Mpc[ind_dgal]['ra'][np.argsort(Pmem)] 
    memcat['dec']= gal_2Mpc[ind_dgal]['dec'][np.argsort(Pmem)] 
    memcat['z'] = gal_2Mpc[ind_dgal]['z'][np.argsort(Pmem)]
    memcat['mK'] = gal_2Mpc[ind_dgal]['mK'][np.argsort(Pmem)]
    memcat['rsky'] = rgc[ind_dgal][np.argsort(Pmem)]
    memcat['rMpc'] = rgcMpc[ind_dgal][np.argsort(Pmem)]
    memcat['Bestfit_Mass'] = gal_2Mpc[ind_dgal]['Bestfit_Mass'][np.argsort(Pmem)]
#    memcat['SFR'] = gal_2Mpc[ind_dgal]['SFR'][np.argsort(Pmem)]
    memcat['pDETECTIFz_Nexcess'] = np.minimum(np.round(Pmem[np.argsort(Pmem)],5),0.99999)
    memcat['pDETECTIFz_lognorm'] = np.minimum(np.round(Pmem_lognorm[np.argsort(Pmem)],5),0.99999)
    memcat['fcorr_profile']  = np.repeat(1,len(ind_dgal))

    

    return memcat



@ray.remote
def membership_oneclus_mass_sigzlim(indc, gal_id, gal_colnames, mem_info, pdz, zz, pdM, MM, pdzclus, masks_im, headmasks, dzpix_68, dzpix, Nfield):
    if not sys.warnoptions:
        warnings.simplefilter('ignore', category=AstropyWarning)
        warnings.simplefilter('ignore', category=AstropyUserWarning)
        warnings.simplefilter("ignore")
        
    if not indc % 50:    
        print('clus ',indc)
        
    wmasks = wcs.WCS(headmasks)

    Nfield_glob_Mz = Nfield / (u.deg*u.deg)

    Nz = len(zz)
    NM = len(MM)
        
    mmem_info = np.copy(mem_info[indc])
    
    ## extract info from mem_info
    raclus = mmem_info[1]
    decclus = mmem_info[2]
    zclus = mmem_info[3] #zpeak (NOT zMST)
    r200Mpc = mmem_info[4]
    rmaxMpc = mmem_info[5]
    rmax = angsep_radius(zclus,rmaxMpc)
    r200 = angsep_radius(zclus,r200Mpc)
    clus_imR200_r200 = mmem_info[6]
    ind_gal2Mpc = mmem_info[7]
    ind_galR200 = mmem_info[8]
    gal_imR200 = mmem_info[9]
    f_loc_DETECTIFz = mmem_info[10]
    f_profile_imR200 = 10**(gal_imR200-clus_imR200_r200)


    gal = Table(gal_id,names=gal_colnames)
    
    ###Ntotclus_mzr (so true for a given cluster whatever galaxy mag,z,rgc)
    #####      
    pdzr = pdz[ind_galR200]
    pdMr = pdM[ind_galR200]
    N_totclus_Mzr200 = Ntotclus_mz(masks_im,wmasks,pdzr,pdMr,NM,Nz,raclus,decclus,r200) / (u.deg*u.deg)
    
    
    
    #### Now work for each galaxy (of given mag)
    Pmem = []

    gal_2Mpc =  gal[ind_gal2Mpc]
    ngal = len(gal_2Mpc)
    pdz_2Mpc = pdz[ind_gal2Mpc]
    
    cluscat = SkyCoord(ra=raclus*u.degree, dec=decclus*u.degree, frame='fk5')
    galcat_2Mpc = SkyCoord(ra=gal_2Mpc['ra']*u.degree, dec=gal_2Mpc['dec']*u.degree, frame='fk5')
    rgc = cluscat.separation(galcat_2Mpc)
    rgcMpc = physep_ang(zclus,rgc.value)
    Mgal = np.array(gal_2Mpc['Bestfit_Mass'])
    

    zzbin = np.linspace(0.005,5.005,501)
    jc = np.digitize(zclus,zzbin)-1
    MMbin = np.linspace(3.95,14.15,103)
    ig = np.digitize(Mgal,MMbin)-1
    dM = 5
    
    
    win = np.array([signal.gaussian(M=501,std=max(2,dzpix_68[ig[indg],jc])) for indg in range(ngal)]) 
    Pcc = np.array([signal.convolve(pdzclus[indc],win[indg], mode='same')/sum(win[indg]) for indg in range(ngal)]) 
    conflim='95'
    #zlims = zlim_pdzclus_gal(Pcc,conflim)
    #izlims = np.digitize(zlims,zzbin)-1
    dz = np.maximum(2,dzpix[ig,jc])

    ###mNlocbkg
    ###########
    mNbkg_glob = np.asarray([mN(Nfield_glob_Mz,ig[indg],jc,dM,dz[indg],NM,Nz) for indg in range(ngal)])

    ###mNtotclus
    ##################
    ##corretion to take into account cluster geometry
    mNtotclus = np.asarray([mN(N_totclus_Mzr200,ig[indg],jc,dM,dz[indg],NM,Nz) for indg in range(ngal)])

    
    ###Need to renormlaize before doing that
    dgal_DETECTIFz = ((mNtotclus * f_profile_imR200) / (f_loc_DETECTIFz * mNbkg_glob)) - 1
    ind_dgal = np.where(dgal_DETECTIFz > 0)[0]
    #cf Cucciati et al. (2018), even though the 2.652 factor is not explained - this is something related exp or ln(10)
    rv_lognorm = lognorm(s=1,loc=0.0,scale=np.sqrt(2.652))
    
    #f_beta = 1 - beta in CB16 formalism. But computed using dgal and the proba of a lognormal distribution for galaxu overdensity.
    fbeta_DETECTIFz_lognorm = rv_lognorm.cdf(dgal_DETECTIFz) 
    fbeta_DETECTIFz_Nexcess = dgal_DETECTIFz/(1.+dgal_DETECTIFz)
    
    Pcc = Pcc[ind_dgal] 
    pdzshift = np.asarray([np.roll(pdz_2Mpc[indg],(jc-np.argmax(pdz_2Mpc[indg]))) for indg in ind_dgal])
    pnorm = np.asarray([np.sum(Pcc[i_dgal]*pdzshift[i_dgal]) if  np.sum(Pcc[i_dgal]*pdzshift[i_dgal]) > np.sum(Pcc[i_dgal]*pdz_2Mpc[indg]) else np.sum(Pcc[i_dgal]*pdz_2Mpc[indg]) for i_dgal,indg in enumerate(ind_dgal)])
    
    Pmem = np.array([(fbeta_DETECTIFz_Nexcess[indg]) * np.sum(pdz_2Mpc[indg]*Pcc[i_dgal]) for i_dgal,indg in enumerate(ind_dgal)])
    Pmem = Pmem / pnorm
    
    Pmem_lognorm = np.array([(fbeta_DETECTIFz_lognorm[indg]) * np.sum(pdz_2Mpc[indg]*Pcc[i_dgal]) for i_dgal,indg in enumerate(ind_dgal)])
    Pmem_lognorm = Pmem_lognorm / pnorm
    
    
    memcat=pd.DataFrame()
    
    memcat['id'] = gal_2Mpc[ind_dgal]['id'][np.argsort(Pmem)]
    memcat['ra'] = gal_2Mpc[ind_dgal]['ra'][np.argsort(Pmem)] 
    memcat['dec']= gal_2Mpc[ind_dgal]['dec'][np.argsort(Pmem)] 
    memcat['z'] = gal_2Mpc[ind_dgal]['z'][np.argsort(Pmem)]
    memcat['mK'] = gal_2Mpc[ind_dgal]['mK'][np.argsort(Pmem)]
    memcat['rsky'] = rgc[ind_dgal][np.argsort(Pmem)]
    memcat['rMpc'] = rgcMpc[ind_dgal][np.argsort(Pmem)]
    memcat['Bestfit_Mass'] = gal_2Mpc[ind_dgal]['Bestfit_Mass'][np.argsort(Pmem)]
#    memcat['SFR'] = gal_2Mpc[ind_dgal]['SFR'][np.argsort(Pmem)]
    memcat['pDETECTIFz_Nexcess'] = np.minimum(np.round(Pmem[np.argsort(Pmem)],5),0.99999)
    memcat['pDETECTIFz_lognorm'] = np.minimum(np.round(Pmem_lognorm[np.argsort(Pmem)],5),0.99999)
    memcat['fcorr_profile']  = np.repeat(1,len(ind_dgal))

    

    return memcat

@ray.remote
def membership_oneclus_mass_sigzlim_sigM(indc, gal_id, gal_colnames, mem_info, pdz, zz, pdM, MM, pdzclus, masks_im, headmasks, dzpix_68, dzpix, dMpix_68, Nfield):
    if not sys.warnoptions:
        warnings.simplefilter('ignore', category=AstropyWarning)
        warnings.simplefilter('ignore', category=AstropyUserWarning)
        warnings.simplefilter("ignore")
        
    if not indc % 50:    
        print('clus ',indc)
        
    wmasks = wcs.WCS(headmasks)

    Nfield_glob_Mz = Nfield / (u.deg*u.deg)

    Nz = len(zz)
    NM = len(MM)
        
    mmem_info = np.copy(mem_info[indc])
    
    ## extract info from mem_info
    raclus = mmem_info[1]
    decclus = mmem_info[2]
    zclus = mmem_info[3] #zpeak (NOT zMST)
    r200Mpc = mmem_info[4]
    rmaxMpc = mmem_info[5]
    rmax = angsep_radius(zclus,rmaxMpc)
    r200 = angsep_radius(zclus,r200Mpc)
    clus_imR200_r200 = mmem_info[6]
    ind_gal2Mpc = mmem_info[7]
    ind_galR200 = mmem_info[8]
    gal_imR200 = mmem_info[9]
    f_loc_DETECTIFz = mmem_info[10]
    f_profile_imR200 = 10**(gal_imR200-clus_imR200_r200)


    gal = Table(gal_id,names=gal_colnames)
    
    ###Ntotclus_mzr (so true for a given cluster whatever galaxy mag,z,rgc)
    #####      
    pdzr = pdz[ind_galR200]
    pdMr = pdM[ind_galR200]
    N_totclus_Mzr200 = Ntotclus_mz(masks_im,wmasks,pdzr,pdMr,NM,Nz,raclus,decclus,r200) / (u.deg*u.deg)
    
    
    
    #### Now work for each galaxy (of given mag)
    Pmem = []

    gal_2Mpc =  gal[ind_gal2Mpc]
    ngal = len(gal_2Mpc)
    pdz_2Mpc = pdz[ind_gal2Mpc]
    
    cluscat = SkyCoord(ra=raclus*u.degree, dec=decclus*u.degree, frame='fk5')
    galcat_2Mpc = SkyCoord(ra=gal_2Mpc['ra']*u.degree, dec=gal_2Mpc['dec']*u.degree, frame='fk5')
    rgc = cluscat.separation(galcat_2Mpc)
    rgcMpc = physep_ang(zclus,rgc.value)
    Mgal = np.array(gal_2Mpc['Bestfit_Mass'])
    

    zzbin = np.linspace(0.005,5.005,501)
    jc = np.digitize(zclus,zzbin)-1
    MMbin = np.linspace(3.95,14.15,103)
    ig = np.digitize(Mgal,MMbin)-1
    dM = np.maximum(2,2*dMpix_68[ig,jc])
    
    
    win = np.array([signal.gaussian(M=501,std=max(2,dzpix_68[ig[indg],jc])) for indg in range(ngal)]) 
    Pcc = np.array([signal.convolve(pdzclus[indc],win[indg], mode='same')/sum(win[indg]) for indg in range(ngal)]) 
    conflim='95'
    #zlims = zlim_pdzclus_gal(Pcc,conflim)
    #izlims = np.digitize(zlims,zzbin)-1
    dz = np.maximum(2,dzpix[ig,jc])

    ###mNlocbkg
    ###########
    mNbkg_glob = np.asarray([mN(Nfield_glob_Mz,ig[indg],jc,dM[indg],dz[indg],NM,Nz) for indg in range(ngal)])

    ###mNtotclus
    ##################
    ##corretion to take into account cluster geometry
    mNtotclus = np.asarray([mN(N_totclus_Mzr200,ig[indg],jc,dM[indg],dz[indg],NM,Nz) for indg in range(ngal)])

    
    ###Need to renormlaize before doing that
    dgal_DETECTIFz = ((mNtotclus * f_profile_imR200) / (f_loc_DETECTIFz * mNbkg_glob)) - 1
    ind_dgal = np.where(dgal_DETECTIFz > 0)[0]
    #cf Cucciati et al. (2018), even though the 2.652 factor is not explained - this is something related exp or ln(10)
    rv_lognorm = lognorm(s=1,loc=0.0,scale=np.sqrt(2.652))
    
    #f_beta = 1 - beta in CB16 formalism. But computed using dgal and the proba of a lognormal distribution for galaxu overdensity.
    fbeta_DETECTIFz_lognorm = rv_lognorm.cdf(dgal_DETECTIFz) 
    fbeta_DETECTIFz_Nexcess = dgal_DETECTIFz/(1.+dgal_DETECTIFz)
    
    Pcc = Pcc[ind_dgal] 
    pdzshift = np.asarray([np.roll(pdz_2Mpc[indg],(jc-np.argmax(pdz_2Mpc[indg]))) for indg in ind_dgal])
    pnorm = np.asarray([np.sum(Pcc[i_dgal]*pdzshift[i_dgal]) if  np.sum(Pcc[i_dgal]*pdzshift[i_dgal]) > np.sum(Pcc[i_dgal]*pdz_2Mpc[indg]) else np.sum(Pcc[i_dgal]*pdz_2Mpc[indg]) for i_dgal,indg in enumerate(ind_dgal)])
    
    Pmem = np.array([(fbeta_DETECTIFz_Nexcess[indg]) * np.sum(pdz_2Mpc[indg]*Pcc[i_dgal]) for i_dgal,indg in enumerate(ind_dgal)])
    Pmem = Pmem / pnorm
    
    Pmem_lognorm = np.array([(fbeta_DETECTIFz_lognorm[indg]) * np.sum(pdz_2Mpc[indg]*Pcc[i_dgal]) for i_dgal,indg in enumerate(ind_dgal)])
    Pmem_lognorm = Pmem_lognorm / pnorm
    
    
    memcat=pd.DataFrame()
    
    memcat['id'] = gal_2Mpc[ind_dgal]['id'][np.argsort(Pmem)]
    memcat['ra'] = gal_2Mpc[ind_dgal]['ra'][np.argsort(Pmem)] 
    memcat['dec']= gal_2Mpc[ind_dgal]['dec'][np.argsort(Pmem)] 
    memcat['z'] = gal_2Mpc[ind_dgal]['z'][np.argsort(Pmem)]
    memcat['mK'] = gal_2Mpc[ind_dgal]['mK'][np.argsort(Pmem)]
    memcat['rsky'] = rgc[ind_dgal][np.argsort(Pmem)]
    memcat['rMpc'] = rgcMpc[ind_dgal][np.argsort(Pmem)]
    memcat['Bestfit_Mass'] = gal_2Mpc[ind_dgal]['Bestfit_Mass'][np.argsort(Pmem)]
#    memcat['SFR'] = gal_2Mpc[ind_dgal]['SFR'][np.argsort(Pmem)]
    memcat['pDETECTIFz_Nexcess'] = np.minimum(np.round(Pmem[np.argsort(Pmem)],5),0.99999)
    memcat['pDETECTIFz_lognorm'] = np.minimum(np.round(Pmem_lognorm[np.argsort(Pmem)],5),0.99999)
    memcat['fcorr_profile']  = np.repeat(1,len(ind_dgal))

    

    return memcat