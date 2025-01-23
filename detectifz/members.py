import sys
import numpy as np
#import numba as nb
from astropy.table import Table, Column
from pathlib import Path
from astropy.io import fits
import ray
from astropy import units
import psutil
from astropy import wcs
import scipy.signal
import astropy.cosmology

from .utils import angsep_radius, physep_ang #, Mlim

from astropy.coordinates import SkyCoord
from photutils import CircularAperture, CircularAnnulus
from photutils import SkyCircularAnnulus, SkyCircularAperture

from astropy.wcs.utils import proj_plane_pixel_area,proj_plane_pixel_scales

# import lnr

import warnings
from astropy.utils.exceptions import AstropyWarning,AstropyUserWarning

import time
from . import MembersUtils as MemUtils
from astropy.stats import sigma_clipped_stats





@ray.remote
def field_weights_noclus(iz,clus_id,colnames,pTH_id,gal_id,weights2d_id,head2d,radnoclus):
    warnings.simplefilter("ignore")
    if not(iz%100):
        print(iz)
    clus = Table(clus_id,names=colnames,copy=True)
    w2d = wcs.WCS(head2d)
    pTH = np.copy(pTH_id)
    idxclus = np.where(pTH[:,iz] > 0)[0]
    clussky = SkyCoord(clus['ra'],clus['dec'],unit='deg',frame='fk5')
    aper = np.array([
        SkyCircularAperture(clussky[i],r=radnoclus*angsep_radius(clus['z'][i],clus['R200c_Mass_median'][i])) for i in idxclus])
    aper_pix = [ap.to_pixel(w2d) for ap in aper]
    aper_masks = [apix.to_mask(method='center') for apix in aper_pix]
    wwe = np.copy(weights2d_id)
    for mask in aper_masks:
        slices_large, slices_small = mask._overlap_slices(wwe.shape)
        wwe[slices_large] *= (~((mask.data[slices_small]).astype(bool))).astype(int)
    weights2d = np.copy(weights2d_id)
    wwe = wwe * weights2d
    weights3d_noclus = wwe.astype(bool)
    area_F_z = np.sum(weights3d_noclus)*proj_plane_pixel_area(w2d)
    
    gal = np.copy(gal_id)
    mask_noclus = np.ones(len(gal)).astype(bool)
    galsky = SkyCoord(gal[:,0],gal[:,1],unit='deg',frame='fk5')
    for indc in idxclus:
        mask_noclus *= clussky[indc].separation(galsky) > radnoclus*angsep_radius(clus['z'][indc],clus['R200c_Mass_median'][indc])   
 
    return weights3d_noclus, area_F_z, mask_noclus


def get_nfield(field,clus,pdzclus,zz,gal,sigz68_z,weights2d,head2d,radnoclus,SNmin,nprocs,getQ=False):

    w2d = wcs.WCS(head2d)
    
    wf = 'weights3d.'+field+'.noclus.SN'+str(SNmin)+'.'+str(radnoclus)+'r200.npz'
    if Path(wf).is_file():
        weights3d_noclus=np.load(wf)['weights3d_noclus']
        area_F_z=np.load(wf)['area_F_z']
        mask_noclus=np.load(wf)['mask_noclus']
    else:
        zzbin = np.linspace(0.005,5.005,501)
        nclus = len(clus)
        pTH = np.zeros((nclus,len(zz)))
        for indc in range(nclus):
            jc = np.digitize(clus['z'][indc],zzbin)-1
            sigz = np.copy(sigz68_z[jc])
            pclus = np.copy(pdzclus[indc])
            dzpix68_z = (sigz / 0.01)#.astype(int)
            #win = scipy.signal.gaussian(M=501,std=max(2,dzpix68_z))
            #Pcc = scipy.signal.convolve(pclus,win, mode='same')/sum(win)
            Pcc = gaussian_filter1d(pclus,dzpix68_z) #max(2,dzpix68_z)) 
            rv = scipy.stats.rv_histogram((Pcc,zzbin))
            zinf, zsup = rv.interval(0.95)
            izinf, izsup = np.digitize(zinf,zzbin)-1, np.digitize(zsup,zzbin)-1
            pTH[indc][izinf:izsup] = 1 
    
        print('compute field area...')        
    
        memo = 2*(1024**3) + (weights2d.astype(int)).nbytes
        mem_avail = psutil.virtual_memory().available
        
        if memo < 0.9*mem_avail:    
            memo_obj = int(0.9*memo)
            memo_heap = max(60000000,memo - memo_obj)
            ray.init(num_cpus=nprocs,memory=memo_heap, object_store_memory=memo_obj,ignore_reinit_error=True,log_to_driver=True)
    
            weights2d_id = ray.put(weights2d.astype(int))
            clus_id = ray.put(clus.to_pandas().to_numpy())
            pTH_id = ray.put(pTH)
            gal_radec = np.c_[gal['ra'],gal['dec']] 
            gal_id = ray.put(gal_radec)

            res = ray.get([field_weights_noclus.remote(iz,clus_id,clus.colnames,pTH_id,gal_id,weights2d_id,head2d,radnoclus) for iz in range(len(zz))])    
            ray.shutdown()  
            res = np.array(res)
            weights3d_noclus = np.stack(res[:,0])
            area_F_z = np.stack(res[:,1])
            mask_noclus = np.stack(res[:,2])

            np.savez('weights3d.'+field+'.noclus.SN'+str(SNmin)+'.'+str(radnoclus)+'r200.npz',
                     weights3d_noclus=weights3d_noclus,
                     area_F_z=area_F_z,
                     mask_noclus=mask_noclus)
        else:
            raise ValueError('Not enough memory available : ',memo,'<',mem_avail)
        print('...done')
    '''
    weights3d_noclus = np.empty((len(zz),weights2d.shape[0],weights2d.shape[1]),dtype='bool')
    area_F_z = np.zeros(len(zz))
    mask_noclus = np.ones((len(zz),len(gal))).astype(bool)
    gal_radec = np.c_[gal['ra'],gal['dec']] 
    res = []
    for iz in range(len(zz)):
        #if not (iz%50):
            #print(iz)
        weights3d_noclus[iz], area_F_z[iz], mask_noclus[iz] = field_weights_noclus(iz,clus.to_pandas().to_numpy(),clus.colnames,pTH,gal_radec,weights2d.astype(int),head2d,radnoclus)
    '''
    print('compute NF_Mz...')
    #sSFRlim = 180
    sSFRlim = 186
    #compute N(M,z) by summing the PDFs(M,z)
    if field == 'UDS':
        offset=0
    if field == 'UltraVISTA':
        offset=0#-2
    if field == 'VIDEO':
        offset=0#-1
    hdff = h5py.File('catalogues/'+field+'/'+field+'.master.PDF_Mz.irac.hdf','r')
    pdMz = hdff['pdf_mass_z']
    Nf_Mz_noclus = np.zeros_like(pdMz[0],dtype=np.float64)
    if getQ:
        #pQzf = 'pQ.z.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+'.offset.npz'
        pQzf = 'pQ.z.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+'.sSFR_10.7.npz'
        if Path(pQzf).is_file():
            pQexists=True
            pQ_z = np.load(pQzf,allow_pickle=True)['pQ_z']
        else:
            pQexists=False
            pdsSFRz = hdff['pdf_ssfr_z']
            pQ_z = np.zeros((len(pdsSFRz),len(zz)),dtype=np.float64)   
        NQf_Mz_noclus = np.zeros_like(pdMz[0],dtype=np.float64)
    else:
        NQf_Mz_noclus=None
        
    for igal in range(len(gal)):
        if not(igal%5000):
            print(igal)
        ppMz = pdMz[igal]
        Nf_Mz_noclus[mask_noclus[:,igal]] += ppMz[mask_noclus[:,igal]]
        if getQ:
            if not(pQexists):
                ppsSFRz =  pdsSFRz[igal]   
                pQ_z[igal] = np.nansum(ppsSFRz[:,:sSFRlim+offset],axis=1)/np.nansum(ppsSFRz,axis=1)    
            ppQ = ppMz * pQ_z[igal][:,None]
            ppQ[np.isnan(ppQ)] = 0
            NQf_Mz_noclus[mask_noclus[:,igal]] += ppQ[mask_noclus[:,igal]]
    hdff.close()
    if getQ and not(pQexists):
        np.savez(pQzf,pQ_z=pQ_z)
    print('...done')    
    
    return Nf_Mz_noclus, NQf_Mz_noclus, area_F_z, weights3d_noclus, mask_noclus

  
def get_nfield_dgal_lim(field,clus,pdzclus,zz,gal,sigz68_z,im3d,weights2d,head2d,nprocs,getQ=False):

    w2d = wcs.WCS(head2d)
    
    wf = 'weights3d.'+field+'.dgal_lim.npz'
    if Path(wf).is_file():
        weights3d_dgal=np.load(wf)['weights3d_dgal']
        area_F_z_dgal=np.load(wf)['area_F_z']
        mask_dgal=np.load(wf)['mask_dgal']
    else:
        zzbin = np.linspace(0.005,5.005,501)
        nclus = len(clus)
    
        print('compute field area...')
        
        weights3d_dgal = np.zeros((len(zz),im3d.shape[1],im3d.shape[2]),dtype='bool')
        
        ##log(dgal+1) > 1
        for i in range(len(im3d)):
            im3d[i][~(weights2d.astype(bool))] = np.nan
            im3d[i][im3d[i] > 0] = np.nan
            weights3d_dgal[i+9] = ~np.isnan(im3d[i])
        weights3d_dgal[:9] = weights3d_dgal[9]
        weights3d_dgal[9+len(im3d):] = weights3d_dgal[9+len(im3d)-1]

        galsky = SkyCoord(gal['ra'],gal['dec'],unit='deg',frame='fk5')
        mask_dgal = np.zeros((len(zz),len(gal)),dtype='bool')
        mask_dgal = np.stack(MemUtils.Ntot_pos(weights3d_dgal,w2d,galsky)).T
        
        area_F_z_dgal = np.sum(weights3d_dgal,axis=(1,2)) * proj_plane_pixel_area(w2d)
        np.savez(wf,weights3d_dgal=weights3d_dgal,area_F_z=area_F_z_dgal,mask_dgal=mask_dgal)

    print('compute NF_Mz...')
    #sSFRlim = 180
    sSFRlim = 186
    #compute N(M,z) by summing the PDFs(M,z)
    if field == 'UDS':
        offset=0
    if field == 'UltraVISTA':
        offset=0#-2
    if field == 'VIDEO':
        offset=0#-1
    hdff = h5py.File('catalogues/'+field+'/'+field+'.master.PDF_Mz.irac.hdf','r')
    pdMz = hdff['pdf_mass_z']
    Nf_Mz_dgal = np.zeros_like(pdMz[0],dtype=np.float64)
    if getQ:
        #pQzf = 'pQ.z.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+'.offset.npz'
        pQzf = 'pQ.z.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+'.sSFR_10.7.npz'
        if Path(pQzf).is_file():
            pQexists=True
            pQ_z = np.load(pQzf,allow_pickle=True)['pQ_z']
        else:
            pQexists=False
            pdsSFRz = hdff['pdf_ssfr_z']
            pQ_z = np.zeros((len(pdsSFRz),len(zz)),dtype=np.float64)   
        NQf_Mz_dgal = np.zeros_like(pdMz[0],dtype=np.float64)
    else:
        NQf_Mz_dgal=None
        
    for igal in range(len(gal)):
        if not(igal%5000):
            print(igal)
        ppMz = pdMz[igal]
        Nf_Mz_dgal[mask_dgal[:,igal]] += ppMz[mask_dgal[:,igal]]
        if getQ:
            if not(pQexists):
                ppsSFRz =  pdsSFRz[igal]   
                pQ_z[igal] = np.nansum(ppsSFRz[:,:sSFRlim+offset],axis=1)/np.nansum(ppsSFRz,axis=1)    
            ppQ = ppMz * pQ_z[igal][:,None]
            ppQ[np.isnan(ppQ)] = 0
            NQf_Mz_dgal[mask_dgal[:,igal]] += ppQ[mask_dgal[:,igal]]
    hdff.close()
    if getQ and not(pQexists):
        np.savez(pQzf,pQ_z=pQ_z)
    print('...done')    
    
    return Nf_Mz_dgal, NQf_Mz_dgal, area_F_z_dgal, weights3d_dgal, mask_dgal
    
    
    
    
@ray.remote   
def clus_weights(indc,clus_id,colnames,gal_id,weights2d_id,head2d):
    if not(indc%300):
        print(indc)
    w2d = wcs.WCS(head2d)

    clus = Table(clus_id,names=colnames,copy=True)
    cc = clus[indc]
    gal = gal_id
    weights2d = np.copy(weights2d_id)
    
    clussky = SkyCoord(cc['ra'],cc['dec'],unit='deg',frame='fk5')
    galsky = SkyCoord(gal[:,0],gal[:,1],unit='deg',frame='fk5')

    mask_clus_r200= clussky.separation(galsky) < angsep_radius(cc['z'],cc['R200c_Mass_median']) 
    mask_clus_2Mpc= clussky.separation(galsky) < angsep_radius(cc['z'],2) 
      
    mask_clus_r200_frac0= clussky.separation(galsky) < angsep_radius(cc['z'],0.5*cc['R200c_Mass_median']) 
    mask_clus_r200_frac1= ((clussky.separation(galsky) >= angsep_radius(cc['z'],0.5*cc['R200c_Mass_median'])) &
                           (clussky.separation(galsky) < angsep_radius(cc['z'],cc['R200c_Mass_median'])))
    mask_clus_r200_frac2= ((clussky.separation(galsky) >= angsep_radius(cc['z'],cc['R200c_Mass_median'])) &
                           (clussky.separation(galsky) < angsep_radius(cc['z'],2*cc['R200c_Mass_median'])))

    aper = SkyCircularAperture(clussky,r=angsep_radius(cc['z'],cc['R200c_Mass_median']))
    aper_data = aper.to_pixel(w2d).to_mask(method='center').multiply(weights2d)
    Omega_C_r200 = np.sum(aper_data)*proj_plane_pixel_area(w2d)
    
    aper = SkyCircularAperture(clussky,r=angsep_radius(cc['z'],0.5*cc['R200c_Mass_median']))
    aper_data = aper.to_pixel(w2d).to_mask(method='center').multiply(weights2d)
    Omega_C_r200_frac0 = np.sum(aper_data)*proj_plane_pixel_area(w2d)
    
    aper = SkyCircularAnnulus(clussky,
                              r_in=angsep_radius(cc['z'],0.5*cc['R200c_Mass_median']),
                              r_out=angsep_radius(cc['z'],1*cc['R200c_Mass_median']))
    aper_data = aper.to_pixel(w2d).to_mask(method='center').multiply(weights2d)
    Omega_C_r200_frac1 = np.sum(aper_data)*proj_plane_pixel_area(w2d)
    
    aper = SkyCircularAnnulus(clussky,
                              r_in=angsep_radius(cc['z'],1*cc['R200c_Mass_median']),
                              r_out=angsep_radius(cc['z'],2*cc['R200c_Mass_median']))
    aper_data = aper.to_pixel(w2d).to_mask(method='center').multiply(weights2d)
    Omega_C_r200_frac2 = np.sum(aper_data)*proj_plane_pixel_area(w2d)
    

    return mask_clus_r200,mask_clus_2Mpc,mask_clus_r200_frac0,mask_clus_r200_frac1,mask_clus_r200_frac2,Omega_C_r200, Omega_C_r200_frac0, Omega_C_r200_frac1, Omega_C_r200_frac2

def get_ntot_r200(field,clus,gal,weights2d,head2d,zz,MM,SNmin,nprocs,getQ=False):
    nclus=len(clus)
    
    print('compute clus area...')        
    memo = 4*(1024**3)
    mem_avail = psutil.virtual_memory().available
        
    if memo < 0.9*mem_avail:    
        memo_obj = int(0.9*memo)
        memo_heap = max(60000000,memo - memo_obj)
        ray.init(num_cpus=nprocs,memory=memo_heap, object_store_memory=memo_obj,ignore_reinit_error=True,log_to_driver=True)
    
        weights2d_id = ray.put(weights2d)  
        clus_id = ray.put(clus.to_pandas().to_numpy())
        gal_radec = np.c_[gal['ra'],gal['dec']] 
        gal_id = ray.put(gal_radec)

        res = ray.get([clus_weights.remote(indc,clus_id,clus.colnames,gal_id,weights2d_id,head2d) for indc in range(nclus)])    
        ray.shutdown()  
        res = np.array(res)
        mask_clus_r200 = np.stack(res[:,0])
        mask_clus_2Mpc = np.stack(res[:,1])
        mask_clus_r200_frac0 = np.stack(res[:,2])
        mask_clus_r200_frac1 = np.stack(res[:,3])
        mask_clus_r200_frac2 = np.stack(res[:,4])
        Omega_C_r200 = np.stack(res[:,5])
        Omega_C_r200_frac0 = np.stack(res[:,6])
        Omega_C_r200_frac1 = np.stack(res[:,7])
        Omega_C_r200_frac2 = np.stack(res[:,8])

    else:
        raise ValueError('Not enough memory available : ',memo,'<',mem_avail)
    print('...done')

    start=time.time()
    Ntot_Mz_r200 = np.zeros((nclus,len(zz),160),dtype=np.float64)
    Ntot_Mz_r200_frac0 = np.zeros((nclus,len(zz),160),dtype=np.float64)
    Ntot_Mz_r200_frac1 = np.zeros((nclus,len(zz),160),dtype=np.float64)
    Ntot_Mz_r200_frac2 = np.zeros((nclus,len(zz),160),dtype=np.float64)

    #sSFRlim = 180
    sSFRlim = 186
    hdff = h5py.File('catalogues/'+field+'/'+field+'.master.PDF_Mz.irac.hdf','r')
    pdMz = hdff['pdf_mass_z']
    if getQ:
        pQzf = 'pQ.z.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+'.sSFR_10.7.npz'
        if Path(pQzf).is_file():
            pQexists=True
            pQ_z = np.load(pQzf,allow_pickle=True)['pQ_z']
        else:
            pQexists=False
            pdsSFRz = hdff['pdf_ssfr_z']
            pQ_z = np.zeros((len(pdsSFRz),len(zz)),dtype=np.float64)   
        NQ_Mz_r200 = np.zeros((nclus,len(zz),160),dtype=np.float64)
        NQ_Mz_r200_frac0 = np.zeros((nclus,len(zz),160),dtype=np.float64)
        NQ_Mz_r200_frac1 = np.zeros((nclus,len(zz),160),dtype=np.float64)
        NQ_Mz_r200_frac2 = np.zeros((nclus,len(zz),160),dtype=np.float64)
    else:
        NQ_Mz_r200=None
        NQ_Mz_r200_frac0=None
        NQ_Mz_r200_frac1=None
        NQ_Mz_r200_frac2=None
        
    for igal in range(len(gal)):
        if not(igal%5000):
            print(igal)
        ppMz = pdMz[igal]
        Ntot_Mz_r200[mask_clus_r200[:,igal]] += ppMz
        Ntot_Mz_r200_frac0[mask_clus_r200_frac0[:,igal]] += ppMz
        Ntot_Mz_r200_frac1[mask_clus_r200_frac1[:,igal]] += ppMz
        Ntot_Mz_r200_frac2[mask_clus_r200_frac2[:,igal]] += ppMz
        if getQ:
            if not(pQexists):
                ppsSFRz =  pdsSFRz[igal]   
                pQ_z[igal] = np.nansum(ppsSFRz[:,:sSFRlim],axis=1)/np.nansum(ppsSFRz,axis=1)    
            ppQ = ppMz * pQ_z[igal][:,None]
            ppQ[np.isnan(ppQ)] = 0
            NQ_Mz_r200[mask_clus_r200[:,igal]] += ppQ
            NQ_Mz_r200_frac0[mask_clus_r200_frac0[:,igal]] += ppQ
            NQ_Mz_r200_frac1[mask_clus_r200_frac1[:,igal]] += ppQ
            NQ_Mz_r200_frac2[mask_clus_r200_frac2[:,igal]] += ppQ
    hdff.close()
    if getQ and not(pQexists):
        np.savez(pQzf,pQ_z=pQ_z)
    
    #Nr200fracf = 'NQtotR200frac.Mz.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+'.offset.npz'
    Nr200fracf = 'NQtotR200frac.Mz.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+'.sSFR_10.7.npz'
    np.savez(Nr200fracf,
             Ntot_Mz_r200_fracs=np.array([Ntot_Mz_r200_frac0,Ntot_Mz_r200_frac1,Ntot_Mz_r200_frac2]),
             NQ_Mz_r200_fracs=np.array([NQ_Mz_r200_frac0,NQ_Mz_r200_frac1,NQ_Mz_r200_frac2]),
             Omega_C_r200_fracs=np.array([Omega_C_r200_frac0,Omega_C_r200_frac1,Omega_C_r200_frac2]) ,
            mask_clus_fracs = np.array([mask_clus_r200_frac0,mask_clus_r200_frac1,mask_clus_r200_frac2]))  
    
    print('done in ',time.time()-start,'s')   
    
    return Ntot_Mz_r200, NQ_Mz_r200, Omega_C_r200, mask_clus_r200, mask_clus_2Mpc 
    
@ray.remote(max_calls=10)
def Pmem_fpos_im3d(indc,clus_id,colnames,gal_id,im3d_id,weights_id,head2d):
    warnings.simplefilter("ignore")
    
    cclus = Table(np.copy(clus_id[indc:indc+1]),names=colnames)

    slice_clus = int(cclus['slice_idx'][0])
    im2d = np.copy(im3d_id[slice_clus])
    w2d = wcs.WCS(head2d)
    
    galsky = SkyCoord(gal_id[:,0],gal_id[:,1],unit='deg',frame='fk5')
    galpix = np.array(galsky.to_pixel(w2d)).T 
    
    rmaxMpc = 2.0                          

    idclus = cclus['id'][0]
    raclus = cclus['ra'][0]
    decclus = cclus['dec'][0]
    zclus = cclus['z'][0]
    r200Mpc = cclus['R200c_Mass_median'][0]
    
    clussky = SkyCoord(raclus,decclus,unit='deg',frame='fk5')
    
    rgc = clussky.separation(galsky)
    
    r2Mpc = angsep_radius(zclus,rmaxMpc)
    r200 = angsep_radius(zclus,r200Mpc)
    #print(indc,r200)
    
    ind_gal2Mpc = np.where(rgc < r2Mpc)[0]
    ind_galR200 = np.where(rgc < r200)[0]
    im2d[~weights_id] = np.nan
    clus_im3d_r200 = MemUtils.log_dgal_r(im2d,w2d,raclus,decclus,r200)
    
    gal_im3d = MemUtils.Ntot_pos_imR200(im2d,w2d,galsky[ind_gal2Mpc])
        
    im3d_info = np.empty(4, dtype='object')
    im3d_info[0] = idclus
    im3d_info[1] = clus_im3d_r200
    im3d_info[2] = gal_im3d
        
    return im3d_info

        
@ray.remote(max_calls=10)
def Pmem_floc_im3d(indc,clus_id,colnames,im3d_id,weights_id,head2d):
    warnings.simplefilter("ignore")
        
    cclus = Table(np.copy(clus_id[indc:indc+1]),names=colnames)
    im2d = np.copy(im3d_id[int(cclus['slice_idx'])])  
    im2d[~weights_id] = np.nan
    w2d = wcs.WCS(head2d)   

    raclus = cclus['ra'][0]
    decclus = cclus['dec'][0]
    zclus = cclus['z'][0]
    
    rinf_Mpc = 3.0
    rsup_Mpc = 5.0
    rinf = MemUtils.angsep_radius(zclus,rinf_Mpc)
    rsup = MemUtils.angsep_radius(zclus,rsup_Mpc)
    #print(indc,zclus,rinf,rsup)
    clus_fieldloc = MemUtils.Nloc_oneclus(im2d,w2d,raclus,decclus,rinf,rsup)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyUserWarning)
        clus_fieldglob = sigma_clipped_stats(im2d,sigma=3)[0]

    floc_im3d = 10**(clus_fieldloc-clus_fieldglob) 
    return floc_im3d
    
    
    
def get_im3d_info(im3d,weights2d,head2d,clus,gal,nprocs):
    
    start = time.time()
    memo = 1.8*1024**3 #1.5*(im3d.nbytes + weights2d.nbytes)
    mem_avail = 10*1024**3#psutil.virtual_memory().available
    
    if memo < 0.9*mem_avail: 
        memo_obj = int(0.9*memo)
        ray.init(num_cpus=int(1*nprocs), 
                 object_store_memory=memo_obj,ignore_reinit_error=True,log_to_driver=True)
        im3d_id = ray.put(im3d)      
        clus_id = ray.put(clus.to_pandas().to_numpy())
        weights_id = ray.put(weights2d)
    
        start = time.time() 
        res = ray.get([Pmem_floc_im3d.remote(indc,clus_id,clus.colnames,im3d_id,weights_id,head2d) for indc in range(len(clus))])
        floc = np.array(res)
        ray.shutdown()
    else:
        raise ValueError('Not enough memory available : ',memo/(1024**3),'<',0.9*mem_avail/(1024**3),'GB')
    
    print('floc done in ',time.time()-start,'s')  
    
    start = time.time()
    memo = 1.8*1024**3 #1.5*(im3d.nbytes + weights2d.nbytes)
    mem_avail = 10*1024**3#psutil.virtual_memory().available   
    if memo < 0.9*mem_avail: 
        gal_radec = np.c_[gal['ra'],gal['dec']]       
        memo_obj = int(0.8*memo)
        ray.init(num_cpus=int(1*nprocs), 
                 object_store_memory=memo_obj,ignore_reinit_error=True,log_to_driver=True)
        im3d_id = ray.put(im3d)   
        weights_id = ray.put(weights2d)
        clus_id = ray.put(clus.to_pandas().to_numpy())
        gal_id = ray.put(gal_radec)
        
        start = time.time() 
        res = ray.get([Pmem_fpos_im3d.remote(indc,clus_id,clus.colnames,gal_id,im3d_id,weights_id,head2d) for indc in range(len(clus))])
        im3d_info = np.array(res)
        ray.shutdown()
        im3d_info[:,3] = floc

    else:
        raise ValueError('Not enough memory available : ',memo/(1024**3),'<',0.9*mem_avail/(1024**3),'GB')
    
    
    return im3d_info

def smooth_Nz_clusR200(survey,Ntot_Mz_r200,NQtot_Mz_r200,NF_Mz_noclus,NQF_Mz_noclus,Omega_z_noclus,Omega_C_r200,zz,MM,sigz95_Mz,getQ=False):
    Ml = Mlim(survey,0,zz)
    Ntot_Mz_r200_smooth = np.zeros_like(Ntot_Mz_r200)
    NF_Mz_noclus_R200clus = NF_Mz_noclus[:,:,None]*(Omega_C_r200/Omega_z_noclus[:,None,None])
    NF_Mz_noclus_R200clus_smooth = np.zeros_like(NF_Mz_noclus_R200clus)
    
    if getQ:
        NQtot_Mz_r200_smooth = np.zeros_like(Ntot_Mz_r200)
        NQF_Mz_noclus_R200clus = NQF_Mz_noclus[:,:,None]*(Omega_C_r200/Omega_z_noclus[:,None,None])
        NQF_Mz_noclus_R200clus_smooth = np.zeros_like(NF_Mz_noclus_R200clus)
    else:
        NQtot_Mz_r200_smooth = None
        NQF_Mz_noclus_R200clus = None
        NQF_Mz_noclus_R200clus_smooth = None
        
    MMbin = np.linspace(5,13,161)
    sigz95_Mz_pix = (sigz95_Mz/0.01).astype(int)
    Mlpix = np.digitize(Ml,MMbin)-1

    for iz in range(len(zz)):
        for jM in range(len(MM)):
            if jM > Mlpix[iz] and not(np.isnan(sigz95_Mz[iz,jM])):
                izinf, izsup = max(0,iz-sigz95_Mz_pix[iz,jM]), min(len(zz)-1,iz+sigz95_Mz_pix[iz,jM]+1)
    
                Ntot_Mz_r200_smooth[:,iz,jM] = np.nansum(Ntot_Mz_r200[:,izinf:izsup,jM],axis=(1))
                NF_Mz_noclus_R200clus_smooth[iz,jM,:] = np.nansum(NF_Mz_noclus_R200clus[izinf:izsup,jM,:],axis=(0))
                if getQ:
                    NQtot_Mz_r200_smooth[:,iz,jM] = np.nansum(NQtot_Mz_r200[:,izinf:izsup,jM],axis=(1))
                    NQF_Mz_noclus_R200clus_smooth[iz,jM,:] = np.nansum(NQF_Mz_noclus_R200clus[izinf:izsup,jM,:],axis=(0))

            
    return Ntot_Mz_r200_smooth, NQtot_Mz_r200_smooth, NF_Mz_noclus_R200clus_smooth, NQF_Mz_noclus_R200clus_smooth
    
def smooth_Nz(survey,Ntot_Mz_r200,NQtot_Mz_r200,NF_Mz_noclus,NQF_Mz_noclus,Omega_z_noclus,Omega_C_r200,zz,MM,sigz95_Mz,getQ=False):
    Ml = Mlim(survey,0,zz)
    Ntot_Mz_r200_smooth = -99.*np.ones_like(Ntot_Mz_r200)
    NF_Mz_noclus_smooth = -99.*np.ones_like(NF_Mz_noclus)
    
    if getQ:
        NQtot_Mz_r200_smooth = np.zeros_like(Ntot_Mz_r200)
        NQF_Mz_noclus_smooth = np.zeros_like(NF_Mz_noclus)
    else:
        NQtot_Mz_r200_smooth = None
        NQF_Mz_noclus = None
        NQF_Mz_noclus_smooth = None
        
    MMbin = np.linspace(5,13,161)
    sigz95_Mz_pix = (sigz95_Mz/0.01).astype(int)
    Mlpix = np.digitize(Ml,MMbin)-1

    for iz in range(len(zz)):
        for jM in range(len(MM)):
            if jM > Mlpix[iz] and not(np.isnan(sigz95_Mz[iz,jM])):
                izinf, izsup = max(0,iz-sigz95_Mz_pix[iz,jM]), min(len(zz)-1,iz+sigz95_Mz_pix[iz,jM]+1)
    
                Ntot_Mz_r200_smooth[:,iz,jM] = np.nansum(Ntot_Mz_r200[:,izinf:izsup,jM],axis=(1))
                NF_Mz_noclus_smooth[iz,jM] = np.nansum(NF_Mz_noclus[izinf:izsup,jM])
                if getQ:
                    NQtot_Mz_r200_smooth[:,iz,jM] = np.nansum(NQtot_Mz_r200[:,izinf:izsup,jM],axis=(1))
                    NQF_Mz_noclus_smooth[iz,jM] = np.nansum(NQF_Mz_noclus[izinf:izsup,jM])

            
    return Ntot_Mz_r200_smooth, NQtot_Mz_r200_smooth, NF_Mz_noclus_smooth, NQF_Mz_noclus_smooth
      
def advindexing_roll(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]    
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]
    return A[rows, column_indices]


def get_Pmem(field,mem_info,mask_clus_2Mpc,nF_Mz,Ntot_Mz_r200,Omega_C_r200,pdzgal,pdzclus,clus,zz,sigz68_Mz,MM,nprocs):
    nclus=len(clus)

    #memo = 2*(mem_info.nbytes + mask_clus_2Mpc.nbytes + nF_Mz.nbytes + Ntot_Mz_r200.nbytes +
    #           Omega_C_r200.nbytes + pdzgal.nbytes + pdzclus.nbytes + sigz68_Mz.nbytes)
    memo = 4*(1024**3)
    mem_avail = psutil.virtual_memory().available
        
    if memo < 0.9*mem_avail:    
        memo_obj = int(0.9*memo)
        memo_heap = max(60000000,memo - memo_obj)
        ray.init(num_cpus=nprocs,memory=memo_heap, object_store_memory=memo_obj,ignore_reinit_error=True,log_to_driver=False)
    
        mem_info_id = ray.put(mem_info)
        mask_clus_2Mpc_id = ray.put(mask_clus_2Mpc)
        nF_Mz_id = ray.put(nF_Mz)
        Ntot_Mz_r200_id = ray.put(Ntot_Mz_r200)
        Omega_C_r200_id = ray.put(Omega_C_r200)
        pdzgal_id = ray.put(pdzgal)
        pdzclus_id = ray.put(pdzclus)
        zclus_id = ray.put(np.array(clus['z']))
        sigz68_Mz_id = ray.put(sigz68_Mz)
                
        res = ray.get([compute_pmem.remote(indc,field,mem_info_id,mask_clus_2Mpc_id,nF_Mz_id,Ntot_Mz_r200_id,Omega_C_r200_id,pdzgal_id,pdzclus_id,zclus_id,zz,sigz68_Mz_id,MM) for indc in range(nclus)])    
        ray.shutdown()  
        res = np.array(res)
        Pmem = res[:,0]
        Mass_CL95 = res[:,1]

    else:
        raise ValueError('Not enough memory available : ',memo,'<',mem_avail)
            
    return Pmem, Mass_CL95

@ray.remote(max_calls=10)
def compute_pmem(indc,field,mem_info_id,mask_clus_2Mpc_id,nF_Mz_id,Ntot_Mz_r200_id,Omega_C_r200_id,pdzgal_id,pdzclus_id,zclus_id,zz,sigz68_Mz_id,MM):
    warnings.simplefilter("ignore")
    #print(indc)

    clus_imR200_r200 = np.copy(mem_info_id[indc,1])
    gal_imR200 = np.copy(mem_info_id[indc,2])
    f_loc_DETECTIFz = np.copy(mem_info_id[indc,3])
    f_profile_imR200 = 10**(gal_imR200-clus_imR200_r200)

    zzbin = np.linspace(0.005,5.005,501)
    MMbin = np.linspace(5,13,161)
    mask_clus_2Mpc = np.copy(mask_clus_2Mpc_id[indc])
    pgal = np.copy(pdzgal_id[mask_clus_2Mpc])
    pgal = pgal / np.sum(pgal,axis=1)[:,None]
    pclus = np.copy(pdzclus_id[indc])
    pclus = pclus / np.sum(pclus)
    
    pdMzf = h5py.File('catalogues/'+field+'/'+field+'.master.PDF_Mz.irac.hdf','r')
    pdMz = pdMzf['pdf_mass_z']
    idx_pdMz = np.where(mask_clus_2Mpc)[0]
    inf_pclus = np.where(pclus > 0)[0][0]
    sup_pclus = np.where(pclus > 0)[0][-1]+1
    ppM = np.sum(pdMz[idx_pdMz,inf_pclus:sup_pclus,:]*pclus[None,inf_pclus:sup_pclus,None],axis=1)
    pdMzf.close()
    
    Mass_CL95 = np.array([weighted_quantile(MM,[0.025,0.5,0.975],
                                            sample_weight=ppM[indg]) for indg in range(np.sum(mask_clus_2Mpc))])
    Mass_l95, Mass, Mass_u95 = Mass_CL95.T

    Masspix_l95 = np.digitize(Mass_l95,MMbin)-1 #np.digitize(Mass,MMbin)-1 - 10 #
    Masspix_u95 = np.digitize(Mass_u95,MMbin)-1 #np.digitize(Mass,MMbin)-1 + 10 #
    jc = np.digitize(zclus_id[indc],zzbin)-1
    #zclus_pdzmax = zz[np.argmax(pclus)]
    #jcc = np.digitize(zclus_pdzmax,zzbin)-1
    
    nans = np.isnan(Mass)
    ig = np.digitize(Mass[~nans],MMbin)-1
    sigz = np.copy(sigz68_Mz_id[jc,ig])
    dzpix68_Mz = (sigz / 0.01).astype(int)
   
    Pcc = np.array([gaussian_filter1d(pclus,max(2,dzpix68_Mz[indg])) for indg in range(np.sum(~nans))])
    zclus_Pccmax = zz[np.argmax(Pcc,axis=1)]
    jcc = np.digitize(zclus_Pccmax,zzbin)-1
    shift = jcc - np.argmax(pgal[~nans],axis=1)
    pnorm = np.sum(advindexing_roll(pgal[~nans],shift)*Pcc,axis=1)
    
    Masspix_comp90 = np.digitize(Mlim(field,0,zz[sup_pclus]),MMbin)-1
    
    
    NF_r200_sumMass = np.array([np.sum((nF_Mz_id*Omega_C_r200_id[indc,None,None])[
        :,max(Masspix_comp90,Masspix_l95[~nans][indg]):min(len(MM),Masspix_u95[~nans][indg])],axis=1) if Masspix_comp90 < Masspix_u95[~nans][indg] else np.repeat(0,len(zz)) for indg in range(np.sum(~nans))])

    Ntot_r200_sumMass = np.array([np.sum((Ntot_Mz_r200_id[indc])[
        :,max(Masspix_comp90,Masspix_l95[~nans][indg]):min(len(MM),Masspix_u95[~nans][indg])],axis=1) if Masspix_comp90 < Masspix_u95[~nans][indg] else np.repeat(0,len(zz)) for indg in range(np.sum(~nans))])

    Pprior_sumMass = 1 - (f_loc_DETECTIFz*NF_r200_sumMass)/(f_profile_imR200[~nans][:,None]*Ntot_r200_sumMass)
    Pprior_sumMass[np.where(Pprior_sumMass < 0)] = np.nan
    
    Pmem = np.zeros(len(Mass))
    Pmem[~nans] = np.nansum(Pprior_sumMass * pgal[~nans] * Pcc, axis=1) / pnorm
    Pmem = np.minimum(Pmem,0.99)
    
    return Pmem, Mass_CL95   


def get_Pmem_smooth(field,mocktype,survey,mem_info,mask_clus_2Mpc,SNF_Mz_noclus_R200clus,SNtot_Mz_r200,pdzgal,pdzclus,clus,zz,sigz68_Mz,MM,nprocs):
    start=time.time()
    nclus=len(clus)

    memo = 4*(1024**3)
    mem_avail = psutil.virtual_memory().available
        
    if memo < 0.9*mem_avail:    
        memo_obj = int(0.9*memo)
        memo_heap = max(60000000,memo - memo_obj)
        ray.init(num_cpus=nprocs,memory=memo_heap, object_store_memory=memo_obj,ignore_reinit_error=True,log_to_driver=False)
    
        mem_info_id = ray.put(mem_info)
        mask_clus_2Mpc_id = ray.put(mask_clus_2Mpc)
        SNF_Mz_noclus_R200clus_id = ray.put(SNF_Mz_noclus_R200clus)
        SNtot_Mz_r200_id = ray.put(SNtot_Mz_r200)
        pdzgal_id = ray.put(pdzgal)
        pdzclus_id = ray.put(pdzclus)
        zclus_id = ray.put(np.array(clus['z']))
        sigz68_Mz_id = ray.put(sigz68_Mz)
        #Mass_clus_id = ray.put(Mass_clus)
                
        res = ray.get([compute_pmem_smooth.remote(indc,field,mocktype,survey,mem_info_id,mask_clus_2Mpc_id,SNF_Mz_noclus_R200clus_id,SNtot_Mz_r200_id,pdzgal_id,pdzclus_id,zclus_id,zz,sigz68_Mz_id,MM) for indc in range(nclus)])    
        ray.shutdown()  
        res = np.array(res)
        Pmem = res[:,0]
        Mass_CL95 = res[:,1]
        Pconv = res[:,2]
    else:
        raise ValueError('Not enough memory available : ',memo,'<',mem_avail)
    print('done in ',time.time()-start,'s')   
        
    return Pmem, Mass_CL95, Pconv

@ray.remote
def compute_pmem_smooth(indc,field,mocktype,survey,mem_info_id,mask_clus_2Mpc_id,SNF_Mz_noclus_R200clus_id,SNtot_Mz_r200_id,pdzgal_id,pdzclus_id,zclus_id,zz,sigz68_Mz_id,MM):
    warnings.simplefilter("ignore")
    print(indc)

    clus_imR200_r200 = np.copy(mem_info_id[indc,1])
    gal_imR200 = np.copy(mem_info_id[indc,2])
    f_loc_DETECTIFz = np.copy(mem_info_id[indc,3])
    f_profile_imR200 = 10**(gal_imR200-clus_imR200_r200)

    zzbin = np.linspace(0.005,5.005,501)
    MMbin = np.linspace(5,13,161)
    mask_clus_2Mpc = np.copy(mask_clus_2Mpc_id[indc])
    pgal = np.copy(pdzgal_id[mask_clus_2Mpc])
    pgal = pgal / np.sum(pgal,axis=1)[:,None]
    pclus = np.copy(pdzclus_id[indc])
    pclus = pclus / np.sum(pclus)
    
    pdMzf = h5py.File('catalogues/'+field+'/'+field+'.master.PDF_Mz.irac.hdf','r')
    pdMz = pdMzf['pdf_mass_z']
    idx_pdMz = np.where(mask_clus_2Mpc)[0]
    inf_pclus = np.where(pclus > 0)[0][0]
    sup_pclus = np.where(pclus > 0)[0][-1]+1
    ppM = np.sum(pdMz[idx_pdMz,inf_pclus:sup_pclus,:]*pclus[None,inf_pclus:sup_pclus,None],axis=1)
    pdMzf.close()
    
    Mass_CL = np.array([weighted_quantile(MM,[0.025,0.16,0.5,0.84,0.975],
                                            sample_weight=ppM[indg]) for indg in range(np.sum(mask_clus_2Mpc))])
    
    Mass_l95, Mass_l68, Mass, Mass_u68, Mass_u95 = Mass_CL.T #Mass_clus_id[indc].T
    #Mass_l95 = Mass-0.5
    #Mass_u95 = Mass+0.5
    #Mass_CL95 = np.c_[Mass_l95, Mass, Mass_u95]
    
    Masspix_l95 = np.digitize(Mass_l95,MMbin)-1 #np.digitize(Mass,MMbin)-1 - 10 #
    Masspix_u95 = np.digitize(Mass_u95,MMbin)-1 #np.digitize(Mass,MMbin)-1 + 10 #
    jc = np.digitize(zclus_id[indc],zzbin)-1
    #zclus_pdzmax = zz[np.argmax(pclus)]
    #jcc = np.digitize(zclus_pdzmax,zzbin)-1
    
    nans = np.isnan(Mass)
    ig = np.digitize(np.minimum(12,Mass[~nans]),MMbin)-1
    sigz = np.copy(sigz68_Mz_id[jc,ig])
    #print(Mass[~nans][np.isnan(sigz)])
    dzpix68_Mz = sigz / 0.01 #.astype(int)
    #Pcc = np.array([gaussian_filter1d(pclus,max(1,dzpix68_Mz[indg])) for indg in range(np.sum(~nans))])
    Pcc = np.array([gaussian_filter1d(pclus,dzpix68_Mz[indg]) for indg in range(np.sum(~nans))])
    Pcclus = np.repeat(gaussian_filter1d(pclus,1),np.sum(~nans)).reshape(len(zz),np.sum(~nans)).T 
    zclus_Pccmax = zz[np.argmax(Pcclus,axis=1)]
    jcc = np.digitize(zclus_Pccmax,zzbin)-1
    shift = jcc - np.argmax(pgal[~nans],axis=1)
    pnorm = np.sum(advindexing_roll(pgal[~nans],shift)*Pcclus,axis=1)
    
    Masspix_comp90 = np.digitize(Mlim(survey,0,zz[sup_pclus]),MMbin)-1
    
    #print('NF_R200_indc',SNF_Mz_noclus_R200clus_id[:,:,indc])
    NF_r200_sumMass = np.array([np.sum(SNF_Mz_noclus_R200clus_id[:,:,indc][
        :,max(Masspix_comp90,Masspix_l95[~nans][indg]):min(len(MM),Masspix_u95[~nans][indg])],axis=1) if Masspix_comp90 < Masspix_u95[~nans][indg] else np.repeat(0,len(zz)) for indg in range(np.sum(~nans))])

    #print('NF',NF_r200_sumMass)
    Ntot_r200_sumMass = np.array([np.sum((SNtot_Mz_r200_id[indc])[
        :,max(Masspix_comp90,Masspix_l95[~nans][indg]):min(len(MM),Masspix_u95[~nans][indg])],axis=1) if Masspix_comp90 < Masspix_u95[~nans][indg] else np.repeat(0,len(zz)) for indg in range(np.sum(~nans))])
    #print('Ntot',Ntot_r200_sumMass)
    #print('fpro',f_profile_imR200.shape)
    #print('floc',f_loc_DETECTIFz.shape)
    Pprior_sumMass = 1 - (f_loc_DETECTIFz*NF_r200_sumMass)/(f_profile_imR200[~nans][:,None]*Ntot_r200_sumMass)
    Pprior_sumMass[np.where(Pprior_sumMass < 0)] = np.nan
    #print('Pprior',Pprior_sumMass)
    
    
    #d_Pprior_sumMass = np.array([(0.5*((1*Omega_C_r200_id[indc]/(3*Omega_F_z_id[:,None]*
    #                    np.sum(SNtot_Mz_r200[indc,:,max(Masspix_comp90,Masspix_l95[~nans][indg]):
    #                                         min(len(MM),Masspix_u95[~nans][indg])],axis=1))) * 
    #           np.sqrt(np.sum(SNF_Mz_noclus[:,max(Masspix_comp90,Masspix_l95[~nans][indg]):
    #                                        min(len(MM),Masspix_u95[~nans][indg])],axis=1)**2 + 
    #                   np.sum(SNF_Mz_noclus[iz,max(Masspix_comp90,Masspix_l95[~nans][indg]):
    #                                        min(len(MM),Masspix_u95[~nans][indg])],axis=1)) )
    #                   ) if Masspix_comp90 < Masspix_u95[~nans][indg] else np.repeat(0,len(zz)) for indg in range(np.sum(~nans))])

    Pmem = np.zeros(len(Mass))
    Pmem[~nans] = np.nansum(Pprior_sumMass * pgal[~nans] * Pcc, axis=1) / pnorm
    Pmem = np.minimum(Pmem,0.99)
    
    Pconv = np.zeros(len(Mass))
    Pconv[~nans] = np.nansum(pgal[~nans] * Pcc, axis=1) / pnorm
    Pconv = np.minimum(Pconv,0.99)
    
    #dPmem = np.zeros(len(Mass))
    #dPmem[~nans] = np.nansum(Pprior_sumMass * pgal[~nans] * Pcc, axis=1) / pnorm
    
    return Pmem, Mass_CL, Pconv #, dPmem



def get_Pmem_dPmem_smooth(field,mocktype,survey,mem_info,mask_clus_2Mpc,SNF_Mz_noclus,Omega_C_r200,Omega_F_z,SNtot_Mz_r200,pdzgal,pdzclus,clus,zz,sigz68_Mz,MM,nprocs,Mass_clus):
    start=time.time()
    nclus=len(clus)

    memo = 8*(1024**3)
    mem_avail = psutil.virtual_memory().available
        
    if memo < 0.9*mem_avail:    
        memo_obj = int(0.9*memo)
        memo_heap = max(60000000,memo - memo_obj)
        ray.init(num_cpus=nprocs,memory=memo_heap, object_store_memory=memo_obj,ignore_reinit_error=True,log_to_driver=False)
    
        mem_info_id = ray.put(mem_info)
        mask_clus_2Mpc_id = ray.put(mask_clus_2Mpc)
        SNF_Mz_noclus_id = ray.put(SNF_Mz_noclus)
        Omega_C_r200_id = ray.put(Omega_C_r200)
        Omega_F_z_id = ray.put(Omega_F_z)
        SNtot_Mz_r200_id = ray.put(SNtot_Mz_r200)
        pdzgal_id = ray.put(pdzgal)
        pdzclus_id = ray.put(pdzclus)
        zclus_id = ray.put(np.array(clus['z']))
        sigz68_Mz_id = ray.put(sigz68_Mz)
        Mass_clus_id = ray.put(Mass_clus)
                
        res = ray.get([compute_pmem_dpmem_smooth.remote(indc,field,mocktype,survey,mem_info_id,mask_clus_2Mpc_id,SNF_Mz_noclus_id,Omega_C_r200_id,Omega_F_z_id,SNtot_Mz_r200_id,pdzgal_id,pdzclus_id,zclus_id,zz,sigz68_Mz_id,MM,Mass_clus_id) for indc in range(nclus)])    
        ray.shutdown()  
        res = np.array(res)
        Pmem = res[:,0]
        Mass_CL95 = res[:,1]
        Pconv = res[:,2]
        dPmem = res[:,3]
    else:
        raise ValueError('Not enough memory available : ',memo,'<',mem_avail)
    print('done in ',time.time()-start,'s')   
        
    return Pmem, Mass_CL95, Pconv, dPmem

@ray.remote
def compute_pmem_dpmem_smooth(indc,field,mocktype,survey,mem_info_id,mask_clus_2Mpc_id,SNF_Mz_noclus_id,Omega_C_r200_id,Omega_F_z_id,SNtot_Mz_r200_id,pdzgal_id,pdzclus_id,zclus_id,zz,sigz68_Mz_id,MM,Mass_clus_id):
    warnings.simplefilter("ignore")
    print(indc)

    clus_imR200_r200 = np.copy(mem_info_id[indc,1])
    gal_imR200 = np.copy(mem_info_id[indc,2])
    f_loc_DETECTIFz = np.copy(mem_info_id[indc,3])
    f_profile_imR200 = 10**(gal_imR200-clus_imR200_r200)

    zzbin = np.linspace(0.005,5.005,501)
    MMbin = np.linspace(5,13,161)
    mask_clus_2Mpc = np.copy(mask_clus_2Mpc_id[indc])
    pgal = np.copy(pdzgal_id[mask_clus_2Mpc])
    pgal = pgal / np.sum(pgal,axis=1)[:,None]
    pclus = np.copy(pdzclus_id[indc])
    pclus = pclus / np.sum(pclus)
    inf_pclus = np.where(pclus > 0)[0][0]
    sup_pclus = np.where(pclus > 0)[0][-1]+1
    '''
    pdMzf = h5py.File('catalogues/'+field+'/'+field+'.master.PDF_Mz.irac.hdf','r')
    pdMz = pdMzf['pdf_mass_z']
    idx_pdMz = np.where(mask_clus_2Mpc)[0]
    ppM = np.sum(pdMz[idx_pdMz,inf_pclus:sup_pclus,:]*pclus[None,inf_pclus:sup_pclus,None],axis=1)
    pdMzf.close()
    
    Mass_CL = np.array([weighted_quantile(MM,[0.025,0.16,0.5,0.84,0.975],
                                            sample_weight=ppM[indg]) for indg in range(np.sum(mask_clus_2Mpc))])
    '''
    Mass_l95, Mass_l68, Mass, Mass_u68, Mass_u95 = Mass_clus_id[indc].T #Mass_CL.T #
    #Mass_l95 = Mass-0.5
    #Mass_u95 = Mass+0.5
    Mass_CL = np.c_[Mass_l95, Mass_l68, Mass, Mass_u68, Mass_u95]
    
    Masspix_l95 = np.digitize(Mass_l95,MMbin)-1 
    Masspix_u95 = np.digitize(Mass_u95,MMbin)-1 
    jc = np.digitize(zclus_id[indc],zzbin)-1
    #zclus_pdzmax = zz[np.argmax(pclus)]
    #jcc = np.digitize(zclus_pdzmax,zzbin)-1
    
    nans = np.isnan(Mass)
    ig = np.digitize(np.minimum(12,Mass[~nans]),MMbin)-1
    sigz = np.copy(sigz68_Mz_id[jc,ig])
    #print(Mass[~nans][np.isnan(sigz)])
    dzpix68_Mz = sigz / 0.01 #.astype(int)
    #Pcc = np.array([gaussian_filter1d(pclus,max(1,dzpix68_Mz[indg])) for indg in range(np.sum(~nans))])
    Pcc = np.array([gaussian_filter1d(pclus,dzpix68_Mz[indg]) for indg in range(np.sum(~nans))])
    Pcclus = np.repeat(gaussian_filter1d(pclus,1),np.sum(~nans)).reshape(len(zz),np.sum(~nans)).T 
    zclus_Pccmax = zz[np.argmax(Pcclus,axis=1)]
    jcc = np.digitize(zclus_Pccmax,zzbin)-1
    shift = jcc - np.argmax(pgal[~nans],axis=1)
    pnorm = np.sum(advindexing_roll(pgal[~nans],shift)*Pcclus,axis=1)
    
    Masspix_comp90 = np.digitize(Mlim(survey,0,zz[sup_pclus]),MMbin)-1
    
    #print('NF_R200_indc',SNF_Mz_noclus_R200clus_id[:,:,indc])
    
    NF_sumMass = np.array([np.sum((SNF_Mz_noclus_id)[
        :,max(Masspix_comp90,Masspix_l95[~nans][indg]):min(len(MM),Masspix_u95[~nans][indg])],axis=1) if Masspix_comp90 < Masspix_u95[~nans][indg] else np.repeat(0,len(zz)) for indg in range(np.sum(~nans))])

    #print('NF',NF_r200_sumMass)
    Ntot_r200_sumMass = np.array([np.sum((SNtot_Mz_r200_id[indc])[
        :,max(Masspix_comp90,Masspix_l95[~nans][indg]):min(len(MM),Masspix_u95[~nans][indg])],axis=1) if Masspix_comp90 < Masspix_u95[~nans][indg] else np.repeat(0,len(zz)) for indg in range(np.sum(~nans))])
    #print('Ntot',Ntot_r200_sumMass)
    #print('fpro',f_profile_imR200.shape)
    #print('floc',f_loc_DETECTIFz.shape)
    C = Omega_C_r200_id[indc] / Omega_F_z_id
    NF_R200_sumMass = NF_sumMass * C
    Pprior_sumMass = 1 - (f_loc_DETECTIFz*NF_R200_sumMass/(f_profile_imR200[~nans][:,None]*Ntot_r200_sumMass))
    Pprior_sumMass[np.where(Pprior_sumMass < 0)] = np.nan
    #print('Pprior',Pprior_sumMass)    
    
    fac = C * f_loc_DETECTIFz/(f_profile_imR200[~nans][:,None]*Ntot_r200_sumMass)
    d_Pprior_sumMass = fac * np.sqrt(NF_sumMass**2 + NF_sumMass)

    Pmem = np.zeros(len(Mass))
    Pmem[~nans] = np.nansum(Pprior_sumMass * pgal[~nans] * Pcc, axis=1) / pnorm
    Pmem = np.minimum(Pmem,0.99)
    
    Pconv = np.zeros(len(Mass))
    Pconv[~nans] = np.nansum(pgal[~nans] * Pcc, axis=1) / pnorm
    Pconv = np.minimum(Pconv,0.99)
    
    dPmem = np.zeros(len(Mass))
    dPmem[~nans] = np.sqrt( np.nansum((d_Pprior_sumMass * pgal[~nans] * Pcc )**2, axis=1) ) / pnorm
    dPmem = np.minimum(dPmem,0.99)
    #dPmem = np.maximum(0,1e-6)

    return Pmem, Mass_CL, Pconv , dPmem


    
    
def make_members(clus,gal,mask_clus_2Mpc,Pmem,Mass_clus,Pconv,field):
    Path('members'+field).mkdir(parents=True, exist_ok=True)
    nclus=len(clus)
    memdet_2Mpc = np.empty(nclus,dtype='object')
    clussky = SkyCoord(clus['ra'],clus['dec'],unit='deg',frame='fk5')
    galsky = SkyCoord(gal['ra'],gal['dec'],unit='deg',frame='fk5')
    for indc in range(nclus):
        memdet_2Mpc[indc] = Table(gal[mask_clus_2Mpc[indc]],copy=True)
        sep = clussky[indc].separation(galsky[mask_clus_2Mpc[indc]])
        sepMpc = physep_ang(clus['z'][indc],sep.to(u.deg).value)
        memdet_2Mpc[indc].add_column(Column(sep,name='rdeg'))
        memdet_2Mpc[indc].add_column(Column(sepMpc,name='rMpc'))
        memdet_2Mpc[indc].add_column(Column(Pmem[indc],name='Pmem'))
        #memdet_2Mpc[indc].add_column(Column(dPmem[indc],name='dPmem'))
        memdet_2Mpc[indc].add_column(Column(Pconv[indc],name='Pconv'))
        memdet_2Mpc[indc].add_column(Column(Mass_clus[indc][:,2],name='Mass_median_pdzclus'))
        memdet_2Mpc[indc].add_column(Column(Mass_clus[indc][:,0],name='Mass_l95_pdzclus'))
        memdet_2Mpc[indc].add_column(Column(Mass_clus[indc][:,1],name='Mass_l68_pdzclus'))
        memdet_2Mpc[indc].add_column(Column(Mass_clus[indc][:,3],name='Mass_u68_pdzclus'))
        memdet_2Mpc[indc].add_column(Column(Mass_clus[indc][:,4],name='Mass_u95_pdzclus'))
        
        memdet_2Mpc[indc].write('members'+field+'/members2Mpc.clus'+str(clus['id'][indc])+'.fits',overwrite=True)
    np.savez('members'+field+'.noclus.'+str(radnoclus)+'r200.sigM95.M90.smooth.Pcclus.npz',mem=memdet_2Mpc)

    return memdet_2Mpc