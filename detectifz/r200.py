import sys
import numpy as np
import numba as nb
from astropy.table import Table, Column
from pathlib import Path
from astropy.io import fits
import ray
from astropy import units
import psutil
from astropy import wcs
import scipy.signal
import astropy.cosmology

from joblib import Parallel, delayed


from .utils import angsep_radius, physep_ang, Mlim_DETECTIFz

from astropy.coordinates import SkyCoord
from photutils import CircularAperture, CircularAnnulus
from photutils import SkyCircularAnnulus, SkyCircularAperture

from astropy.wcs.utils import proj_plane_pixel_area,proj_plane_pixel_scales

import lnr

import time

import warnings
from astropy.utils.exceptions import AstropyWarning,AstropyUserWarning

def dgal_R200(inputs):
    rr,rrMpc,sepMpc,zclus,zinf,zsup,weights,w2d,galmc,galsky,Nmc,fov_area,clussky,cosmo,Mlim10_mc,zz,glob_fdens, loc_fdens,area_f_loc = inputs
    NM_f_glob, dNM_f_glob = glob_fdens
    NM_f_loc, dNM_f_loc = loc_fdens

    Nr = len(rrMpc)
    DELTA_gal = np.zeros(Nr)
    dDELTA_gal = np.zeros(Nr)
    sepgal = clussky.separation(galsky) 
    
    for i in range(Nr):
        if i>5 and DELTA_gal[i-5] < 200/cosmo.Om0 and np.sum(DELTA_gal > 200/cosmo.Om0) > 0:
            break
        aperture = SkyCircularAperture(clussky, r=rr[i])
        aperture_pix = aperture.to_pixel(w2d) 
        aperture_masks = aperture_pix.to_mask(method='center')
        aperture_data = aperture_masks.multiply(weights)
        area_notmasked_deg2 = np.sum(aperture_data) * proj_plane_pixel_area(w2d) * (units.deg**2)
                
        ## volume masked
        m = sepMpc.value/rrMpc[i] < 1
        if np.sum(m) > 0:
            clusvol_masked = np.sum( 
            (physep_ang(zclus,proj_plane_pixel_scales(w2d)[0],H0=cosmo.H0,Om0=cosmo.Om0)**2).value *(
             2*sepMpc[m].value*np.tan(np.arccos(sepMpc.value/rrMpc[i])[m]))) * (units.Mpc**3)
        else:
            clusvol_masked = 0 * (units.Mpc**3) 
        clusvol_tot = (4/3.) * np.pi * (rrMpc[i]*units.Mpc)**3
        Vclus_m = clusvol_tot - clusvol_masked
        
        mm = np.isnan(aperture_data)
        Vfield_m =  (physep_ang(zclus,proj_plane_pixel_scales(w2d)[0],H0=cosmo.H0,Om0=cosmo.Om0)**2).value * (
            np.sum(aperture_masks.data)-np.sum(aperture_masks.data[mm])) * cosmo.angular_diameter_distance_z1z2(
            zinf,zsup).value * (units.Mpc**3)

        masksep = sepgal < rr[i]
        M, z = galmc[:Nmc,masksep,4], galmc[:Nmc,masksep,3]
        z = np.minimum(np.maximum(zz[0], z), zz[-1])
        mask_Ncr = (M > Mlim10_mc[:Nmc,masksep]) & (z >= zinf) & (z < zsup)
        #NM_cr = (np.sum(10**galmc[:,masksep,4][mask_Ncr])/Nmc) - (NM_f_loc*area_notmasked_deg2/area_f_loc)
        NM_f_loc_cr = NM_f_loc*area_notmasked_deg2/area_f_loc
        NM_f_glob_cr = NM_f_glob*area_notmasked_deg2/fov_area
        Ntot_cr = np.sum(10**galmc[:Nmc,masksep,4][mask_Ncr])/Nmc
        NM_cr = Ntot_cr - NM_f_loc_cr
        DELTA_gal[i] = (NM_cr/Vclus_m) / (NM_f_glob_cr/Vfield_m)

        #dNM_totr = np.sqrt(Ntot_cr)/np.sqrt(Nmc)
        #dNM_totr = np.sqrt(np.sum(np.array([len(galmc[imc,:,4][masksep[imc]]) for imc in range(Nmc)])))/Nmc
        dNM_totr = np.std(np.array([np.sum(10**galmc[imc,masksep,4][mask_Ncr[imc]]) for imc in range(Nmc)]))
        #dNM_f_loc = np.sqrt(NM_f_loc)/np.sqrt(Nmc)
        #dNM_f_glob = np.sqrt(NM_f_glob)/np.sqrt(Nmc)
        
        dDELTA_gal[i] = (Vfield_m/Vclus_m) * np.sqrt(
                            ((dNM_totr*fov_area) / (NM_f_glob*area_notmasked_deg2) )**2 +
                            ((dNM_f_loc*fov_area) / (NM_f_glob*area_f_loc))**2 +
                            ((dNM_f_glob*(NM_f_loc_cr-Ntot_cr)) / ((NM_f_glob**2)*(fov_area/area_notmasked_deg2)))**2  )
        
        dgal = DELTA_gal[i]

    return DELTA_gal, dDELTA_gal

def dgal_R200_fine(inputs):
    rr,rrMpc,sepMpc,zclus,zinf,zsup,weights,w2d,galmc,galsky,Nmc,fov_area,clussky,cosmo,Mlim10_mc,zz,glob_fdens, loc_fdens,area_f_loc = inputs
    NM_f_glob, dNM_f_glob = glob_fdens
    NM_f_loc, dNM_f_loc = loc_fdens
    Nr = len(rrMpc)
    DELTA_gal = np.zeros(Nr)
    dDELTA_gal = np.zeros(Nr)
    sepgal = clussky.separation(galsky) 

    for i in range(Nr):
        aperture = SkyCircularAperture(clussky, r=rr[i])
        aperture_pix = aperture.to_pixel(w2d) 
        aperture_masks = aperture_pix.to_mask(method='center')
        aperture_data = aperture_masks.multiply(weights)
        area_notmasked_deg2 = np.sum(aperture_data) * proj_plane_pixel_area(w2d) * (units.deg**2)
                
        ## volume masked
        m = sepMpc.value/rrMpc[i] < 1
        if np.sum(m) > 0:
            clusvol_masked = np.sum( 
            (physep_ang(zclus,proj_plane_pixel_scales(w2d)[0],H0=cosmo.H0,Om0=cosmo.Om0)**2).value *(
             2*sepMpc[m].value*np.tan(np.arccos(sepMpc.value/rrMpc[i])[m]))) * (units.Mpc**3)
        else:
            clusvol_masked = 0 * (units.Mpc**3) 
        clusvol_tot = (4/3.) * np.pi * (rrMpc[i]*units.Mpc)**3
        Vclus_m = clusvol_tot - clusvol_masked
        
        mm = np.isnan(aperture_data)
        Vfield_m =  (physep_ang(zclus,proj_plane_pixel_scales(w2d)[0],H0=cosmo.H0,Om0=cosmo.Om0)**2).value * (
            np.sum(aperture_masks.data)-np.sum(aperture_masks.data[mm])) * cosmo.angular_diameter_distance_z1z2(
            zinf,zsup).value * (units.Mpc**3)

        masksep = sepgal < rr[i]
        M, z = galmc[:Nmc,masksep,4], galmc[:Nmc,masksep,3]
        z = np.minimum(np.maximum(zz[0], z), zz[-1])
        mask_Ncr = (M >  Mlim10_mc[:Nmc,masksep]) & (z >= zinf) & (z < zsup)
        #print(mask_Ncr.shape)
        #NM_cr = (np.sum(10**galmc[:,masksep,4][mask_Ncr])/Nmc) - (NM_f_loc*area_notmasked_deg2/area_f_loc)
        NM_f_loc_cr = NM_f_loc*area_notmasked_deg2/area_f_loc
        NM_f_glob_cr = NM_f_glob*area_notmasked_deg2/fov_area
        Ntot_cr = np.sum(10**galmc[:Nmc,masksep,4][mask_Ncr])/Nmc
        NM_cr = Ntot_cr - NM_f_loc_cr
        DELTA_gal[i] = (NM_cr/Vclus_m) / (NM_f_glob_cr/Vfield_m)

        #dNM_totr = np.sqrt(np.sum(np.array([len(galmc[imc,masksep,4][mask_Ncr[imc]]) for imc in range(Nmc)])))/Nmc
        dNM_totr = np.std(np.array([np.sum(10**galmc[imc,masksep,4][mask_Ncr[imc]]) for imc in range(Nmc)]))
        #print(dNM_f_glob,dNM_f_loc,dNM_totr)
        #dNM_totr = np.sqrt(Ntot_cr)/np.sqrt(Nmc)
        #dNM_f_loc = np.sqrt(NM_f_loc)/np.sqrt(Nmc)
        #dNM_f_glob = np.sqrt(NM_f_glob)/np.sqrt(Nmc)
        
        dDELTA_gal[i] = (Vfield_m/Vclus_m) * np.sqrt(
                            ((dNM_totr*fov_area) / (NM_f_glob*area_notmasked_deg2) )**2 +
                            ((dNM_f_loc*fov_area) / (NM_f_glob*area_f_loc))**2 +
                            ((dNM_f_glob*(NM_f_loc_cr-Ntot_cr)) / ((NM_f_glob**2)*(fov_area/area_notmasked_deg2)))**2  )
        
        dgal = DELTA_gal[i]
        
    return DELTA_gal, dDELTA_gal

def R200fit_bces(indc,DELTA_gal,dDELTA_gal,rrMpc,d200):
    #print('start fit',indc)
    x_bces= lambda A, B, y: (y/10**A)**(1/B)      
    negs = (DELTA_gal <= 0) | np.isinf(DELTA_gal) | np.isnan(DELTA_gal)
    if np.sum(~negs) > 0:
        idxpeak = np.argmax(DELTA_gal[~negs])
        x1=np.log10(rrMpc[~negs])[idxpeak:]
        x2=np.log10(DELTA_gal[~negs])[idxpeak:]
        x1err=np.repeat(0.5*(rrMpc[1]-rrMpc[0]),rrMpc.shape[0])[~negs][idxpeak:]/(10**x1*np.log(10))
        x2err=dDELTA_gal[~negs][idxpeak:]/(10**x2*np.log(10))
        cov=np.zeros((len(x1),len(x2)))
        try:
            res = lnr.bces(x1,x2,
               x1err=x1err,
               x2err=x2err,
               logify=False,verbose='quiet',bootstrap=1000)       
        except:
            print('bootstrap failed at ',indc,' we revert to analytical error')
            res = lnr.bces(x1,x2,
               x1err=x1err,
               x2err=x2err,
               logify=False,verbose='quiet',bootstrap=False)               
        a=res[0][0]
        b=res[1][0]
        a_err=res[0][1]
        b_err=res[1][1] 
        '''
        b,a,b_err,a_err,covab=bces.bces.bcesp(x1,x1err,x2,x2err,cov,nsim=1000,ncores=1)
        a=a[0]
        b=b[0]
        a_err=a_err[0]
        b_err=b_err[0] 
        '''
        r200best = x_bces(a,b,d200)
        a_err = np.array([a_err]).flatten()
        b_err = np.array([b_err]).flatten()
        if a_err.size == 1:
            a_err = [a_err, a_err]
        if b_err.size == 1:
            b_err = [b_err, b_err]
        err = [x_bces(a-a_err[0], b-b_err[0],d200), x_bces(a-a_err[0], b+b_err[1],d200),
            x_bces(a+a_err[1], b-b_err[0],d200), x_bces(a+a_err[1], b+b_err[1],d200)]
        r200inf = np.min(err, axis=0)
        r200sup = np.max(err, axis=0)
        r200inf = r200inf[0]
        r200sup = r200sup[0]
        
        if r200best > 2:
            return np.array([-1,-1,-1])
        if r200best > 0 and r200best < 2 and r200sup > 2:
            r200sup = 2
        if r200best > 0 and r200best < 2 and r200inf < 0:
            r200inf = 0
        return np.array([r200inf,r200best,r200sup])
    else:
        return np.array([-1,-1,-1])

    
def field_mass_density(inputs):    
    rr,rrMpc,sepMpc,zclus,zinf,zsup,weights,w2d,galmc,galsky,Nmc,fov_area,clussky,cosmo,Mlim10_mc,zz = inputs
    Nr = len(rrMpc)
    DELTA_gal = np.zeros(Nr)
    dDELTA_gal = np.zeros(Nr)

    ##get area of local field
    aperture_f_loc = SkyCircularAnnulus(clussky, r_in=angsep_radius(zclus,3,H0=cosmo.H0,Om0=cosmo.Om0),
                                        r_out=angsep_radius(zclus,5,H0=cosmo.H0,Om0=cosmo.Om0))
    aperture_f_loc_pix = aperture_f_loc.to_pixel(w2d) 
    aperture_f_loc_masks = aperture_f_loc_pix.to_mask(method='center')
    aperture_f_loc_data = aperture_f_loc_masks.multiply(weights)
    area_f_loc = np.sum(aperture_f_loc_data) * proj_plane_pixel_area(w2d) * (units.deg**2)
    
    ##get mass density of global field
    M, z = galmc[:Nmc,:,4], galmc[:Nmc,:,3]
    #z = np.minimum(np.maximum(zz[0], z), zz[-1])
    mask_Nf_glob = (M > Mlim10_mc[:Nmc]) & (z >= zinf) & (z < zsup)
    #NM_f_glob = np.sum(10**galmc[:,:,4][mask_Nf_glob])/Nmc
    NM_f_glob = np.sum(10**galmc[:Nmc,:,4][mask_Nf_glob])/Nmc
    #dNM_f_glob = np.sqrt(np.sum(np.array([len(galmc[imc,:,4][mask_Nf_glob[imc]]) for imc in range(Nmc)])))/Nmc
    dNM_f_glob = np.std(np.array([np.sum(10**galmc[imc,:,4][mask_Nf_glob[imc]]) for imc in range(Nmc)]))
    glob = [NM_f_glob, dNM_f_glob]

    
    #start_glob = time.time()
    #M, z = galmc[:Nmc,:,4], galmc[:Nmc,:,3]
    #glob = get_numdens(M, z, Mlim10_mc, zinf, zsup, Nmc)
    #end_glob = time.time()
    #time_glob_numba = end_glob - start_glob
    
    ##get mass density of local field
    sepgal = clussky.separation(galsky)   
    masksep_Nf_loc = (sepgal > angsep_radius(zclus,3,H0=cosmo.H0,Om0=cosmo.Om0)) & (sepgal < angsep_radius(zclus,5,H0=cosmo.H0,Om0=cosmo.Om0))
    M, z = galmc[:Nmc,masksep_Nf_loc,4], galmc[:Nmc,masksep_Nf_loc,3]
    z = np.minimum(np.maximum(zz[0], z), zz[-1])
    mask_Nf_loc = (M > Mlim10_mc[:Nmc,masksep_Nf_loc]) & (z >= zinf) & (z < zsup)
    #NM_f_loc = np.sum(10**galmc[:,masksep_Nf_loc,4][mask_Nf_loc])/Nmc
    NM_f_loc = np.sum(10**galmc[:Nmc,masksep_Nf_loc,4][mask_Nf_loc])/Nmc
    #dNM_f_loc = np.sqrt(np.sum(np.array([len(galmc[imc,masksep_Nf_loc,4][mask_Nf_loc[imc]]) for imc in range(Nmc)])))/Nmc
    dNM_f_loc = np.std(np.array([np.sum(10**galmc[imc,masksep_Nf_loc,4][mask_Nf_loc[imc]]) for imc in range(Nmc)]))

    loc = [NM_f_loc, dNM_f_loc]
    
    return glob, loc, area_f_loc    


@nb.njit()
def get_numdens(M, z, Mlim10_mc, zinf, zsup, Nmc):
    #mask_Nf_glob = (M > Mlim10_mc) & (z >= zinf) & (z < zsup)
    N_f_glob_mc = np.zeros(Nmc)
    for imc in range(Nmc):
        Mimc, zimc = M[imc], z[imc]
        Mmask = (Mimc > Mlim10_mc[imc])
        zmask = ( (zimc >= zinf) & 
                 (zimc < zsup) )
        mask_Nf_glob = Mmask & zmask
        
        Mimc = Mimc[mask_Nf_glob]
        N_f_glob_mc[imc] = np.sum(10**Mimc)
        
    NM_f_glob = np.mean(N_f_glob_mc)
    dNM_f_glob = np.std(N_f_glob_mc)

    return NM_f_glob, dNM_f_glob


#@ray.remote(max_calls=50)
def R200(indc,clus_id,colnames,weights_id,head2d,pdzclus_id,sigz68_z_id,galmc,Nmc,rrMpc,Mlim10_mc_id,zz):
    #print('start',indc)
    H0_fid = 70.
    Om0_fid=0.3
    H0_mock = 73. #H0_fid
    Om0_mock = Om0_fid
    rrMpc_coarse = rrMpc[::2]
    warnings.simplefilter("ignore")

    clus = clus_id #Table(clus_id,names=colnames)
    raclus=clus['ra'][indc] 
    decclus=clus['dec'][indc] 
    zclus=clus['z'][indc]
    
    if not(indc%int(len(clus)/(100/5))):
        print(int(100*indc/len(clus)),'% completed')
    
    w2d = wcs.WCS(head2d)
    weights = np.copy(weights_id)

    #zzbin = np.linspace(0.005,5.005,501)
    dz = zz[1]-zz[0]
    zzbin = np.linspace(zz[0]-dz/2, zz[-1]+dz/2, len(zz)+1)
    jc = np.digitize(zclus,zzbin)-1
    sigz68_z = np.copy(sigz68_z_id[jc])
    pdzclus = np.copy(pdzclus_id[indc])
    dzpix68_z = (sigz68_z / 0.01).astype(int)
    win = scipy.signal.gaussian(M=501,std=max(2,dzpix68_z))
    Pcc = scipy.signal.convolve(pdzclus,win, mode='same')/sum(win)
    rv = scipy.stats.rv_histogram((Pcc,zzbin))
    zinf, zsup = rv.interval(0.95)
    #print(zinf,zsup)
    
    cosmo=astropy.cosmology.FlatLambdaCDM(H0=H0_mock,Om0=Om0_mock)
    d200=200./cosmo.Om0
    #cosmo=astropy.cosmology.FlatLambdaCDM(H0=73.,Om0=0.25)
    rr = angsep_radius(zclus,rrMpc,H0=H0_mock,Om0=Om0_mock)
    Nr = len(rr)
    rr_coarse = rr[::2]
    #Nr_coarse = len(rr_coarse)
    
    clussky = SkyCoord(raclus, decclus, unit='deg', frame='fk5') 
    #DELTA_gal = np.zeros(Nr)
    
    ## pre-process for volume mask computation
    aperture = SkyCircularAperture(clussky, r=rr[-1])
    aperture_pix = aperture.to_pixel(w2d) 
    aperture_masks = aperture_pix.to_mask(method='center')
    aperture_data = aperture_masks.multiply(weights)
    xx = np.where(aperture_data==0)[0]
    yy = np.where(aperture_data==0)[1]
    pixaper = np.c_[yy,xx] - np.array(aperture_data.shape)/2
    clusradec = np.c_[raclus,decclus]
    cluspix = w2d.wcs_world2pix(clusradec,0) 
    radec = w2d.wcs_pix2world(pixaper+cluspix,0)
    cluscat = SkyCoord(ra=clusradec[:,0]*units.degree, dec=clusradec[:,1]*units.degree, frame='fk5')
    mcat = SkyCoord(ra=radec[:,0]*units.degree, dec=radec[:,1]*units.degree, frame='fk5')
    sep = cluscat.separation(mcat)
    sepMpc = physep_ang(zclus,sep.value,H0=H0_mock,Om0=Om0_mock)

    ragal = galmc[0,:,1]
    decgal = galmc[0,:,2]
    galsky = SkyCoord(ra=ragal, dec=decgal, unit='deg',frame='fk5')
    fov_area = np.sum(weights) * proj_plane_pixel_area(w2d) *units.deg**2
    
    inputs = [rr_coarse,rrMpc_coarse,sepMpc,zclus,zinf,zsup,weights,w2d,
              galmc,galsky,Nmc,fov_area,clussky,cosmo,Mlim10_mc_id,zz]
    glob_fdens, loc_fdens, area_f_loc = field_mass_density(inputs)

    inputs = [rr_coarse,rrMpc_coarse,sepMpc,zclus,zinf,zsup,weights,w2d,
              galmc,galsky,Nmc,fov_area,clussky,cosmo,Mlim10_mc_id,zz,
              glob_fdens, loc_fdens, area_f_loc]
    DELTA_gal_coarse,dDELTA_gal_coarse = dgal_R200(inputs)
    #print(indc,'deltaM_coarse OK')

    rrMpc_fine = np.zeros(20)
    DELTA_gal_fine = np.zeros(20)
    dDELTA_gal_fine = np.zeros(20)
    if len(np.where(DELTA_gal_coarse >= 200/cosmo.Om0)[0]) > 0:
        indr200c_Om0_coarse = np.max(np.where(DELTA_gal_coarse >= 200/cosmo.Om0)[0])
        idxr = np.where(rrMpc == rrMpc_coarse[indr200c_Om0_coarse])[0][0]
        rrMpc_fine = rrMpc[max(0,idxr-5):min(idxr+6,Nr)]
        rr_fine = rr[max(0,idxr-5):min(idxr+6,Nr)]
        inputs_fine = [rr_fine,rrMpc_fine,sepMpc,zclus,zinf,zsup,weights,w2d,
                       galmc,galsky,Nmc,fov_area,clussky,cosmo,Mlim10_mc_id,zz,
                       glob_fdens, loc_fdens, area_f_loc]

        DELTA_gal_fine,dDELTA_gal_fine = dgal_R200_fine(inputs_fine)
        #print(indc,'deltaM_fine OK')
        
        start_fit = time.time()
        if len(np.where(DELTA_gal_fine >= 200/cosmo.Om0)[0]) > 0:
            isfine=True
            indr200c_Om0_fine = np.max(np.where(DELTA_gal_fine >= 200/cosmo.Om0)[0])
            r200c_Om0 = rrMpc_fine[indr200c_Om0_fine]*H0_mock/H0_fid 
            r200fit = R200fit_bces(indc,DELTA_gal_fine,dDELTA_gal_fine,rrMpc_fine,d200)*H0_mock/H0_fid 
        else:
            isfine=False
            r200c_Om0 = rrMpc_coarse[indr200c_Om0_coarse]*H0_mock/H0_fid 
            r200fit = R200fit_bces(indc,DELTA_gal_coarse,dDELTA_gal_coarse,rrMpc_coarse,d200)*H0_mock/H0_fid 
    else:
        isfine=False
        r200c_Om0=-1
        r200fit = np.array([-1,-1,-1])
    
    return r200c_Om0, DELTA_gal_coarse, dDELTA_gal_coarse, DELTA_gal_fine, dDELTA_gal_fine, rrMpc_fine , r200fit, isfine  

        
def get_R200(detectifz):
        
    clusf = ( detectifz.config.rootdir+'/candidats_'+detectifz.field+
             '_SN'+str(detectifz.config.SNmin)+
             '_Mlim'+str(np.round(detectifz.config.lgmass_lim,2))+
             '.sigz68_z_'+detectifz.config.avg+'.r200.fits' )
    
    if Path(clusf).is_file():
        print('already saved, we read it')
        clus_r200 = Table.read(clusf)
    else:    
        start = time.time()
        rrMpc = np.arange(0.1,2,0.05)
        nclus = len(detectifz.clus)

        
        zmc = np.minimum(np.maximum(detectifz.data.zz[0], 
                                    detectifz.data.galcat_mc[:,:,3]), 
                         detectifz.data.zz[-1])
        
        Mlim10_mc = Mlim_DETECTIFz(detectifz.data.logMlim90,10,zmc)
        
        memo = 1.8*1024**3 #1.5 * (im3d.nbytes + weights.nbytes)
        mem_avail = 10*1024**3 #psutil.virtual_memory().available
        '''
        if memo < 0.9*mem_avail:    
            memo_obj = int(0.9*memo)
            ray.init(num_cpus=detectifz.config.nprocs, 
                     object_store_memory=memo_obj,ignore_reinit_error=True,log_to_driver=False)
    
            weights_id = ray.put(detectifz.weights2d)  
            clus_id = ray.put(detectifz.clus.to_pandas().to_numpy())
            pdzclus_id = ray.put(detectifz.pdzclus)
            sigz68_z_id = ray.put(detectifz.data.sigs.sigz68_z)
            galmc_id = ray.put(detectifz.data.galcat_mc)
            Mlim10_mc_id = ray.put(Mlim10_mc)

            #try:
            #Nmc = 10  #work on 10 realisation only for speed
            res = ray.get([R200.remote(indc, clus_id, detectifz.clus.colnames, 
                                       weights_id, detectifz.head2d, pdzclus_id, 
                                       sigz68_z_id, galmc_id, detectifz.config.Nmc, rrMpc, 
                                       Mlim10_mc_id, detectifz.data.zz) 
                           for indc in range(nclus)],timeout=5000)  

            ray.shutdown()
            res = np.array(res)
        '''
        
        res = np.array(Parallel(n_jobs=int(1 * detectifz.config.nprocs), max_nbytes=1e6)(
            delayed(R200)(
                indc, 
                detectifz.clus, 
                detectifz.clus.colnames, 
                detectifz.weights2d, 
                detectifz.head2d, 
                detectifz.pdzclus, 
                detectifz.data.sigs.sigz68_z, 
                detectifz.data.galcat_mc, 
                detectifz.config.Nmc, 
                rrMpc, 
                Mlim10_mc, 
                detectifz.data.zz)
            for indc in range(nclus)))
        
        r200c = np.stack(res[:,0]) 
        DELTA_gal_coarse = res[:,1]
        dDELTA_gal_coarse = res[:,2]
        DELTA_gal_fine = res[:,3]
        dDELTA_gal_fine = res[:,4]
        rrMpc_fine = res[:,5]
        r200fit = np.stack(res[:,6])
        isfine = res[:,7]

        np.savez('dM_'+detectifz.field+'.r200.npz',r200c=r200c, 
                 DELTA_gal_coarse=DELTA_gal_coarse, 
                 dDELTA_gal_coarse=dDELTA_gal_coarse, 
                 DELTA_gal_fine=DELTA_gal_fine, 
                 dDELTA_gal_fine=dDELTA_gal_fine,
                 rrMpc_fine=rrMpc_fine,
                 r200fit=r200fit,
                 isfine=isfine)

        clus_r200 = Table(detectifz.clus,names=detectifz.clus.colnames,copy=True)
        clus_r200.add_column(Column(r200c,name='R200c_Mass_nofit'))
        clus_r200.add_column(Column(r200fit[:,1],name='R200c_Mass_median'))
        clus_r200.add_column(Column(r200fit[:,0],name='R200c_Mass_l68'))
        clus_r200.add_column(Column(r200fit[:,2],name='R200c_Mass_u68'))
        clus_r200.write(clusf,overwrite=True)
        print('R200 done in', time.time()-start,'s')
        '''
        else:
            raise ValueError('Not enough memory available : ',memo,'<',mem_avail)
        '''    
             
    return clus_r200 