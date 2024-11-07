import numpy as np
import numba as nb
from astropy.table import Table, Column
import h5py
from pathlib import Path
from astropy.io import fits
import ray

import scipy.stats
import qp
from twopiece.scale import tpnorm
from .utils import weighted_quantile, radec2detectifz, detectifz2radec, numba_loop_kde

from collections import namedtuple

import scipy.interpolate

import time

from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy import wcs

import subprocess
from scipy.ndimage.filters import gaussian_filter1d

#@nb.njit(parallel=True)
def quantile_sig_mc(sig_indiv, binz_MC, Nz, Nzmin, Nzmax, binM_MC, NM, NMmin, NMmax, 
                    idx_lgmass_lim, quantile, Nmc):
    mask_Mlim = (binM_MC >= idx_lgmass_lim)
    sig_Mz = np.zeros((Nz,NM))
    sig_z = np.zeros(Nz)
    for iz in nb.prange(Nzmin, Nzmax):
        for iMC in range(Nmc):
            mask_z = (binz_MC[iMC] == iz)
            m = mask_z & mask_Mlim[iMC]
            s = sig_indiv[m]
            try:
                sig_z[iz] = np.quantile(s, quantile)
            except:
                continue
            for jM in range(NMmin, NMmax):
                mask_M = (binM_MC[iMC] == jM)
                m = mask_z & mask_M
                s = sig_indiv[m]
                try:
                    sig_Mz[iz, jM] = np.quantile(s, quantile) 
                except:
                    continue
    return sig_Mz, sig_z



Sigs = namedtuple('Sigs', 'sigz68_Mz sigz95_Mz sigz68_z sigz95_z sigz0')


class Data(object):
    
    def __init__(self, config=None, tile_id=None):
        self.config = config
        
        self.field = config.field
        self.rootdir = config.rootdir
        print(self.rootdir)
        self.Nmc = config.Nmc
        self.lgmass_lim = config.lgmass_lim
        print(self.lgmass_lim)
        self.masksfile_radec = self.rootdir+'/masks.'+self.field+'.radec.fits'
        self.masksfile = self.rootdir+'/masks.'+self.field+'.radec.fits'   ## to modify for coord cahnge of masks
        
        self.tile_id = tile_id        
        
        ## get maglim mask
        self.get_galcat()
        
        ###get 1D PDFs (z and M)
        self.zz = np.arange(config.zmin_pdf, config.zmax_pdf+config.pdz_dz, config.pdz_dz)
        self.MM = np.arange(config.Mmin_pdf, config.Mmax_pdf+config.pdM_dM, config.pdM_dM)

        
        galcat_mc, xyminmax = self.get_data_detectifz()
        #self.galcat = galcat
        self.galcat_mc = galcat_mc
        self.galcat_mc_master = np.vstack(self.galcat_mc)
        self.xyminmax = xyminmax
        
        self.masks = self.get_masks()
        
        print('get sigz ...')
        start = time.time()
        sigz68_Mz,sigz95_Mz,sigz68_z,sigz95_z,sigz0 = self.get_sig()     
        self.sigs = Sigs(sigz68_Mz, sigz95_Mz, sigz68_z, sigz95_z, sigz0)
        print('done in ', time.time()-start,'s')
        
        print('get Mlim 90%')
        if not (self.config.selection == 'masslim'):
            self.compute_Mlim()
        
    def get_galcat(self):
        #galcat and pdz inputs
        gal0 = Table.read(self.rootdir+'/galaxies.'+self.field+'.galcat.fits')
        gal = gal0.filled(-99)
        
        gal.rename_columns([self.config.gal_id_colname,
                            self.config.ra_colname,self.config.dec_colname,
                            self.config.z_med_colname, self.config.z_l68_colname, self.config.z_u68_colname,
                            self.config.M_med_colname,self.config.M_l68_colname, self.config.M_u68_colname,
                            self.config.obsmag_colname],
                           ['id',
                            'ra_original','dec_original',
                            #'ra','dec',
                            'z', 'z_l68', 'z_u68',
                           'Mass_median','Mass_l68','Mass_u68',
                           'obsmag'])
        
        try: # self.config.obsmag_lim is None:
            self.obsmag_lim = self.config.obsmag_lim
        except:
            try:
                h, bin_edges = np.histogram(gal['obsmag'], bins = np.arange(10,30,0.1))
                self.obsmag_lim = bin_edges[np.argmax(h)-1]                
            except:
                raise ValueError('obsmag_lim not defined and could not be computed from galaxy catalogue')
        #m90 = 22.
        self.mask_mlim = gal['obsmag'] < self.obsmag_lim
        self.galcat = gal[self.mask_mlim]
        
        ## TO DO -- coord_change that then works with mask
        self.skycoords_center = SkyCoord(ra = np.median(self.galcat['ra_original']), 
                                                        dec = np.median(self.galcat['dec_original']), unit='deg', frame='icrs')
         
        np.savez(self.rootdir+'skycoords_center.npz', 
                 ra=self.skycoords_center.ra.value, 
                 dec=self.skycoords_center.dec.value)
            
        skycoords_galaxies  = SkyCoord(ra = self.galcat['ra_original'], 
                                                        dec = self.galcat['dec_original'], unit='deg', frame='icrs')
         
        ra_detectifz, dec_detectifz = radec2detectifz(self.skycoords_center, skycoords_galaxies)
        self.galcat.add_column(Column(ra_detectifz, name='ra'))
        self.galcat.add_column(Column(dec_detectifz, name='dec'))
        #self.galcat.rename_columns(['ra_original', 'dec_original'], ['ra', 'dec'])
        
        self.galcat.write(self.rootdir+'/galaxies.'+self.field+'.galcat.detectifz.fits', overwrite=True)
        
    
    def get_data_detectifz(self):
        '''
        read galaxy catalogue and HDF_PDF_Mz file and draw 100 realization from it
    
        Save an array containing the 100 MC galaxy catalogues with (id,ra,dec,z,M*) to .npz file
        '''
                
        Mzf = self.rootdir+'/galaxies.'+self.field+'.'+str(self.Nmc)+'MC.Mz.fits'
        Mzf_masked =  self.rootdir+'/galaxies.'+self.field+'.'+str(self.Nmc)+'MC.Mz.masked_m90.fits'
        #print(Mzf)
        if Path(Mzf_masked).is_file():
            print('samples already saved, we read it')
            #Mz_t = np.load(Mzf_masked)['Mz']
            tMz = Table.read(Mzf_masked)
            if not('lgM_SAMPLES' in tMz.colnames):
                #tMz['lgM_SAMPLES'] = -99 * np.ones((len(t_Mz), self.Nmc))
                tMz['lgM_SAMPLES'] = np.repeat(self.galcat['Mass_median'], 
                                               self.Nmc).reshape((len(t_Mz), self.Nmc))
                self.has_lgM = False
        elif Path(Mzf).is_file() and not(Path(Mzf_masked).is_file()):
            print('samples already saved, we read it')
            #Mz = np.load(Mzf)['Mz'][self.mask_mlim]
            tMz = Table.read(Mzf)[self.mask_mlim]
            if not('lgM_SAMPLES' in tMz.colnames):
                #tMz['lgM_SAMPLES'] = -99 * np.ones((len(t_Mz), self.Nmc))
                tMz['lgM_SAMPLES'] = np.repeat(self.galcat['Mass_median'][self.mask_mlim], 
                                               self.Nmc).reshape((len(tMz), self.Nmc))
                self.has_lgM = False
        else:    
            print('sample (M,z)...')
            Mz = self.sample_pdf()
            tMz = Table(np.moveaxis(Mz, 2, 1), names=['lgM_SAMPLES', 'Z_SAMPLES'])
            tMz.write(Mzf_masked)
            #np.savez(Mzf_masked,Mz=Mz)
            #Mz = Mz[self.mask_mlim]
        
        
        idmc = np.repeat(np.array(self.galcat['id']),self.Nmc).reshape(len(self.galcat),self.Nmc)
        ramc = np.repeat(np.array(self.galcat['ra']),self.Nmc).reshape(len(self.galcat),self.Nmc)
        decmc = np.repeat(np.array(self.galcat['dec']),self.Nmc).reshape(len(self.galcat),self.Nmc)
        zmc = tMz['Z_SAMPLES']
        Mmc = tMz['lgM_SAMPLES']
    
        galmc = np.stack([idmc,ramc,decmc,zmc,Mmc]).T
        
        xyminmax = np.array([self.galcat['ra'].min(),self.galcat['ra'].max(),
                             self.galcat['dec'].min(),self.galcat['dec'].max()])
    
        return galmc, xyminmax


    def get_masks(self):
        ### convert radec mask (FITS with wcs) to DETECTIFz coordinates mask
        masks_radec = fits.open(self.masksfile)[0]
        
        x = np.arange(masks_radec.header['NAXIS1'])
        y = np.arange(masks_radec.header['NAXIS2'])
        X, Y = np.meshgrid(x, y)
        (ra_masks_radec, 
         dec_masks_radec) = wcs.WCS(masks_radec).wcs_pix2world(X, Y, 0)
        
        
        ra_c = float(np.load(self.rootdir+'skycoords_center.npz')['ra'])
        dec_c = float(np.load(self.rootdir+'skycoords_center.npz')['dec'])

        #skycoord_center = SkyCoord(ra=np.median(ra_masks_radec), 
        #                           dec=np.median(dec_masks_radec), 
        #                           unit='deg', frame='icrs')
        skycoord_center = SkyCoord(ra=ra_c, 
                                   dec=dec_c, 
                                  unit='deg', frame='icrs')
        
        x_detectifz = np.zeros_like(ra_masks_radec)
        y_detectifz = np.zeros_like(dec_masks_radec)

        for i in range(masks_radec.header['NAXIS2']):
            skycoord_galaxies = SkyCoord(ra=ra_masks_radec[i], 
                                         dec=dec_masks_radec[i], 
                                         unit='deg', frame='icrs')
            (x_detectifz[i], 
             y_detectifz[i]) = radec2detectifz(skycoord_center, skycoord_galaxies)
            
        xmin, xmax = x_detectifz.min(), x_detectifz.max()
        ymin, ymax = y_detectifz.min(), y_detectifz.max()
        
        dlim = 0.0
        
        # first define grid limits and size

        xsize = (((xmax + dlim) - (xmin - dlim)) / self.config.pixdeg).astype(int)
        ysize = (((ymax + dlim) - (ymin - dlim)) / self.config.pixdeg).astype(int)

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [1, 1]
        w.wcs.cdelt = np.array([self.config.pixdeg, self.config.pixdeg])
        w.wcs.crval = [xmin - dlim, ymin - dlim]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        headmasks_detectifz = w.to_header()
        headmasks_detectifz.set("NAXIS", 2)
        headmasks_detectifz.set("NAXIS1", xsize)
        headmasks_detectifz.set("NAXIS2", ysize)
        
        x = np.arange(headmasks_detectifz['NAXIS1'])
        y = np.arange(headmasks_detectifz['NAXIS2'])
        X, Y = np.meshgrid(x, y)
        x_masks_detectifz, y_masks_detectifz = wcs.WCS(headmasks_detectifz).wcs_pix2world(X, Y, 0)
        
        ra_masks_detectifz =  np.zeros_like(x_masks_detectifz)
        dec_masks_detectifz =  np.zeros_like(y_masks_detectifz)

        for i in range(headmasks_detectifz['NAXIS2']):
            (ra_masks_detectifz[i], 
             dec_masks_detectifz[i]) = detectifz2radec(skycoord_center, 
                                                       (x_masks_detectifz[i], 
                                                        y_masks_detectifz[i]))
        
        coords = SkyCoord(ra = ra_masks_detectifz.flatten(),
                 dec = dec_masks_detectifz.flatten(), unit = 'deg', frame='icrs')
        pix = wcs.utils.skycoord_to_pixel(coords, wcs = wcs.WCS(masks_radec.header))
        pixcat = Table()
        pixcat['xpix_mask'] = pix[0]
        pixcat['ypix_mask'] = pix[1]
        pixcat.write(self.rootdir+'/pixcat.fits', overwrite=True)
                
        try:
            process = subprocess.Popen(["rm", self.rootdir+"pixcat_mask."+self.field+".tmp.fits"])
            process.wait()
            result = process.communicate()
        except:
            pass
        
        process = subprocess.Popen(["venice", "-cat", 
                                    self.rootdir+"/pixcat.fits", 
                                    "-catfmt", 
                                    "fits", 
                                    "-xcol", 
                                    "xpix_mask", 
                                    "-ycol", 
                                    "ypix_mask", 
                                    "-m", 
                                    self.rootdir+"/masks."+self.field+".radec.fits", 
                                    "-o", 
                                    self.rootdir+"/pixcat_mask."+self.field+".tmp.fits"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
        process.wait()
        result = process.communicate()
        #print(result)
        
        pixcat_masks = Table.read(self.rootdir+'/pixcat_mask.'+self.field+'.tmp.fits')
        
        hdu = fits.PrimaryHDU(np.array(pixcat_masks['flag']).reshape(
            (headmasks_detectifz['NAXIS2'],
             headmasks_detectifz['NAXIS1'])),
                              header=headmasks_detectifz)
        hdu.writeto(self.rootdir+'/masks.'+self.field+'.detectifz.fits', 
                    overwrite=True)
        
        
        try:
            process = subprocess.Popen(["rm", 
                                        self.rootdir+"pixcat.fits", 
                                        self.rootdir+"pixcat_mask."+self.field+".tmp.fits"])
            process.wait()
            result = process.communicate()
        except:
            pass        
        

        return hdu
    
    
    def sample_pdf(self):
        '''
        sample PDF. -- need to consider the different cases accepted by detectifz
        ## 1. PDF(M,z)
        ## 2. PDF(z) + PDF(M) (these can be normal or asymetric half-normal if necessary)
        ## any modern survey gives at least z +/- ez and M +/- eM so its sufficient.
        ## we DO NOT consider the case PDF(z) + M(z) +/- eM(z) that almost never shows up
        '''
        
        if self.Mz_MC_exists:
            
            Mz = np.load( self.rootdir+"/galaxies."+self.field+"."+
                     str(int(self.Nmc))+"MC.Mz.npz" )['Mz']
        
        else:
            
            if self.pdf_Mz: ##only for backward compatibility, but photo-z code should return samples
            
                start = time.time()
                pdMzf = h5py.File(self.pdf_file,'r')
                zz = pdMzf['z'][()]
                MM = pdMzf['mass_bins'][()]
                ext = [zz[0],zz[-1],MM[0],MM[-1]]

                Mz = np.zeros((len(gal),self.Nmc,2))

                for i in range(np.ceil(len(gal)/1000).astype(int)):
                    print('chuck',i)
                    idxi = np.arange(1000*(i),np.minimum(1000*(i+1),len(gal)))
                    pdMz = pdMzf['pdf_mass_z'][idxi][()]
    
                    memo = 1.5*(pdMz.nbytes + Mz.nbytes)
                    mem_avail = psutil.virtual_memory().available    
                    if memo < 0.9*mem_avail:  
                        memo_obj = int(0.9*memo)
                        memo_heap = np.maximum(60000000,(memo - memo_obj))
                        ray.init(num_cpus=int(1*nprocs),memory=memo_heap,
                             object_store_memory=memo_obj,ignore_reinit_error=True,log_to_driver=False)
                        pdMz_id = ray.put(pdMz) 
    
                        Mz[idxi[0]:idxi[-1]+1] = np.array(ray.get([
                            MCsampling.remote(i,pdMz_id,self.Nmc,ext) for i,_ in enumerate(idxi)]))
        
                        ray.shutdown() 
                        #np.savez(Mzf,Mz=Mz)
                    else:
                        raise ValueError('Not enough memory available : ',memo,'<',mem_avail)

                print('MC sampling done in', time.time()-start,'s')   
            
            else:
                #print('###to implement using scipy.stats.rv_histogram -- fairly straight forward')
                zz = self.zz
                dz = zz[1]-zz[0]
                zzbin = np.linspace(zz[0]-dz/2, zz[-1]+dz/2, len(zz)+1)
                z = np.array([scipy.stats.rv_histogram((self.pdz[idx], zzbin)).rvs(size=self.Nmc) 
                              for idx in range(len(self.pdz))])
        
                MM = self.MM
                dM= MM[1]-MM[0]
                MMbin = np.linspace(MM[0]-dM/2, MM[-1]+dM/2, len(MM)+1)
                M = np.array([scipy.stats.rv_histogram((self.pdM[idx], MMbin)).rvs(size=self.Nmc) 
                              for idx in range(len(self.pdM))])
            
                Mz = np.moveaxis(np.stack([M,z]), 0, -1)
                 
                
            #Mzf = self.rootdir+'/galaxies.'+self.field+'.'+str(self.Nmc)+'MC.Mz.masked_m90.npz'   
            #np.savez(Mzf,Mz=Mz)
        
        return Mz
    
    
    def get_sig(self):
        
        avg,nprocs,fit_Olga = self.config.avg, self.config.nprocs, self.config.fit_Olga
    
        sig_Mz = np.empty(2,dtype='object')
        sig_z = np.empty(2,dtype='object')

        psig='z'
        for i,conflim in enumerate([self.config.conflim_1sig]):
            #for j,psig in enumerate(['z']): #,'Mass']):
            if fit_Olga:
                sig_Mzf = self.config.rootdir+'/sig'+psig+conflim+'.Mz.'+self.field+'.fitOlga.'+avg+'.npz'
                sig_zf = (self.config.rootdir+'/sig'+psig+conflim+'.z.'+self.field+
                      '.fitOlga.'+avg+'.npz')
            else:
                sig_Mzf = self.config.rootdir+'/sig'+psig+conflim+'.Mz.'+self.field+'.MC.'+avg+'.mag90.npz'
                sig_zf = (self.config.rootdir+'/sig'+psig+conflim+'.z.'+self.field+
                      '.MC.'+avg+'.mag90.Mlim'+str(np.round(self.config.lgmass_lim,2))+'.npz')
                
            if Path(sig_Mzf).is_file() and Path(sig_zf).is_file():
                sig_Mz[i] = np.load(sig_Mzf)['sig']
                sig_z[i] = np.load(sig_zf)['sig']
                sig_Mz[i] = np.maximum(0.01, sig_Mz[i])
                sig_z[i] = np.maximum(0.01, sig_z[i])
            else:
                ### we don't care about uncertainty on sig_z,
                ### so we can run only on 2 MC realisations of the PDFs
                if fit_Olga:
                    sig_Mz[i], sig_z[i] = self.compute_sig_fitOlga(conflim,psig,avg)
                else:
                    sig_Mz[i], sig_z[i] = self.compute_sig_MC(conflim,psig,avg,nprocs,25)
                    sig_Mz[i] = gaussian_filter1d(sig_Mz[i], 5, axis=0)
                    sig_z[i] = gaussian_filter1d(sig_z[i], 5)
                    
                sig_Mz[i] = np.maximum(0.01, sig_Mz[i])
                sig_z[i] = np.maximum(0.01, sig_z[i])
                np.savez(sig_Mzf,sig=sig_Mz[i])
                np.savez(sig_zf,sig=sig_z[i])
        

        #sigz68_Mz, sigM68_Mz, sigz95_Mz, sigM95_Mz = sig_Mz.flatten()
        #sigz68_z, sigM68_z, sigz95_z, sigM95_z = sig_z.flatten()
        sigz68_Mz, sigz95_Mz = sig_Mz#.flatten()
        sigz68_z, sigz95_z = sig_z#.flatten()   


        sigz0 = 0.01 ##backward compatibility
            
        return sigz68_Mz,sigz95_Mz,sigz68_z,sigz95_z,sigz0
    

    
    def compute_sig_fitOlga(self,conflim,psig,avg):
        
        if self.config.selection == 'maglim':
            t = Table.read(self.config.fitdir_Olga+
                           '/fit_dz_'+
                           self.config.field+'_'+
                           self.config.release+'.txt', 
                           format='ascii.commented_header')
            LIMIT=self.config.obsmag_lim
            
        elif self.config.selection == 'masslim':
            t = Table.read(self.config.fitdir_Olga+
                           '/fit_dz_'+
                           self.config.field+'_'+
                           self.config.release+'_logSM.txt', 
                           format='ascii.commented_header') 
            LIMIT=self.config.lgmass_lim

        sig_z = (t[t['LIMIT'] ==  LIMIT]['A[0]'] +
                   t[t['LIMIT'] ==  LIMIT]['A[1]'] * self.zz +
                   t[t['LIMIT'] ==  LIMIT]['A[2]'] * self.zz**2 +
                   t[t['LIMIT'] ==  LIMIT]['A[3]'] * self.zz**3 +
                   t[t['LIMIT'] ==  LIMIT]['A[4]'] * self.zz**4)
        
        sig_Mz = -99*np.ones((len(self.zz), len(self.MM)))
        
        return sig_Mz, sig_z
    
    def compute_sig_MC(self,conflim,psig,avg,nprocs,Nmc):
        sig_indiv = np.array(0.5*(self.galcat[psig+'_u'+conflim]-self.galcat[psig+'_l'+conflim]))
        quantile = int(avg[:-1])*0.01
                
        dz = 0.5*np.diff(self.zz)[0]
        Nz = len(self.zz)
        zzbin = np.linspace(self.zz[0]-dz,self.zz[-1]+dz,Nz+1) 
        binz_MC = np.digitize(self.galcat_mc[:Nmc,:,3],zzbin)-1
                
        dM = 0.5*np.diff(self.MM)[0]
        NM = len(self.MM)
        MMbin = np.linspace(self.MM[0]-dM,self.MM[-1]+dM,NM+1)        
        binM_MC = np.digitize(self.galcat_mc[:Nmc,:,4],MMbin)-1  
        
        idx_lgmass_lim = np.digitize(self.lgmass_lim,MMbin)-1
        
        Nzmax = int(min(Nz, ((self.config.zmax + 0.1*(1+self.config.zmax)) / (2*dz))))
        Nzmin = int(max(0, ((self.config.zmin - 0.1*(1+self.config.zmin)) / (2*dz))))
        
        NMmin = int(np.nanmin(binM_MC))
        NMmax = int(np.nanmax(binM_MC))
        
        #print(Nzmin, Nzmax, NMmin, NMmax)

        nb.set_num_threads(int(self.config.nprocs))
                
        _, _ = quantile_sig_mc(sig_indiv, binz_MC, Nz, 0, 1, 
                               binM_MC, NM, 0, 1, 
                               idx_lgmass_lim, quantile, 1)
        sig_Mz, sig_z = quantile_sig_mc(sig_indiv, binz_MC, Nz, Nzmin, Nzmax, 
                                        binM_MC, NM, NMmin, NMmax, 
                                        idx_lgmass_lim, quantile, Nmc)
        
        kernel1d = Gaussian1DKernel(1)
        sig_z = convolve(sig_z, kernel1d)
        kernel2d = Gaussian2DKernel(1)
        sig_Mz = convolve(sig_Mz, kernel2d)
        
        return sig_Mz, sig_z
    

    def compute_Mlim(self):
        
        logMlimz = np.zeros(len(self.zz))

        logMlim = self.galcat['Mass_median'] + 0.4 * (self.galcat['obsmag'] - self.obsmag_lim)
        
        for iz,z in enumerate(self.zz):
            zlims = ( (self.galcat['z'] > z-self.sigs.sigz68_z[iz]) & 
                         (self.galcat['z'] < z+self.sigs.sigz68_z[iz]) )
            try:
                logMlimz[iz] = np.quantile(logMlim[zlims][self.galcat['obsmag'][zlims] > 
                                        np.quantile(self.galcat['obsmag'][zlims],0.8)],self.config.lgmass_comp)
            except:
                logMlimz[iz] = -99
                
            if self.config.fit_Olga:
                logMlimz[iz] = -99
            
        self.logMlim90 = scipy.interpolate.interp1d(self.zz, logMlimz)
                
