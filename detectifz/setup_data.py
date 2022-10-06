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
from .utils import weighted_quantile, radec2detectifz, numba_loop_kde

from collections import namedtuple

import scipy.interpolate

import time

from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel

from astropy.coordinates import SkyCoord

###missing many imports

@nb.njit(parallel=True)
def quantile_sig_mc(sig_indiv, binz_MC, Nz, Nzmin, Nzmax, binM_MC, NM, NMmin, NMmax, 
                    idx_lgmass_lim, quantile, Nmc):
    mask_Mlim = (binM_MC >= idx_lgmass_lim)
    sig_Mz = np.zeros((Nz,NM))
    sig_z = np.zeros(Nz)
    for iz in nb.prange(Nzmin, Nzmax):
        for iMC in nb.prange(Nmc):
            mask_z = (binz_MC[iMC] == iz)
            m = mask_z & mask_Mlim[iMC]
            s = sig_indiv[m]
            try:
                sig_z[iz] = np.quantile(s, quantile)
            except:
                continue
            for jM in nb.prange(NMmin, NMmax):
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
        self.masksfile_radec = self.rootdir+'/masks.'+self.field+'.radec.fits'
        self.masksfile = self.rootdir+'/masks.'+self.field+'.radec.fits'   ## to modify for coord cahnge of masks

        self.pdf_Mz = config.pdf_Mz
        
        
        self.tile_id = tile_id
        Mz_MC_exists, pdz_datatype, pdM_datatype= config.datatypes
        
        self.Mz_MC_exists = Mz_MC_exists
        self.pdz_datatype = pdz_datatype
        self.pdM_datatype = pdM_datatype
        
        ## get maglim mask
        self.get_galcat()
        
        ###get 1D PDFs (z and M)
        self.zz = np.arange(config.zmin_pdf, config.zmax_pdf+config.pdz_dz, config.pdz_dz)
        self.MM = np.arange(config.Mmin_pdf, config.Mmax_pdf+config.pdM_dM, config.pdM_dM)
        
        print('get 1D PDFs')
        print('photo-z ...')
        self.pdz = self.get_pdf('z')
        print('stellar-mass ...')
        self.pdM = self.get_pdf('M')  
        
        ### get the Nmc MC samples (M, z) -- if they do not exists, get z_MC and M_MC separately.
        #### FOR QUANTILES AND TPNORM, use native ampling funtions rather than rv_histogram ?
        #### Should be the same for tpnorm. For QUANTILES it is different.
        ### I think from some tests on realistic PDF(z) that 
        #### qp.quant distribution rvs function return wrong results.
        ### --> SO I CAN JUST USE rv_histogram().rvs() for all datatypes here !
        #print('sample (M,z)')
        #self.Mz = self.sample_pdf()
        
        galcat_mc, zz, pdz, MM, pdM, xyminmax = self.get_data_detectifz()
        #self.galcat = galcat
        self.galcat_mc = galcat_mc
        self.galcat_mc_master = np.vstack(self.galcat_mc)
        self.zz = zz
        self.MM = MM
        self.pdz = pdz
        self.pdM = pdM
        self.xyminmax = xyminmax
        
        self.masks = self.get_masks()
        
        print('get sigz ...')
        start = time.time()
        sigz68_Mz,sigz95_Mz,sigz68_z,sigz95_z,sigz0 = self.get_sig()     
        self.sigs = Sigs(sigz68_Mz, sigz95_Mz, sigz68_z, sigz95_z, sigz0)
        print('done in ', time.time()-start,'s')
        
        print('get Mlim 90%')
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
                            'z', 'z_l68', 'z_u68',
                           'Mass_median','Mass_l68','Mass_u68',
                           'obsmag'])
        
        if self.config.obsmag90 is None:
            h, bin_edges = np.histogram(gal['obsmag'], bins = np.arange(10,30,0.1))
            self.obsmag90 = bin_edges[np.argmax(h)-1]
        else:
            self.obsmag90 = self.config.obsmag90
        #m90 = 22.
        self.mask_m90 = gal['obsmag'] < self.obsmag90
        self.galcat = gal[self.mask_m90]
        
        ## TO DO -- coord_change that then works with mask
        self.skycoords_center = SkyCoord(ra = np.median(self.galcat['ra_original']), 
                                                        dec = np.median(self.galcat['dec_original']), unit='deg', frame='fk5')
         
        skycoords_galaxies  = SkyCoord(ra = self.galcat['ra_original'], 
                                                        dec = self.galcat['dec_original'], unit='deg', frame='fk5')
         
        ra_detectifz, dec_detectifz = radec2detectifz(self.skycoords_center, skycoords_galaxies)
        self.galcat.add_column(Column(ra_detectifz, name='ra'))
        self.galcat.add_column(Column(dec_detectifz, name='dec'))
        #self.galcat.rename_columns(['ra_original', 'dec_original'], ['ra', 'dec'])
        
    
    def get_data_detectifz(self):
        '''
        read galaxy catalogue and HDF_PDF_Mz file and draw 100 realization from it
    
        Save an array containing the 100 MC galaxy catalogues with (id,ra,dec,z,M*) to .npz file
        '''
                
        Mzf = self.rootdir+'/galaxies.'+self.field+'.'+str(self.Nmc)+'MC.Mz.npz'
        #print(Mzf)
        if Path(Mzf).is_file():
            print('samples already saved, we read it')
            Mz = np.load(Mzf)['Mz'][self.mask_m90]
        else:    
            print('sample (M,z)...')
            Mz = self.sample_pdf()
            np.savez(Mzf,Mz=Mz)
            Mz = Mz[self.mask_m90]
        
        pdz_file_masked = self.rootdir+'/galaxies.'+self.field+'.pdz.masked_m90.npz'
        if Path(pdz_file_masked).is_file():
            pdz = np.load(pdz_file_masked)['pz']
            zz = np.load(pdz_file_masked)['z']
        else:
            pdz = np.load(self.rootdir+'/galaxies.'+self.field+'.pdz.npz')['pz'][self.mask_m90]
            zz = np.load(self.rootdir+'/galaxies.'+self.field+'.pdz.npz')['z']
        
        pdM_file_masked = self.rootdir+'/galaxies.'+self.field+'.pdM.masked_m90.npz'
        if Path(pdM_file_masked).is_file():
            pdM = np.load(pdM_file_masked)['pM']
            MM = np.load(pdM_file_masked)['M']
        else:
            pdM = np.load(self.rootdir+'/galaxies.'+self.field+'.pdM.npz')['pM'][self.mask_m90]
            MM = np.load(self.rootdir+'/galaxies.'+self.field+'.pdM.npz')['M']


        #pdM_file_masked = self.rootdir+'/galaxies.'+self.field+'.pd
        #zz = np.linspace(0.01,5,500)
        #MM = np.linspace(5.025,12.975,160)
        
        idmc = np.repeat(np.array(self.galcat['id']),self.Nmc).reshape(len(self.galcat),self.Nmc)
        ramc = np.repeat(np.array(self.galcat['ra']),self.Nmc).reshape(len(self.galcat),self.Nmc)
        decmc = np.repeat(np.array(self.galcat['dec']),self.Nmc).reshape(len(self.galcat),self.Nmc)
        zmc = Mz[:,:,1]
        Mmc = Mz[:,:,0]
    
        galmc = np.stack([idmc,ramc,decmc,zmc,Mmc,]).T
        
        xyminmax = np.array([self.galcat['ra'].min(),self.galcat['ra'].max(),
                             self.galcat['dec'].min(),self.galcat['dec'].max()])
    
        return galmc, zz, pdz, MM, pdM, xyminmax
    
    
    def get_pdf(self, param):
        
        if param == 'z':
            datatype = self.pdz_datatype
            xx = self.zz
        if param == 'M':
            datatype = self.pdM_datatype
            xx = self.MM

        pdf_fname = 'pd'+param
        pdf_name = 'p'+param
        samples_fname = 'samples_'+param
        quantiles_fname = 'quantiles_'+param
        tp_fname = 'tpnorm_'+param

        print(self.rootdir)

        if datatype == 'PDF' or Path(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.npz').is_file():
            ## we just have to read the file
            print('read the .npz file ...')
            pdf = np.load(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.npz')[pdf_name][self.mask_m90]
            x = np.load(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.npz')[param]
            ## we should add a check that pdf and xx have the same sampling, 
            ## allowing for round off differences
            if np.all( np.round(x, 3) != np.round(xx, 3) ):
                raise ValueError('sampling indicated in config.py for '+param+
                             'differs from the one from the PDF file')
                
        elif Path(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.masked_m90.npz').is_file():
            pdf = np.load(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.masked_m90.npz')[pdf_name]
            x = np.load(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.masked_m90.npz')[param]
            ## we should add a check that pdf and xx have the same sampling, 
            ## allowing for round off differences
            if np.all( np.round(x, 3) != np.round(xx, 3) ):
                raise ValueError('sampling indicated in config.py for '+param+
                             'differs from the one from the PDF file')
        
        elif ( datatype == 'samples' and 
            not( Path(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.npz').is_file() ) ):
            ### do KDE estimate
            print('KDE estimate...')
            if self.Mz_MC_exists:
                if param == 'z':
                    dataset = np.load(self.rootdir+'/galaxies.'+self.field+'.'+
                                      str(self.Nmc)+'MC.Mz.npz')['Mz'][self.mask_m90, :, 1]
                if param == 'M':
                    dataset = np.load(self.rootdir+'/galaxies.'+self.field+'.'+
                                      str(self.Nmc)+'MC.Mz.npz')['Mz'][self.mask_m90, :, 0]
            else:
                dataset = np.load(self.rootdir+'/galaxies.'+self.field+'.'+samples_fname+'.npz')['samples'][self.mask_m90]
            #pdf = np.array([scipy.stats.gaussian_kde(dataset[i])(xx) for i in range(len(dataset))])
            pdf = numba_loop_kde(xx, dataset)
            if param == 'z':
                np.savez(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.masked_m90.npz', 
                         pz=pdf, z=xx)
            if param == 'M':
                np.savez(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.masked_m90.npz', 
                         pM=pdf, M=xx)
        
        elif ( datatype == 'quantiles' and 
            not( Path(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.npz').is_file() ) ):
            ## use qp to approxiamte PDF
            print('run qp estimate...')
            quantiles_def = np.load(self.rootdir+'/galaxies.'+self.field+'.'+quantiles_fname+'.npz')['quantiles_def']
            quantiles = np.load(self.rootdir+'/galaxies.'+self.field+'.'+quantiles_fname+'.npz')['quantiles'][self.mask_m90]
            dist = qp.stats.quant(quants=quantiles_def, locs=quantiles)
            pdf = dist.pdf(zz)        
        
        elif ( datatype == 'tpnorm' and 
            not( Path(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.npz').is_file() ) ):
            ### two piece normal PDF for l68, u68
            print('compute two piece normal...')
            #med = np.load(self.rootdir+'/galaxies.'+self.field+'.'+tp_fname+'.npz')['med']
            #l68 = np.load(self.rootdir+'/galaxies.'+self.field+'.'+tp_fname+'.npz')['l68']
            #u68 = np.load(self.rootdir+'/galaxies.'+self.field+'.'+tp_fname+'.npz')['u68']
            tab = Table.read(self.rootdir+'/galaxies.'+self.field+'.galcat.fits')[self.mask_m90]
            
            if param == 'z':
                med = np.array(tab[self.config.z_med_colname])
                l68 = np.array(tab[self.config.z_l68_colname])
                u68 = np.array(tab[self.config.z_u68_colname])
                
            if param == 'M':
                med = np.array(tab[self.config.M_med_colname])
                l68 = np.array(tab[self.config.M_l68_colname])
                u68 = np.array(tab[self.config.M_u68_colname])
                
            #sig_l68 = np.maximum(0.01, sig_l68)
            #sig_u68 = np.maximum(0.01, sig_u68)
            sig_l68 = np.maximum(0.01, med - l68)
            sig_u68 = np.maximum(0.01, u68 - med)
            
            pdf = np.array([tpnorm(loc=med[i], 
                                 sigma1=sig_l68[i], 
                                 sigma2=sig_u68[i]).pdf(xx) 
                          for i in range(len(med))])
            if param == 'z':
                np.savez(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.masked_m90.npz', 
                         pz=pdf, z=xx)
            if param == 'M':
                np.savez(self.rootdir+'/galaxies.'+self.field+'.'+pdf_fname+'.masked_m90.npz', 
                         pM=pdf, M=xx)
                           
        else:
            raise ValueError('wrong value for PDF_z_datatype keyword in config.py.'+
                             ' Should be "PDF", "samples", "quantiles" or "tpnorm"')

        return pdf



    def get_masks(self):
        ### need to load intermediate files 
        ###- sigz68.mz., masks, etc - here and pass them to the functions that need them    
        #masks_im = fits.getdata(self.masksfile)
        #headmasks = fits.getheader(self.masksfile)
        #return masks_im,headmasks
        masks = fits.open(self.masksfile)[0]
        return masks
    
    
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
            
                Mz = np.moveaxis(np.stack([z,M]), 0, -1)
                 
                
            Mzf = self.rootdir+'/galaxies.'+self.field+'.'+str(self.Nmc)+'MC.Mz.npz'   
            np.savez(Mzf,Mz=Mz)
        
        return Mz
    
    
    def get_sig(self):
        
        avg,nprocs = self.config.avg,self.config.nprocs
    
        sig_Mz = np.empty(2,dtype='object')
        sig_z = np.empty(2,dtype='object')

        psig='z'
        for i,conflim in enumerate(['68']):
            #for j,psig in enumerate(['z']): #,'Mass']):
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
                sig_Mz[i], sig_z[i] = self.compute_sig_MC(conflim,psig,avg,nprocs,2)
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

        logMlim = self.galcat['Mass_median'] + 0.4 * (self.galcat['obsmag'] - self.obsmag90)
        
        for iz,z in enumerate(self.zz):
            zlims = ( (self.galcat['z'] > z-self.sigs.sigz68_z[iz]) & 
                         (self.galcat['z'] < z+self.sigs.sigz68_z[iz]) )
            try:
                logMlimz[iz] = np.quantile(logMlim[zlims][self.galcat['obsmag'][zlims] > 
                                        np.quantile(self.galcat['obsmag'][zlims],0.8)],self.config.lgmass_comp)
            except:
                logMlimz[iz] = -99
            
        self.logMlim90 = scipy.interpolate.interp1d(self.zz, logMlimz)
                
