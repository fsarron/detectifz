import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import pickle
import numpy as np
#import numba as nb
from astropy.table import Table
import h5py
from pathlib import Path
from astropy.io import fits
import ray
from . import density, detection, cleaning, r200, members, membership_jwst
from .tiling import Tile
from .utils import Mlim_DETECTIFz, detectifz2radec
from astropy import units
from collections import namedtuple

import time

import warnings
from astropy.utils.exceptions import AstropyWarning,AstropyUserWarning

warnings.filterwarnings("ignore")

class DETECTIFz(object):
    
    def __init__(self, field='UDS', data=None, config=None):
        self.field = field
        self.data = data
        self.config = config
        #self.config.Nmc = self.config.Nmc
        
        print('run gal_Mlim')
        self.get_galMlim()
       
        print('run zslice')
        self.create_slices_sigz()
        np.savez(self.config.rootdir+
                 '/zslices.'+self.field+
                 '_Mlim'+str(np.round(self.config.lgmass_lim,2))+
                 '.sigz68_z_'+self.config.avg+'.npz', 
                 zslices = self.zslices)
        
        
    def get_galMlim(self):
 
        z_mc, z_mc_master, z_galcat = (self.data.galcat_mc[:,:,3],
                                      self.data.galcat_mc_master[:,3],
                                      self.data.galcat['z'])
        ##force z in the interpolation range
        z_mc, z_mc_master, z_galcat = (
            np.minimum(np.maximum(self.data.zz[0], z_mc), self.data.zz[-1]),
            np.minimum(np.maximum(self.data.zz[0], z_mc_master), self.data.zz[-1]),
            np.minimum(np.maximum(self.data.zz[0], z_galcat), self.data.zz[-1]))
     
            
        if self.config.selection == 'legacy' or self.config.selection == 'maglim':

            lgM_lim_mc = Mlim_DETECTIFz(self.data.logMlim90,
                                        self.data.lgmass_lim,
                                        z_mc)
            lgM_lim_mc_master = Mlim_DETECTIFz(self.data.logMlim90,
                                        self.data.lgmass_lim,
                                        z_mc_master)
            lgM_lim_galcat = Mlim_DETECTIFz(self.data.logMlim90,
                                        self.data.lgmass_lim,
                                        z_galcat)
        
        elif self.config.selection == 'masslim':
            lgM_lim_mc = self.data.lgmass_lim
            lgM_lim_mc_master = self.data.lgmass_lim
            lgM_lim_galcat = self.data.lgmass_lim
            
        
        ### galmc    
        self.maskMlim_mc = self.data.galcat_mc[:,:,4] >= lgM_lim_mc
        self.galcatmc_Mlim = np.empty(self.data.Nmc,dtype='object')
        for i in range(self.data.Nmc):
            self.galcatmc_Mlim[i] = self.data.galcat_mc[i][self.maskMlim_mc[i]]

        ### galmc_master
        maskMlim_master = self.data.galcat_mc_master[:,4] >= lgM_lim_mc_master
        self.galcatmc_master_Mlim = self.data.galcat_mc_master[maskMlim_master]
        
        ### galcat
        maskMlim = self.data.galcat['Mass_median'] >= lgM_lim_galcat
        self.galcat_Mlim = self.data.galcat[maskMlim]
        #self.pdz_Mlim = self.data.pdz[maskMlim]
            
        
    def create_slices_sigz(self):
        zcentre0=self.config.zmin 
        zcentre=[self.config.zmin] 
        zhigh=[self.config.zmin]
        zlow = []

        dz=np.diff(self.data.zz)[0]
    
        i=0

        while (zhigh[i] < self.config.zmax):
            jj = np.intc((zcentre[i]-self.data.zz[0])/dz)
            if self.data.sigs.sigz68_z[jj] > 0:
                zlow.append(zcentre[i] - max(0.01,self.data.sigs.sigz68_z[jj]))
                zhigh.append(zcentre[i] + max(0.01,self.data.sigs.sigz68_z[jj]))
            else:
                raise ValueError("There is a z>=zmin at which sigz68 <= 0. Check sigz file")
    
            zcentre.append(zcentre[i]+self.config.dzmap)
            i = i+1
        

        centre = np.array(zcentre[:-1])
        zinf = np.array(zlow)
        zsup = np.array(zhigh[1:])
        
        self.zslices = np.c_[centre,zinf,zsup]
        self.zslices = self.zslices[zinf > self.config.zmin]  #added compared to Legacy version (Sarron+21)
        

    def run_main(self):
        
        if not sys.warnoptions:
            warnings.simplefilter('ignore', category=AstropyWarning)
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            warnings.simplefilter("ignore")
        
        print('run get_dmap')
        self.im3d,self.weights2d,self.head2d = density.get_dmap(self)
        
        clusdetf = (self.config.rootdir+'/candidats_'+self.field+'_SN'+str(self.config.SNmin)+
                    '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.fits')
        pdzclusdetf = (self.config.rootdir+'/pdz_im3d.candidats_'+self.field+'_SN'+str(self.config.SNmin)+
                       '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.npz')
        
        subdetsf = (self.config.rootdir+'/subdets_'+self.field+'_SN'+str(self.config.SNmin)+
                    '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.npz')
        
        det_slicesf = (self.config.rootdir+'/detslices_'+self.field+'_SN'+str(self.config.SNmin)+
                    '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.npz')

        alldet_f = (self.config.rootdir+'/alldet_'+self.field+'_SN'+str(self.config.SNmin)+
                    '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.npz')
        
        segm_f = (self.config.rootdir+'/segm_'+self.field+'_SN'+str(self.config.SNmin)+
                    '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.npz')
        print('get catalogue')
        if Path(clusdetf).is_file() and Path(pdzclusdetf).is_file():
            print('clus and pdzclus file exists, we just read it')
            self.clus = Table.read(clusdetf)
            self.pdzclus = np.load(pdzclusdetf)['pz']
        else:    
            print('run detection')
            self.det_slices, alldet, pos_slices, segm = detection.detection(self)
            np.savez(det_slicesf, *self.det_slices)
            np.savez(alldet_f, *alldet)
            np.savez(segm_f, *segm)

            print('run cleaning')
            self.clus, self.subdets = cleaning.cleaning(self,alldet)
            self.clus.write(clusdetf,overwrite=True)
            np.savez(subdetsf, *self.subdets)
            print('run clus_pdz')
            self.pdzclus = cleaning.clus_pdz_im3d(self,1)
            #self.clus.write(clusdetf,overwrite=True)
            np.savez(pdzclusdetf,pz=self.pdzclus,z=self.data.zz)

        # Obtaining ICRS coordinates on the final detection catalog

        true_ra, true_dec = detectifz2radec(self.data.skycoords_center,[self.clus['ra'],self.clus['dec']])
        self.tmaster = self.clus
        self.tmaster['ra_detectifz'] = self.tmaster['ra']
        self.tmaster['dec_detectifz'] = self.tmaster['dec']
        self.tmaster['ra'] = true_ra
        self.tmaster['dec'] = true_dec
        self.tmaster.write(self.config.rootdir+'/detections_'+self.field+'.fits',overwrite=True)
        

        
    def run_R200(self):
        print('run R200')
        clus_r200 = r200.get_R200(self)
    
        ##clean
        self.clus_r200_clean = clus_r200[clus_r200['R200c_Mass_median'] > 0.0]
        self.pdzclus_r200_clean = self.pdzclus[clus_r200['R200c_Mass_median'] > 0.0]
        
        ##save
        self.clus_r200_clean.write(self.config.rootdir+'/candidats_'+self.field+'_SN'+str(self.config.SNmin)+
                              '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.r200.clean.fits',
                              overwrite=True)
        np.savez(self.config.rootdir+'/pdz_im3d.candidats_'+self.field+'_SN'+str(self.config.SNmin)+
                 '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.r200.clean.npz',
                 pz=self.pdzclus_r200_clean,z=self.data.zz)
        

    def run_Pmem_jwst(self):
        print('run Pmem JWST')
        pdzgal_arx = np.load(self.config.rootdir+'galaxies.'+self.field+'.pdz.npz')
        pdzgal = pdzgal_arx['pz']
        zzgal = pdzgal_arx['z']
        
        pmem24, pmem21_z, pmem21_Mz, pconv_z, pconv_Mz, prior_clus, mask_inclus, prior_z, Npos, NtotR, Nbkg, wnoclus= membership_jwst.get_pmem(self, zzgal, pdzgal)
        
        pmemf = (self.config.rootdir+'/p_mem.'+self.field+'_SN'+str(self.config.SNmin)+
                    '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.npz')

        
        np.savez(pmemf, 
                 pmem24=pmem24,
                 pmem21_z=pmem21_z,
                 pmem21_Mz=pmem21_Mz,
                 pconv_z=pconv_z,
                 pconv_Mz=pconv_Mz,
                 prior_clus = prior_clus,
                 mask_inclus=mask_inclus,
                 prior_z = prior_z, 
                 Npos = Npos, 
                 NtotR = NtotR, 
                 Nbkg = Nbkg,
                 wnoclus = wnoclus)
    
        
    
    
    def run_Pmem(self):
    
        print('run im3d_info')
        #start_im3d_info = time.time()
        im3d_info_f = ( 'im3d_info.'+self.field+'_Mlim'+str(np.round(self.config.lgmass_lim,2))+'_SN'+str(self.config.SNmin)+
                       '.sigz68_z_'+self.config.avg+'.r200.clean.im3d.npz' )
        if Path(im3d_info_f).is_file():
            im3d_info = np.load(im3d_info_f,allow_pickle=True)['im3d_info']
        else:    
            im3d_info = members.get_im3d_info(self.im3d,self.weights2d,self.head2d,
                                              self.clus_r200_clean,self.data.galcat,
                                             self.config.nprocs)
            np.savez(im3d_info_f,im3d_info=im3d_info)
        #print('im3d_info done in ',time.time()-start_im3d_info,'s') 
        
        print('run get_nfield_noclus')
        NFf = 'NF.Mz.'+field+'.PDF_Mz.noclus.SN'+str(SNmin)+'.'+str(radnoclus)+'r200.sSFR_10.7.npz'
        w3df = 'weights3d.'+field+'.noclus.SN'+str(SNmin)+'.'+str(radnoclus)+'r200.npz'    
        if Path(NFf).is_file() and Path(w3df).is_file():
            print('already saved, we read it')
            NF_Mz_noclus= np.load(NFf,allow_pickle=True)['NF_Mz']
            NQF_Mz_noclus= np.load(NFf,allow_pickle=True)['NQF_Mz']
            Omega_F_z = np.load(NFf,allow_pickle=True)['Omega_F_z']
            nF_Mz_noclus = NF_Mz_noclus / Omega_F_z[:,None]
        else:
            NF_Mz_noclus,NQF_Mz_noclus,Omega_F_z,weights3d_noclus, mask_noclus = get_nfield(
                field,clus_r200_clean,pdzclus_r200_clean,zz,gal,sigz68_z,weights2d,
                head2d,radnoclus,SNmin,nprocs,getQ=True)
            np.savez(NFf,NF_Mz=NF_Mz_noclus,
                     NQF_Mz=NQF_Mz_noclus,
                     Omega_F_z=Omega_F_z, 
                     weights3d_noclus=weights3d_noclus, 
                     mask_noclus=mask_noclus)
            nF_Mz_noclus = NF_Mz_noclus / Omega_F_z[:,None]

    
        print('run get_ntot_R200')
        Ntot_r200f = 'NQtotR200.Mz.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+'.sSFR_10.7.npz'
        if Path(Ntot_r200f).is_file():
            print('already saved, we read it')
            Ntot_Mz_r200= np.load(Ntot_r200f,allow_pickle=True)['Ntot_Mz_r200']
            NQ_Mz_r200 = np.load(Ntot_r200f,allow_pickle=True)['NQ_Mz_r200']
            Omega_C_r200= np.load(Ntot_r200f,allow_pickle=True)['Omega_C_r200']
            mask_clus_r200 = np.load(Ntot_r200f,allow_pickle=True)['mask_clus_r200']
            mask_clus_2Mpc = np.load(Ntot_r200f,allow_pickle=True)['mask_clus_2Mpc']
        else:
            (Ntot_Mz_r200,NQ_Mz_r200, Omega_C_r200, 
             mask_clus_r200, mask_clus_2Mpc)  = get_ntot_r200(
                field,clus_r200_clean,gal,weights2d,
                head2d,zz,MM,SNmin,nprocs,getQ=True)
            
            np.savez(Ntot_r200f,Ntot_Mz_r200=Ntot_Mz_r200,
                     NQ_Mz_r200=NQ_Mz_r200,
                     Omega_C_r200=Omega_C_r200,
                     mask_clus_r200 = mask_clus_r200,
                     mask_clus_2Mpc=mask_clus_2Mpc)    
     
        SN_f = 'SNQtot.Mz.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+'.npz'
        if Path(SN_f).is_file():
            print('already saved, we read it')
            SNtot_Mz_r200 = np.load(SN_f,allow_pickle=True)['SNtot_Mz_r200']
            SNQtot_Mz_r200 = np.load(SN_f,allow_pickle=True)['SNQtot_Mz_r200']
            SNF_Mz_noclus = np.load(SN_f,allow_pickle=True)['SNF_Mz_noclus']
            SNQF_Mz_noclus = np.load(SN_f,allow_pickle=True)['SNQF_Mz_noclus']
        else:    
            (SNtot_Mz_r200, SNQtot_Mz_r200, 
             SNF_Mz_noclus, SNQF_Mz_noclus) = smooth_Nz(
                survey,Ntot_Mz_r200,NQ_Mz_r200,NF_Mz_noclus,
                NQF_Mz_noclus,Omega_F_z,Omega_C_r200,zz,MM,sigz95_Mz,getQ=True)     
            
            np.savez(SN_f,SNtot_Mz_r200=SNtot_Mz_r200,
                     SNQtot_Mz_r200=SNQtot_Mz_r200,
                     SNF_Mz_noclus=SNF_Mz_noclus,
                     SNQF_Mz_noclus=SNQF_Mz_noclus)   
        
        
        print('run get_Pmem')
        Pmemf = ('Pmem.'+field+'.PDF_Mz.clus.SN'+str(SNmin)+
                 '.noclus.'+str(radnoclus)+'r200.sigM95.M90.smooth.Pcclus.npz')
        if Path(Pmemf).is_file():
            print('already saved, we read it')
            Pmem = np.load(Pmemf,allow_pickle=True)['Pmem']
            Mass_clus = np.load(Pmemf,allow_pickle=True)['Mass_clus']
            Pconv = np.load(Pmemf,allow_pickle=True)['Pconv']
        else:    
            Pmem, Mass_clus, Pconv = get_Pmem_smooth(field,mocktype,survey,
                                                     im3d_info,mask_clus_2Mpc,SNF_Mz_noclus_R200clus,
                                                     SNtot_Mz_r200,pdz,pdzclus_r200_clean,clus_r200_clean,
                                                     zz,sigz68_Mz,MM,nprocs)
            np.savez(Pmemf,Pmem=Pmem,Mass_clus=Mass_clus,Pconv=Pconv)
    
    
        print('Add Ngal, SMtot to clus catalogue + make cluster member catalogues')
        print('mak members')
        memdet_2Mpc = make_members(clus_r200_clean,gal,mask_clus_2Mpc,Pmem,Mass_clus,Pconv,field)
        print('add Ngal/SMtot')
        clus = add_Ngal_SMtot(field,clus_r200_clean,memdet_2Mpc,nprocs)
        clusfinalf = ('candidats_'+field+'_SN'+str(SNmin)+
                      '_Mlim10.PDF_Mz.irac.sigz68_z_'+avg+
                      '.r200.clean.Ngal_SMtot.sigM95.M90.smooth.Pcclus.fits')
        clus.write(clusfinalf,overwrite=True)
        
    

    
