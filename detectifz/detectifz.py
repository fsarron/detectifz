import sys
import numpy as np
#import numba as nb
from astropy.table import Table
import h5py
from pathlib import Path
from astropy.io import fits
import ray
from . import density, detection, cleaning, r200, members
from .tiling import Tile
from .utils import Mlim_DETECTIFz
from astropy import units
from collections import namedtuple

import time

import warnings
from astropy.utils.exceptions import AstropyWarning,AstropyUserWarning
###missing many imports

global nprocs
nprocs=8
avg='68p'  

#def Mlim(field,masslim,z):
#    if field == 'UDS' or field == 'H15_UDS':
#        Mlim = 7.847 + 1.257*z - 0.150*z**2
#    if field == 'UltraVISTA' or field == 'H15_UltraVISTA':
#        Mlim = 8.378 + 1.262*z - 0.153*z**2
#    if field == 'VIDEO' or field == 'H15_VIDEO':
#        Mlim = 8.455 + 1.697*z - 0.278*z**2
#    else:
#        raise ValueError('field'+field+' is not implemented !')
#    return np.maximum(Mlim,masslim)


class DETECTIFz(object):
    
    def __init__(self, field='UDS', data=None, config=None):
        self.field = field
        self.data = data
        self.config = config
        self.config.Nmc = self.config.Nmc_fast
        
        print('run gal_Mlim')
        self.get_galMlim()
       
        print('run zslice')
        self.create_slices_sigz()
        np.savez(self.config.rootdir+'/zslices.'+self.field+
                       '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.npz', zslices = self.zslices)
        
        
    def get_galMlim(self):
        field,masslim,z = self.data.field,self.config.lgmass_lim,self.data.galcat_mc[:,:,3]
        ##force z in the interpolation range
        z = np.minimum(np.maximum(self.data.zz[0], z), self.data.zz[-1])
        self.maskMlim_mc = self.data.galcat_mc[:,:,4] > Mlim_DETECTIFz(self.data.logMlim90,masslim,z)
    
        self.galcatmc_Mlim = np.empty(self.data.Nmc,dtype='object')
        for i in range(self.data.Nmc):
            self.galcatmc_Mlim[i] = self.data.galcat_mc[i][self.maskMlim_mc[i]]
    
        field,masslim,z = self.data.field,self.data.lgmass_lim,self.data.galcat_mc_master[:,3]
        ##force z in the interpolation range
        z = np.minimum(np.maximum(self.data.zz[0], z), self.data.zz[-1])
        maskMlim_master = self.data.galcat_mc_master[:,4] > Mlim_DETECTIFz(self.data.logMlim90,masslim,z)
        self.galcatmc_master_Mlim = self.data.galcat_mc_master[maskMlim_master]
        
        field,masslim,z = self.data.field,self.data.lgmass_lim,self.data.galcat['z']
        ##force z in the interpolation range
        z = np.minimum(np.maximum(self.data.zz[0], z), self.data.zz[-1])
        maskMlim = self.data.galcat['Mass_median'] > Mlim_DETECTIFz(self.data.logMlim90,masslim,z)
        self.galcat_Mlim = self.data.galcat[maskMlim]
        self.pdz_Mlim = self.data.pdz[maskMlim]
        
    def create_slices_sigz(self):
        zcentre0=self.config.zmin 
        zcentre=[self.config.zmin] 
        zhigh=[self.config.zmin]
        zlow = []

        dz=np.diff(self.data.zz)[0]
    
        i=0

        while (zhigh[i] < self.config.zmax):
            jj = np.int((zcentre[i]-self.data.zz[0])/dz)
            if self.data.sigs.sigz68_z[jj] > 0:
                zlow.append(zcentre[i] - max(0.01,self.data.sigs.sigz68_z[jj]))
                zhigh.append(zcentre[i] + max(0.01,self.data.sigs.sigz68_z[jj]))
            else:
                raise ValueError("There is a z>=0.1 at which sigz68 == 0. Check sigz file")
    
            zcentre.append(zcentre[i]+self.config.dzslice)
            i = i+1
        

        centre = np.array(zcentre[:-1])
        zinf = np.array(zlow)
        zsup = np.array(zhigh[1:])
        
        self.zslices = np.c_[centre,zinf,zsup]
        
        

    def run(self):
        
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
    
        print('get catalogue')
        if Path(clusdetf).is_file() and Path(pdzclusdetf).is_file():
            print('clus and pdzclus file exists, we just read it')
            self.clus = Table.read(clusdetf)
            self.pdzclus = np.load(pdzclusdetf)['pz']
        else:    
            print('run detection')
            self.det_slices, alldet, pos_slices = detection.detection(self)
       
            print('run cleaning')
            self.clus, self.selfsubdets = cleaning.cleaning(self,alldet)
    
            print('run clus_pdz')
        #clus,pdzclus = clus_pdz(survey,gal_Mlim,pdz_Mlim,zz,masks_im,headmasks,clus0,2)
            self.pdzclus = cleaning.clus_pdz_im3d(self,1)
            self.clus.write(clusdetf,overwrite=True)
            np.savez(pdzclusdetf,pz=self.pdzclus,z=self.data.zz)
        
        '''
        print('run R200')
        clus_r200 = r200.get_R200(self)
    
        ##clean
        self.clus_r200_clean = clus_r200[clus_r200['R200c_Mass_median'] > 0.1]
        self.pdzclus_r200_clean = self.pdzclus[clus_r200['R200c_Mass_median'] > 0.1]
        
        ##save
        self.clus_r200_clean.write(self.config.rootdir+'/candidats_'+self.field+'_SN'+str(self.config.SNmin)+
                              '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.r200.clean.fits',
                              overwrite=True)
        np.savez(self.config.rootdir+'/pdz_im3d.candidats_'+self.field+'_SN'+str(self.config.SNmin)+
                 '_Mlim'+str(np.round(self.config.lgmass_lim,2))+'.sigz68_z_'+self.config.avg+'.r200.clean.npz',
                 pz=self.pdzclus_r200_clean,z=self.data.zz)
    
        '''
    
    
    
    
    
    
    
    
    
    
    def run_Pmem(self):
    
        print('run im3d_info')
        #start_im3d_info = time.time()
        im3d_info_f = ( 'im3d_info.'+self.field+'_Mlim'+str(np.round(self.param.lgmass_lim,2))+'_SN'+str(self.config.SNmin)+
                       '.sigz68_z_'+self.config.avg+'.r200.clean.im3d.npz' )
        if Path(im3d_info_f).is_file():
            im3d_info = np.load(im3d_info_f,allow_pickle=True)['im3d_info']
        else:    
            im3d_info = members.get_im3d_info(self.im3d,self.weights2d,self.head2d,
                                              self.clus_r200_clean,self.data.galcat,
                                             self.config.nprocs)
            np.savez(im3d_info_f,im3d_info=im3d_info)
        #print('im3d_info done in ',time.time()-start_im3d_info,'s') 
        
        
        '''
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
        '''
    

    
    
