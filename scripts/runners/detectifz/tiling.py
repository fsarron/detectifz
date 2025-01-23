import sys
import numpy as np
from astropy.table import Table

from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units

from astropy.io import fits
from astropy import wcs

import subprocess
from pathlib import Path

import numba as nb

from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel

from detectifz.utils import celestial_rectangle_area

@nb.njit(parallel=True)
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



class Tile(object):
    
    def __init__(self, tileid=None, corners=None, border_width=None, field=None, release=None, tilesdir=None):
        if not (corners is None): 
            self.id = tileid
            self.corners = corners
            self.border_width = border_width
            self.bottom_left = corners[0][0]
            self.top_left = corners[0][1]
            self.bottom_right = corners[1][0]
            self.top_right = corners[1][1]
            self.field = field
            self.release = release
            self.tilesdir = tilesdir
            
            self._core_bottom_left = tuple(np.add(corners[0][0] , (border_width, border_width)))
            self._core_top_left = tuple(np.add(corners[0][1] , (border_width, -border_width)))
            self._core_bottom_right =  tuple(np.add(corners[1][0], (-border_width, border_width)))
            self._core_top_right = tuple(np.add(corners[1][1], (-border_width, -border_width)))
            
        else:
            raise TypeError('Could not init Tile object without corners')
            
            
    def run_venice_inout(self):
 
        self.FITSmasks.writeto(self.FITSmasks_filename, overwrite=True)

        coords = SkyCoord(ra = self.galcat_raw['ra'],
                 dec = self.galcat_raw['dec'], unit = 'deg', frame='fk5')
        pix = wcs.utils.skycoord_to_pixel(coords, wcs = wcs.WCS(self.FITSmasks.header))
        self.galcat_raw['xpix_mask'] = pix[0]
        self.galcat_raw['ypix_mask'] = pix[1]

        self.galcat_raw.write(self.galcat_raw_filename, overwrite=True)
        
                
        process = subprocess.Popen(["venice", "-cat", 
                                    self.galcat_raw_filename, 
                                    "-catfmt", 
                                    "fits", 
                                    "-xcol", 
                                    "xpix_mask", 
                                    "-ycol", 
                                    "ypix_mask", 
                                    "-m", 
                                    self.FITSmasks_filename, 
                                    "-o", 
                                    self.thistile_dir+"/galcat_mask."+self.field+".tmp.fits"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
        process.wait()
        result = process.communicate()
        
        
        galcat = Table.read(self.thistile_dir+"/galcat_mask."+self.field+".tmp.fits")
        self.galcat = galcat[galcat['flag'] == 1]
        self.galcat_filename = self.thistile_dir+"/galaxies."+self.field+".galcat.fits"
        self.galcat.write(self.galcat_filename, overwrite=True) 
        
        process = subprocess.Popen(["rm", self.thistile_dir+"/galcat_mask."+self.field+".tmp.fits"])
        process.wait()
        result = process.communicate()
        
        
        
    def run_venice_pixelize(self):
        
        (rainf, rasup, 
             decinf, decsup) = (
                self.bottom_left[0], self.top_right[0], 
                self.bottom_left[1], self.top_right[1] )
            
        ##lance venice to get pixelized mask at giuven resolution with given (ra,dec) limits
        process = subprocess.Popen(["venice", 
                                    "-m", 
                                    self.master_masksfile, 
                                    "-nx",
                                    str(int((rasup-rainf)/self.pixdeg)),
                                    "-ny",
                                    str(int((decsup-decinf)/self.pixdeg)),
                                    "-xmin", 
                                    str(rainf),
                                    "-xmax",
                                    str(rasup),
                                    "-ymin",
                                    str(decinf),
                                    "-ymax",
                                    str(decsup),
                                    "-o", 
                                    self.thistile_dir+"/masks."+self.field+".tmp.mat"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
        #print("the commandline is {}".format(process.args))
        process.wait()
        result = process.communicate()
        #print(result)
        flag = np.loadtxt(self.thistile_dir+'/masks.'+self.field+'.tmp.mat')
        
        ### from the pixelized matrix, make a fits hdu with WCS information
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [1.0, 1.0]
        w.wcs.cdelt = np.array([self.pixdeg, self.pixdeg])
        w.wcs.crval = [rainf, decinf]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        self.FITSmasks = fits.PrimaryHDU(data=flag.astype(np.int8),header=w.to_header())
        self.FITSmasks_filename = self.thistile_dir+'/masks.'+self.field+'.radec.fits'
        self.FITSmasks.writeto(self.FITSmasks_filename, overwrite=True)

        
    def run_cutout(self):

        (rainf, rasup, 
             decinf, decsup) = (
                self.bottom_left[0], self.top_right[0], 
                self.bottom_left[1], self.top_right[1] )
        
        FITSmasks = fits.open(self.config_tile.masksfile)[0]
        
        ##SKY MASK (BAD REGIONS -- FITS IMAGE)
        racentre = (rainf + rasup)/2.
        deccentre = (decinf + decsup)/2.

        size_sky_ra = (rasup - rainf) * units.deg
        size_sky_dec = (decsup - decinf) * units.deg

        centre_sky = SkyCoord(ra=racentre, 
                                dec=deccentre, unit='deg', frame='icrs')
            
        FITScutout = Cutout2D(data = FITSmasks.data,
                              position = centre_sky,
                              size = (size_sky_dec, size_sky_ra),
                              wcs = wcs.WCS(FITSmasks.header))
            
        self.FITSmasks = FITSmasks
        self.FITSmasks.data = FITScutout.data
        self.FITSmasks.header.update(FITScutout.wcs.to_header())
                    
            
        self.galcat_raw_filename = self.thistile_dir+'/galaxies.'+self.field+'.galcat_raw.fits'
        self.FITSmasks_filename = self.thistile_dir+'/masks.'+self.field+'.radec.fits'
        
    def write_config_detectifz(self):
        
        f=open(self.tilesdir+'/config_master_detectifz.ini','r')
        self.config = f.readlines()
        f.close()
        self.config.append("\n")
        self.config.append("[GENERAL]")
        self.config.append("\n")
        self.config.append("field="+self.field+"\n")
        self.config.append("release="+self.release+"\n")
        self.config.append("rootdir="+self.thistile_dir+"/ \n")
        
        with open(self.thistile_dir+'/config_detectifz.ini', 'w') as f:
            for line in self.config:
                f.write(line)
                
 
            
class Tiles(object):

    def __init__(self, config_tile=None):
        self.config_tile = config_tile
        if self.config_tile is None:
            raise TypeError('Could not init Tiles object without config file')   
            
    def get_tiles(self):
                    
        self.config_tile.tilesdir = (self.config_tile.tiles_rootdir+
                                     '/'+self.config_tile.field+
                                     '/'+self.config_tile.release)
        Path(self.config_tile.tilesdir).mkdir(parents=True, exist_ok=True)   
        
        
        ramin, ramax, decmin, decmax = (self.config_tile.ramin, 
                                        self.config_tile.ramax, 
                                        self.config_tile.decmin, 
                                        self.config_tile.decmax )
        max_area = self.config_tile.max_area
        border_width = self.config_tile.border_width
        
        #total_area = (ramax-ramin)*(decmax-decmin)
        total_area = celestial_rectangle_area(ramin, decmin, ramax, decmax)
        
        Nmin_tiles =  total_area/max_area
        Nsplit = np.ceil(np.sqrt(Nmin_tiles)).astype(int) + 1 
        
        ras = np.linspace(ramin, ramax, Nsplit)
        rainfs, rasups = ras[:-1]-border_width, ras[1:]+border_width

        decs = np.linspace(decmin, decmax, Nsplit)
        decinfs, decsups = decs[:-1]-border_width, decs[1:]+border_width
        
        c00 = np.meshgrid(rainfs, decinfs)
        c01 = np.meshgrid(rainfs, decsups)
        c10 = np.meshgrid(rasups, decinfs)
        c11 = np.meshgrid(rasups, decsups)
        
        self.tiles = [] #np.empty((Nsplit-1,Nsplit-1), dtype='object')
        l = 0

        for i in range(Nsplit-1):
            for j in range(Nsplit-1):
                corners = [[(c00[0][i,j], c00[1][i,j]) , (c01[0][i,j], c01[1][i,j])],
                           [(c10[0][i,j], c10[1][i,j]), (c11[0][i,j], c11[1][i,j])]]
                
                tile = Tile(tileid=l, corners=corners, 
                                       border_width = border_width,
                                       field=self.config_tile.field,
                                      release=self.config_tile.release,
                                      tilesdir = self.config_tile.tilesdir)
                self.tiles.append(tile)

                l += 1
                
                
    def compute_sig_MC(self,conflim,psig,avg,nprocs,Nmc):
        sig_indiv = np.array(0.5*(self.galcat[psig+'_u'+conflim]-self.galcat[psig+'_l'+conflim]))
        quantile = int(avg[:-1])*0.01

        dz = 0.5*np.diff(self.zz)[0]
        Nz = len(self.zz)
        zzbin = np.linspace(self.zz[0]-dz,self.zz[-1]+dz,Nz+1) 
        binz_MC = np.digitize(self.galcat_mc[:Nmc,:,0],zzbin)-1

        dM = 0.5*np.diff(self.MM)[0]
        NM = len(self.MM)
        MMbin = np.linspace(self.MM[0]-dM,self.MM[-1]+dM,NM+1)        
        binM_MC = np.digitize(self.galcat_mc[:Nmc,:,1],MMbin)-1  

        idx_lgmass_lim = np.digitize(self.config_tile.lgmass_lim,MMbin)-1

        Nzmax = int(min(Nz, ((self.config_tile.zmax + 0.1*(1+self.config_tile.zmax)) / (2*dz))))
        Nzmin = int(max(0, ((self.config_tile.zmin - 0.1*(1+self.config_tile.zmin)) / (2*dz))))

        NMmin = int(np.nanmin(binM_MC))
        NMmax = int(np.nanmax(binM_MC))

        #print(Nzmin, Nzmax, NMmin, NMmax)

        nb.set_num_threads(int(self.config_tile.nprocs))

        print('init numba func')
        _, _ = quantile_sig_mc(sig_indiv, binz_MC, Nz, 0, 1, 
                               binM_MC, NM, 0, 1, 
                               idx_lgmass_lim, quantile, 1)
        print('run get sig')
        sig_Mz, sig_z = quantile_sig_mc(sig_indiv, binz_MC, Nz, Nzmin, Nzmax, 
                                        binM_MC, NM, NMmin, NMmax, 
                                        idx_lgmass_lim, quantile, Nmc)
        print('done')
        
        kernel1d = Gaussian1DKernel(1)
        sig_z = convolve(sig_z, kernel1d)
        kernel2d = Gaussian2DKernel(1)
        sig_Mz = convolve(sig_Mz, kernel2d)

        return sig_Mz, sig_z

                
    def get_sig(self):
        
        avg,nprocs,fit_Olga = self.config_tile.avg, self.config_tile.nprocs, self.config_tile.fit_Olga
    
        sig_Mz = np.empty(2,dtype='object')
        sig_z = np.empty(2,dtype='object')
        
        
        self.rootdir = self.config_tile.tiles_rootdir + self.config_tile.field + '/' + self.config_tile.release
        

        psig='z'
        for i,conflim in enumerate([self.config_tile.conflim_1sig]):
            #for j,psig in enumerate(['z']): #,'Mass']):
            if fit_Olga:
                self.sig_Mzf = self.rootdir+'/sig'+psig+conflim+'.Mz.'+self.config_tile.field+'.fitOlga.'+avg+'.npz'
                self.sig_zf = (self.rootdir+'/sig'+psig+conflim+'.z.'+self.config_tile.field+
                      '.fitOlga.'+avg+'.npz')
            else:
                self.sig_Mzf = self.rootdir+'/sig'+psig+conflim+'.Mz.'+self.config_tile.field+'.MC.'+avg+'.mag90.npz'
                self.sig_zf = (self.rootdir+'/sig'+psig+conflim+'.z.'+self.config_tile.field+
                      '.MC.'+avg+'.mag90.Mlim'+str(np.round(self.config_tile.lgmass_lim,2))+'.npz')
                
            if Path(self.sig_Mzf).is_file() and Path(self.sig_zf).is_file():
                sig_Mz[i] = np.load(self.sig_Mzf)['sig']
                sig_z[i] = np.load(self.sig_zf)['sig']
                sig_Mz[i] = np.maximum(0.01, sig_Mz[i])
                sig_z[i] = np.maximum(0.01, sig_z[i])
            else:
                ### we don't care about uncertainty on sig_z,
                ### so we can run only on 2 MC realisations of the PDFs
                if fit_Olga:
                    #sig_Mz[i], sig_z[i] = self.compute_sig_fitOlga(conflim,psig,avg)
                    print('fit Olga -- this is done in each tile !')
                else:
                    sig_Mz[i], sig_z[i] = self.compute_sig_MC(conflim,psig,avg,nprocs,2)
                    
                sig_Mz[i] = np.maximum(0.01, sig_Mz[i])
                sig_z[i] = np.maximum(0.01, sig_z[i])
                np.savez(self.sig_Mzf,sig=sig_Mz[i])
                np.savez(self.sig_zf,sig=sig_z[i])
                
        #sigz68_Mz, sigM68_Mz, sigz95_Mz, sigM95_Mz = sig_Mz.flatten()
        #sigz68_z, sigM68_z, sigz95_z, sigM95_z = sig_z.flatten()
        sigz68_Mz, sigz95_Mz = sig_Mz#.flatten()
        sigz68_z, sigz95_z = sig_z#.flatten()   


        sigz0 = 0.01 ##backward compatibility
            
        return sigz68_Mz,sigz95_Mz,sigz68_z,sigz95_z,sigz0
    
    
    def run_get_sig(self):
        
        self.config_tile.galcatfile = ( self.config_tile.datadir+
                                       self.config_tile.field+
                                       '/'+
                                       self.config_tile.field+
                                       '_for_release_'+
                                       self.config_tile.release+
                                        '.DETECTIFz.galcat.fits'
                                      )
        
        
        self.galcat = Table.read(self.config_tile.galcatfile)
    
        self.galcat.rename_columns([self.config_tile.z_l68_colname, 
                                self.config_tile.z_u68_colname],
                           ['z_l68', 
                            'z_u68'])
    
        self.config_tile.galcat_mcfile = (self.config_tile.datadir+
                                       self.config_tile.field+
                                       '/samples/'+
                                       self.config_tile.field+
                                       '_for_release_'+
                                       self.config_tile.release+
                                        '.DETECTIFz.samples.fits'
                                         )
    
        tMz = Table.read(self.config_tile.galcat_mcfile)
        if not('lgM_samples' in tMz.colnames):
                tMz['lgM_samples'] = np.repeat(self.galcat['Mass_median'], 
                                                          self.config_tile.nMC).reshape((len(tMz), 
                                                                                         self.config_tile.nMC))
                self.has_lgM = False
                
                
        zmc = tMz['Z_SAMPLES']
        Mmc = tMz['lgM_samples']
    
        self.galcat_mc = np.stack([zmc,Mmc]).T
                
        
        
                ###get 1D PDFs (z and M)
        self.zz = np.arange(self.config_tile.zmin_pdf, 
                            self.config_tile.zmax_pdf+self.config_tile.pdz_dz, 
                            self.config_tile.pdz_dz)
        self.MM = np.arange(self.config_tile.Mmin_pdf, 
                            self.config_tile.Mmax_pdf+self.config_tile.pdM_dM, 
                            self.config_tile.pdM_dM)

        
        
        
        self.get_sig()
    
    
            
    def run_tiling(self):
        
        self.run_get_sig()
        
        self.config_tile.galcatfile = ( self.config_tile.datadir+
                                       self.config_tile.field+
                                       '/'+
                                       self.config_tile.field+
                                       '_for_release_'+
                                       self.config_tile.release+
                                        '.DETECTIFz.galcat.fits'
                                      )
        
        galcat_main = Table.read(self.config_tile.galcatfile)
        
        ## before tiling, get sig_MC
        #self.get_sig()

        for i, tile in enumerate(self.tiles):
            (rainf, rasup, 
             decinf, decsup) = (
                tile.bottom_left[0], tile.top_right[0], 
                tile.bottom_left[1], tile.top_right[1])
            
            ##GALAXY CATALOGUE
            maskgal_tile  = ((galcat_main[self.config_tile.ra_colname] > rainf) &
                             (galcat_main[self.config_tile.ra_colname] <= rasup) & 
                             (galcat_main[self.config_tile.dec_colname] > decinf) &
                             (galcat_main[self.config_tile.dec_colname] <= decsup))
            
            tile.galcat = galcat_main[maskgal_tile]
            #tile.galcat_raw_filename = 'tile'
            #if self.config_tile.Mz_MC_exists:
                
            self.config_tile.Mz_MC_file = ( self.config_tile.datadir+
                                       self.config_tile.field+
                                       '/samples/'+
                                       self.config_tile.field+
                                       '_for_release_'+
                                       self.config_tile.release+
                                        '.DETECTIFz.samples.fits'
                                      )
            ### for tests    
            #tile.Mz_MC = Table(np.moveaxis(
            #    np.load(self.config_tile.Mz_MC_file)['Mz'][maskgal_tile],
            #    2,
            #    1),
            #                   names=['lgM_samples', 'z_samples'])
            ##final version should be :
            tile.Mz_MC = Table.read(self.config_tile.Mz_MC_file)[maskgal_tile]
            if not('lgM_samples' in tile.Mz_MC.colnames):
                #tile.Mz_MC['lgM_samples'] = -99 * np.ones((len(tile.Mz_MC), self.config_tile.nMC))
                tile.Mz_MC['lgM_samples'] = np.repeat(tile.galcat['Mass_median'], 
                                               self.config_tile.nMC).reshape(
                    (len(tile.Mz_MC), self.config_tile.nMC))
                
            tile.thistile_dir = tile.tilesdir+'/tile'+'{:04d}'.format(tile.id)
            tile.master_masksfile = self.config_tile.masksfile
            tile.pixdeg = self.config_tile.pixdeg
            
            if not Path(tile.tilesdir).is_dir():
                #process = subprocess.Popen(["mkdir", self.tilesdir])
                #process.wait()
                #result = process.communicate()
                raise OSError('working directory does not exists, '+
                          'you should create it and put'+
                          'a config_detectifz_master.py file inside')
            
            if not Path(tile.thistile_dir).is_dir():
                process = subprocess.Popen(["mkdir", tile.thistile_dir])
                process.wait()
                result = process.communicate()
            
            
            tile.galcat_filename = tile.thistile_dir+"/galaxies."+tile.field+".galcat.fits"
            tile.galcat.write(tile.galcat_filename, overwrite=True) 
            
            #if self.config_tile.Mz_MC_exists:
            tile.Mz_MC_filename = ( tile.thistile_dir+"/galaxies."+tile.field+"."+
                                       str(int(self.config_tile.nMC))+"MC.Mz.fits" )
            
            tile.Mz_MC.write(tile.Mz_MC_filename, overwrite=True)
            
            if self.config_tile.maskstype == 'ds9' :
                tile.run_venice_pixelize()
                
            elif self.config_tile.maskstype == 'fits' :
                tile.run_cutout()
                
            elif self.config_tile.maskstype == 'none' :
                f = open(self.config_tile.datadir+
                         self.config_tile.field+
                         '/none.reg', "w")
                f.write("# FILTER HSC-G\n")
                f.write("wcs; fk5\n")
                f.write("circle("+str(np.median(tile.galcat[self.config_tile.ra_colname]))+
                        ","+str(np.median(tile.galcat[self.config_tile.dec_colname]))+
                        ",0.00000001d)")
                f.close()
                tile.master_masksfile = (self.config_tile.datadir+
                                         self.config_tile.field+
                                         '/none.reg')
                
                tile.run_venice_pixelize()
                
                
    
            #tile.run_venice_inout()
            
            tile.write_config_detectifz()
            
            np.savez(tile.thistile_dir+'/tile_object_init.npz', 
                     tileid=tile.id, 
                     corners=tile.corners, 
                     border_width = tile.border_width,
                     field=tile.field,
                     tilesdir = tile.tilesdir)
            
            
            process = subprocess.Popen(["cp", self.sig_Mzf, tile.thistile_dir])
            process.wait()
            result = process.communicate()
            
            process = subprocess.Popen(["cp", self.sig_zf, tile.thistile_dir])
            process.wait()
            result = process.communicate()
            
            
#        
#        ###DETECTIFz CONFIG FILE
#
#    
#    #def save_tiling(self):
#            
#class Tiles(object):
#    
#    def __init__(self, field='UDS', params=None):
#        self.field = field
#        self.data = data
#        self.params = params
 
