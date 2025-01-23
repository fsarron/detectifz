#import sys
import numpy as np
import multiprocessing
from pathlib import Path
from astropy.table import Table
import scipy.stats
import numba as nb
import h5py
from scipy.interpolate import interp1d
from numpy.random import RandomState

from pathlib import Path

#rootdir = '/Users/tdusserre/Documents/Thèse/PYTHON/Q1_Protoclusters/detectifz_data'
#field = 'G252.47-48.60'


# @nb.njit()
# def rvs(pdf, x, xmin, xmax, size=1):
#     cdf = pdf  # in place
#     cdf[1:] = np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(x)) #, out=cdf[1:])
#     cdf[0] = 0
#     cdf /= cdf[-1]

#     t_lower = np.interp(xmin, x, cdf)
#     t_upper = np.interp(xmax, x, cdf)
#     u = np.random.uniform(t_lower, t_upper, size=size)
#     samples = np.interp(u, cdf, x)
#     return samples

# @nb.njit(parallel=True)
# def sample_tpdf(tpdf_z, zz, size):
#     np.random.seed(123456)
#     N = len(tpdf_z)
#     zmin, zmax = zz.min(), zz.max()
#     samples = np.zeros((N, size))
#     for i in nb.prange(N):
#         samples[i] = rvs(tpdf_z[i], zz[i], zmin, zmax, size=size)
#     return samples

def rvs(pdf, x, xmin, xmax, size=1):
    pdf = pdf # Cumulative sum takes place on the first axis and it must be done along zz
    cdf = pdf  # in place
    cdf[1:] = np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(x),axis=0) # Integration of pdf = cdf
    cdf[0] = 0
    cdf /= cdf[-1] # Normalization of cdf so that it's bijective from x to [0;1]

    t_lower = interp1d(x, cdf)(xmin)
    t_upper = interp1d(x, cdf)(xmax)
    u = np.random.uniform(0, 1, size=size)
    samples = np.interp(u,cdf,x)# Value of x matching cdf = random between 0 and 1 ( x = cdf^-1(u) )
    # samples = interp1d(cdf, x,axis=1)(u) 
    return samples

def sample_tpdf(tpdf_z, zz, size):
    np.random.seed(123456)
    N = len(tpdf_z)
    zmin, zmax = zz.min(), zz.max()
    samples = np.zeros((N, size))
    # samples = rvs(tpdf_z, zz, zmin, zmax, size=size)
    for i in range(N):
        samples[i] = rvs(tpdf_z[i], zz, zmin, zmax, size=size) # this originally indicated zz[i], which could not work inside rvs function
    return samples

ncpus = int(multiprocessing.cpu_count())
nb.set_num_threads(ncpus)

def run_samples(config,Nmc):

    rootdir = config.rootdir
    field = config.field
    PDFz_colname = config.PDFz_colname
    datadir = Path(rootdir)
    table_pdzf = datadir.joinpath('galaxies.'+field+'.galcat.fits')
    reduced_table_path = datadir.joinpath(rootdir+'/galaxies.'+field+'.'+str(Nmc)+'MC.Mz.fits')

    if Path(reduced_table_path).is_file():
        print('')
        print('Samples already created.')
        print('')

    else:

        ### redshift 
        print('read table...')
        table_pdz = Table.read(table_pdzf,format='ascii.ecsv')
        print('done.')
            
        print('convert to array...')
        tpdf_z = np.array(table_pdz[PDFz_colname]).astype(np.float32)
        print('done.')

        # Redshift over which PDF is sampled (based on Euclid mission)

        zz = np.arange(0,6.01,0.01).astype(np.float32) #np.array(table_pdz['Z'], np.float32)

        # Nmc = 100

        #np.random.seed(123456)
        #RandomState(123456)

        samples_z  = sample_tpdf(tpdf_z, zz, Nmc)

        t_samples = table_pdz
        #t_samples.remove_columns(['p_z_normalised', 'p_z', 'm_z'])


        t_samples['Z_SAMPLES'] = samples_z


        # t_samples['lgM_SAMPLES'] = np.array([np.interp(t_samples['Z_SAMPLES'][igal], 
        #                                   table_pdz['Z'][igal], 
        #                                   table_pdz['m_z'][igal])
        #                         for igal in range(len(t_samples))])

        # This line creates a mock and simple logmass pdf since Euclid does not provide any
        t_samples['lgM_SAMPLES'] = np.ones((len(t_samples), Nmc))

        t_samples.keep_columns(['OBJECT_ID',
                                'RIGHT_ASCENSION',
                                'DECLINATION', 
                                'Z_SAMPLES',
                                'lgM_SAMPLES'])

        t_samples.write(reduced_table_path, overwrite=True)
        #should be named 
        #self.rootdir+'/galaxies.'+self.field+'.'+str(self.Nmc)+'MC.Mz.fits'
        #    or 
        #self.rootdir+'/galaxies.'+self.field+'.'+str(self.Nmc)+'MC.Mz.masked_m90.fits'

    return




