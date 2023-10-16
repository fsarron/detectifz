rootdir = '/Volumes/Expansion/lance_detectifz/GAEA/GAEA_ECLQ'
field = 'GAEA_ECLQ'
tilesdir = rootdir+'/'+field+'_DETECTIFzrun'

galcatfile = rootdir+'/'+field+'_data/GAEA_ECLQ_SDR3_DDP_for_DETECTIFz.fits'

masksfile = rootdir+'/'+field+'_data/none.reg'
maskstype = 'none'  #'ds9' (when .reg) or 'fits'

Mz_MC_exists = True #False
Mz_MC_file = rootdir+'/'+field+'_data/GAEA_ECLQ_samples_Mz_for_release_SDR3_DDP.npz'
nMC = 100

ra_colname = 'RIGHT_ASCENSION'
dec_colname = 'DECLINATION'

sky_lims = [-2.6,2.6,-2.6,2.6]
max_area = 2.0 ##deg2
border_width = 0.05 #deg
pixdeg = 0.0008