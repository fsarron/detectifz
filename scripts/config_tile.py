rootdir = '/Users/fsarron'
field = 'HSC_test'
tilesdir = rootdir+'/'+field+'_DETECTIFzrun' 

galcatfile = rootdir+'/'+field+'_data/234279.fits'
FITSmasksfile = rootdir+'/'+field+'_data/masks_subHSC.fits'


ra_colname = 'ra'
dec_colname = 'dec'

sky_lims = [30.0,35.0,-6.5,-3.5]
max_area = 1.0 ##deg2
border_width = 0.05 #deg
