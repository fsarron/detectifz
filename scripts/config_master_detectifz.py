lgmass_lim = 10
zmin = 0.1
zmax = 3.5
dzslice = 0.01
dzmap = 0.01
pixdeg = 0.0008
nMC = 100
SNmin = 1.5
dclean = 0.5
radnoclus = 2
avg = '68p'
nprocs = 6

Mz_MC_exists = False #True/False (bool)
pdz_datatype = 'PDF' #'PDF', 'samples', 'quantiles', 'tpnorm'
pdM_datatype = 'PDF' #'PDF', 'samples', 'quantiles', 'tpnorm'

datatypes = [Mz_MC_exists, pdz_datatype, pdM_datatype]

zmin_pdf, zmax_pdf = 0.01, 5.0
pdz_dz = 0.01

Mmin_pdf, Mmax_pdf = 5.0, 13.0
pdM_dM = 0.05

pdf_Mz = False ### always false, keep so far for backward compatibility

#field='UDS'
#root_filename='/Users/fsarron/DETECTIFz_data/'+field+'.master.PDF_Mz.irac'

