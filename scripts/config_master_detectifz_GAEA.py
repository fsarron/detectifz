lgmass_comp = 0.9
lgmass_lim = 10.4
zmin = 1.25
zmax = 3.5
dzslice = 0.01
dzmap = 0.01
pixdeg = 0.0008
Nmc = 100
Nmc_fast = 25
SNmin = 1.5
dclean = 0.5
radnoclus = 2
avg = '68p'
nprocs = 6


gal_id_colname = 'GALNUM'
ra_colname = 'RIGHT_ASCENSION'
dec_colname = 'DECLINATION'
obsmag_colname = 'obs_magH'
z_med_colname = 'z_median'
z_l68_colname = 'z_l70'
z_u68_colname = 'z_u70'
M_med_colname = 'Mass_median'
M_l68_colname = 'Mass_l68'
M_u68_colname = 'Mass_u68'


Mz_MC_exists = True #True/False (bool)
pdz_datatype = 'samples' #'PDF', 'samples', 'quantiles', 'tpnorm'
pdM_datatype = 'samples' #'PDF', 'samples', 'quantiles', 'tpnorm'

datatypes = [Mz_MC_exists, pdz_datatype, pdM_datatype]

zmin_pdf, zmax_pdf = 0.00, 6.0
pdz_dz = 0.01

Mmin_pdf, Mmax_pdf = 5.0, 13.0
pdM_dM = 0.05

pdf_Mz = False ### always false, keep so far for backward compatibility
