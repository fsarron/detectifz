obsmag90 = 24.3
lgmass_comp = 0.9
lgmass_lim = 10.
zmin = 0.1
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

gal_id_colname = 'id'
ra_colname = 'ra'
dec_colname = 'dec'
obsmag_colname = 'mK_tot'
z_med_colname = 'z'
z_l68_colname = 'z_l68'
z_u68_colname = 'z_u68'
M_med_colname = 'Mass_median'
M_l68_colname = 'Mass_l68'
M_u68_colname = 'Mass_u68'


Mz_MC_exists = True #True/False (bool)
pdz_datatype = 'PDF' #'PDF', 'samples', 'quantiles', 'tpnorm'
pdM_datatype = 'PDF' #'PDF', 'samples', 'quantiles', 'tpnorm'

datatypes = [Mz_MC_exists, pdz_datatype, pdM_datatype]

zmin_pdf, zmax_pdf = 0.01, 5.0
pdz_dz = 0.01

Mmin_pdf, Mmax_pdf = 5.025, 12.975
pdM_dM = 0.05

pdf_Mz = False ### always false, keep so far for backward compatibility
