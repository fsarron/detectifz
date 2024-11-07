import numpy as np
from astropy.table import Table, vstack

import astropy.cosmology 
from astropy import units

from scipy.spatial.transform import Rotation


from astropy.coordinates import SkyCoord

def angsep_radius(z, radius, H0=70.0, Om0=0.3):
    """Compute the angular separation in degrees corresponding to a physical
    seration (radius) in Mpc for a source at redshift z in a
    FlatLambdaCDM cosmology.

    Parameters
    ----------
    z: float, redshift
    radius: float, physical separation in Mpc
    H0: float, Hubble constant
    Om0: float, total matter density parameter

    Returns
    -------
    astropy.Quantity, the angular separation in degrees
    """
    cosmo = astropy.cosmology.FlatLambdaCDM(H0=H0, Om0=Om0)
    return (((radius * 1000) * units.kpc) * cosmo.arcsec_per_kpc_proper(z)).to(
        units.deg
    )


def detectifz2radec(skycoords_center, detectifz_coords):
    '''
    Rotate the sphere so that the tile centered at (ra, dec) = (0, 0) 
    gets back to its original coordinate frame
    The forward rotation is roughly equivalent to the approximate multiplication of RA by cos(Dec)
    used in e.g. AMASCFI, but is more mathematiocaaly justified and avoid approximation.
    The rotation idea was taken from discussions in (and strategy adopted by) 
    the OU-LE3-CL of the Euclid consortium
    '''
    
    detectifz_x, detectifz_y = detectifz_coords
    #convert skycoords_center, detectifzcoords_galaxies (ra, dec) to (phi, theta)_radians
    phi = np.deg2rad(detectifz_x)
    theta = np.pi/2. - np.deg2rad(detectifz_y) # theta == pi/2 - dec

    phi_c = skycoords_center.ra.radian
    theta_c = skycoords_center.dec.radian  ## here theta == dec because this is just 
                                         ###the angle between (0,0) ad (ra_c, dec_c)


    #rz, ry from https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    rz = Rotation.from_matrix([[np.cos(phi_c), np.sin(phi_c), 0.0],
                     [-np.sin(phi_c), np.cos(phi_c), 0.0],
                     [0.0, 0.0, 1.0]])  ##Rz (!! numpy transpose) -- rotation by theta_c around z axis
    ry = Rotation.from_matrix([[np.cos(theta_c), 0.0, np.sin(theta_c)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(theta_c), 0.0, np.cos(theta_c)]]) ##Ry (!! numpy transpose) --rotation by phi_c around y
    
    rot = (ry * rz).inv() #inverse of full rotation 
    
    detectifz_xyz = np.array([np.sin(theta)*np.cos(phi),
                   np.sin(phi)*np.sin(theta),
                   np.cos(theta)]).T  ##[x,y,z] in original spherical coordinates  
                                       
    (original_x, 
     original_y, 
     original_z ) = rot.apply(detectifz_xyz).T ##apply rotation to original [x,y,x] to get [x,y,z]_rot in rotated frame

    original_phi = np.arctan2(original_y, original_x)  ## get phi in rotated frame from [x,y,z]_rot
    original_theta = np.arctan2(np.sqrt(original_x**2 + 
                                       original_y**2), original_z) ## get theta in rotated frame from [x,y,z]_rot
    
    original_phi[original_phi < 0] += 2. * np.pi
    
    return np.rad2deg(original_phi), np.rad2deg(np.pi / 2. - original_theta) # dec == pi/2 - theta




cosmo = astropy.cosmology.FlatLambdaCDM(H0=73., Om0=0.25)

rootdir = '/data80/sarron/detectifz_runs/EUCLID_WP11/GAEA_ECLQ/SDR3_DDP/'

#tmaster0= Table.read(rootdir + 'tile0000' + '/candidats_GAEA_ECLQ_SN1.5_Mlim10.25.sigz68_z_50p.r200.clean.fits')

tmaster0 = Table()

for tile in ['tile0000',
             'tile0001',
             'tile0002',
             'tile0003',
             'tile0004',
             'tile0005',
             'tile0006',
             'tile0007',
             'tile0008',
             'tile0009',
             'tile0010',
             'tile0011',
             'tile0012',
             'tile0013',
             'tile0014',
             'tile0015'] :
    thisdir = rootdir + tile + '/'
    
    t = Table.read(thisdir + 'candidats_GAEA_ECLQ_SN1.5_Mlim10.25.sigz68_z_50p.fits')
    
    ra_c = np.load(thisdir + 'skycoords_center.npz')['ra']
    if ra_c > 180. :
        ra_c -= 360.0
    dec_c = np.load(thisdir + 'skycoords_center.npz')['dec']
    
    t.rename_columns(['ra', 'dec'], ['ra_detectifz', 'dec_detectifz'])
    
    skycoords_center = SkyCoord(ra=ra_c, dec=dec_c, unit='deg')
    
    detectifz_coords = (t['ra_detectifz'], t['dec_detectifz'])
    
    t['ra'], t['dec'] = detectifz2radec(skycoords_center, detectifz_coords)
    tmaster0 = vstack([tmaster0, t])
    
tmaster = Table()
tmaster['ID_over'] = np.arange(len(tmaster0))
tmaster['RA_over'] = tmaster0['ra']
tmaster['RA_over'][tmaster['RA_over'] > 180] -= 360

tmaster['Dec_over'] = tmaster0['dec']
tmaster['z_over'] = tmaster0['z']
tmaster['SN'] = tmaster0['SN']
tmaster['Membership'] = -99 * np.ones(len(tmaster))
tmaster['size'] = angsep_radius(tmaster['z_over'], tmaster0['rMpc_subdets'].value).to(units.arcmin)

#master['size'] = angsep_radius(tmaster['z_over'], tmaster0['R200c_Mass_median']).to(units.arcmin)

tmaster.write(rootdir + 'detections_DETECTIFz_GAEA_ECLQ_SDR3_DDP_241023.fits')

    
    





