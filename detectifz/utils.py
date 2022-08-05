import numpy as np
import astropy.cosmology
from astropy import units
from pinky import Pinky
import ray

# cosmo


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


def physep_ang(z, radius, H0=70.0, Om0=0.3):
    """Compute the physical separation in Mpc corresponding to an angular
    separation (radius) in degrees for a source at redshift z in a
    FlatLambdaCDM cosmology.

    Parameters
    ----------
    z: float, redshift
    radius: float, angular separation in degrees
    H0: float, Hubble constant
    Om0: float, total matter density parameter

    Returns
    -------
    astropy.Quantity, the physical separation in Mpc
    """
    cosmo = astropy.cosmology.FlatLambdaCDM(H0=H0, Om0=Om0)
    return (((radius) * units.deg) / cosmo.arcsec_per_kpc_proper(z)).to(units.Mpc)


# MC sampling
@ray.remote(max_calls=100)
def MCsampling(i, pdMz_id, Nmc, ext):
    """Compute Nmc realizations of one PDF(M,z) using Pinky.

    Parameters
    ----------
    i: integer, index of line in file
    pdMz_id: pointer, reference to access h5py "pdf_mass_z" array
    Nmc: integer, number of Monte Carlo realizations
    ext: list or np.array, extent of the distribution in mass and redshift
            e.g. [zz[0], zz[-1], MM[0], MM[-1]] i.e. [0., 5., 5.025, 12.975]

    Returns
    -------
    Mz: 2D np.array, size=(Nmc,2), array of MC sampled (z, M).
    """
    ppMz = np.copy(pdMz_id[i]).T
    ppMz = ppMz / np.sum(ppMz)
    Mz = Pinky(P=ppMz, extent=ext).sample(Nmc, r=10)
    return Mz


# nans
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Parameters
    ----------
    y: 1d numpy array with possible NaNs

    Returns
    -------
    nans: logical indices of NaNs
    index: a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices

    Example
    -------
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


# stats
def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)



def Mlim_DETECTIFz(f_logMlim, masslim, z):
    """Either 90 per cent stellar mass completennes limit at a given redshift z in REFINE using Mundy+17 (2017MNRAS.470.3507M) values, or constant value "masslim" if completeness is above masslim "value". 90 per cent stellar mass completennes limit can be obtained at all redshifts calling with masslim=-1.

    Parameters
    ----------
    f_logMlim: scipy.interpolate.intrep1d object
    masslim: float, minimum mass for cut higher than 90 per cent completeness
    z: float or np.array, redshift
    Returns
    -------
    float or np.array, maximum of masslim and 90 per cent completeness limit at each redshift z
    """

    return np.maximum(f_logMlim(z),masslim)




# REFINE specific
#def Mlim(field, masslim, z):
#    """Either 90 per cent stellar mass completennes limit at a given redshift z in REFINE using Mundy+17 (2017MNRAS.470.3507M) values, or constant value "masslim" if completeness is above masslim "value". 90 per cent stellar mass completennes limit can be obtained at all redshifts calling with masslim=-1.
#
#    Parameters
#    ----------
#    field: str, name of the field, accepted values are 'UDS', 'H15_UDS', 'UltraVISTA', 'H15_UltraVISTA', 'VIDEO', or 'H15_VIDEO'
#    masslim: float, minimum mass for cut higher than 90 per cent completeness
#    z: float or np.array, redshift
#    Returns
#    -------
#    float or np.array, maximum of masslim and 90 per cent completeness limit at each redshift z
#    """
#    if field == 'UDS' or field == 'H15_UDS':
#        Mlim = 7.847 + 1.257*z - 0.150*z**2
#    if field == 'UltraVISTA' or field == 'H15_UltraVISTA':
#        Mlim = 8.378 + 1.262*z - 0.153*z**2
#    if field == 'VIDEO' or field == 'H15_VIDEO':
#        Mlim = 8.455 + 1.697*z - 0.278*z**2
#    else:
#        raise ValueError('field'+field+' is not implemented !')
#    return np.maximum(Mlim,masslim)

