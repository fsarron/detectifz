## How to run DETECTIFz on your data

# Requirements : 

For a small enough field (tipically $< 10 {\rm deg^2}$), you can run DETECTIFz witout tiling. To do so, you need to : 

## 1. Create a directory for the run ```detectifz_run```

```bash
mkdir a/dir/in/your/path/detectifz_run/
```

## 2. Add the config_detectifz_notiling.ini in ```detectifz_run```

```bash
cp /your/path/to/detectifz/scripts/configs/config_detectifz_notiling.ini a/dir/in/your/path/detectifz_run/
```

Modify the parameters for the run at your convenience in the config file

## 3. Add the necessary data files in ```detectifz_run```
- Galaxy catalogue :
    This is a FITS Table containing at least columns listed in the config_detectifz_notiling.ini (entry ```gal_id_colname``` to ```M_u68_colname```)
	
- Galaxy redshift/mass **samples** file containing samples of p(M, z) or p(z) for each galaxy:
    This is a FITS Table containing at least a column names 'z_samples', and optionally 'lgM_samples'
	
### Important notes:
	
- In most situations, the prefered configuration of DETECTIFz should be to use the stellar-mass density based on samples of p(M, z).
- The default fallback when there are no 'lgM_samples' in in the samples file is to make a 'lgM_samples', where we set set the realisation lgM to the M_med_colname for all realisations. 
-  As this is not ideal (we do not account for stellar mass uncertianty nor its correlation with redshift), in this case we stringlt suggest setting ```use_mass_density = False``` in the config file i.e. to ask detectifz to use galaxy density.

##
##
        
# Running detectifz

Once these requirements are fullfiled, simply do
```
cd a/dir/in/your/path/detectifz_run/

run_detectifz -c config_detectifz_notiling.ini
```

This will output sevral files, most notably : 
- 3D oversensity map as a FITS cube
- detection catalogue
- detection catalogue p(z)

In this version, the probability membership computation has been removed for the time being (due to a change in the data model used)