#include<stdio.h>
#include<string.h>
#include<stdlib.h>
// Simple executable that creates the appropriate config file for a given field name

int main(int argc, char **argv){
    FILE *config;

    // Creating the file name

    char configdir[100] = "/home/tdusserr/detectifz-cluster-head/scripts/configs/"; // for glx-calcul3
    // char configdir[100] = "/Users/tdusserre/Documents/Thèse/PYTHON/Q1_Protoclusters/detectifz-cleaning/scripts/configs/";
    char intr[50] = "config_detectifz_notiling_";
    char* field = argv[1];
    char ext[50] = ".ini";
    char filename[250];

    strcpy(filename,configdir);
    strcat(filename,intr);
    strcat(filename,field);
    strcat(filename,ext);

    // Writing the file

    config = fopen(filename,"w");

    fprintf(config,
        "[DETECTIFz] \n\n"

        ";groups / protoclusters \n"
        "objects_detected = protoclusters \n"
        "use_mass_density = False \n\n"

        ";if you have no mask file, just use none \n"
        "input_masksfile = none \n\n"

        ";'ds9' (when .reg) / 'fits' or none \n"
        "maskstype = none  \n\n"

        "obsmag_lim = 24 \n"
        "lgmass_comp = 0.90 \n"
        "lgmass_lim = 0 \n"
        "zmin = 1.35 \n"
        "zmax = 3.5 \n"
        "dzmap = 0.01 \n"
        "pixdeg = 0.0016 \n"
        "Nmc = 100 \n"
        "SNmin = 1.5 \n"
        "dclean = 0.5 \n"
        "radnoclus = 2 \n"
        "conflim_1sig = 68 \n"
        "conflim_2sig = 95 \n"
        "avg = 50p \n"
        "nprocs = 1 \n\n"

        "gal_id_colname = OBJECT_ID \n"
        "ra_colname = RIGHT_ASCENSION \n"
        "dec_colname = DECLINATION \n"
        "obsmag_colname = obs_magH \n"
        "z_med_colname = PHZ_MEDIAN \n"
        "z_l68_colname = PHZ_70_INT_L \n"
        "z_u68_colname = PHZ_70_INT_U \n"
        "PDFz_colname = PHZ_PDF \n"
        "M_med_colname = Mass_median \n"
        "M_l68_colname = Mass_l70 \n"
        "M_u68_colname = Mass_u70 \n\n"

        "zmin_pdf = 0.00 \n"
        "zmax_pdf = 6.0 \n"
        "pdz_dz = 0.01 \n\n"

        "Mmin_pdf = 5.0 \n"
        "Mmax_pdf = 13.0 \n"
        "pdM_dM = 0.05 \n\n"

        "; 'maglim', 'masslim' or 'legacy' \n"
        "selection = masslim \n\n"

        ";used for protoclusters detection only \n"
        "r_equiv_min_cMpc = 0.0 \n"
        "n_subdets_min = 3 \n\n"


        "[MEMBERSHIPS] \n\n"

        ";PDFz_filename = '' \n"
        ";FITS, npz or None \n"
        "PDFz_filetype = FITS \n\n"


        "[GENERAL] \n"
        "field = %s \n"
        "release = Q1 \n"
        "rootdir = /data/cluster/EUCLID/Q1/detectifz-data \n"
        ,
        argv[1]
    );

    fclose(config);

    return 0;
}
