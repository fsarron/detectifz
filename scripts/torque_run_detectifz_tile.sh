#!/bin/sh
#PBS -S /bin/sh
#PBS -N detectifz_tile
#PBS -j oe
#PBS -l nodes=1:ppn=8,walltime=10:00:00

module purge

source ~/.bash_profile

conda activate detectifz

cd $tiledir

run_detectifz_tile --configuration=$config_file --tile=$tile

exit 0
