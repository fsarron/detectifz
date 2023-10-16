#!/bin/bash

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# use here your expected variables
echo "ROOTDIR = $ROOTDIR"
echo "FIELD = $FIELD"
echo "RELEASE = $RELEASE"
echo "SCRIPTS_DIR = $SCRIPTS_DIR"

echo "EXTRA_VALUES = $EXTRA_VALUES"

#detectifz_dir = ${1}
#detectifz_exec = ${2}

##TO DO PRIOR TO LAUNCH
WORKDIR=$ROOTDIR/$FIELD/$RELEASE
mkdir $ROOTDIR/$FIELD
mkdir $WORKDIR
cp $SCRIPTS_DIR/config_tile_WP11.ini $WORKDIR/config_tile.ini
cp $SCRIPTS_DIR/config_detectifz_WP11.ini $WORKDIR/config_master_detectifz.ini
cp $SCRIPTS_DIR/torque_run_detectifz_tile.sh $WORKDIR/

cd $WORKDIR


#make_input_Euclid --configuration=config_tile.ini

#run_tiling --configuration=config_tile.ini

for i in $(ls -d */)
do 
    tileid=`echo ${i%%/} | awk -F / '{print $2}'`
    echo $tileid
    tiledir=$WORKDIR/${i}
    cd $tiledir
    qsub $WORKDIR/torque_run_detectifz_tile.sh -v 'tiledir='$tiledir', config_file=config_detectifz.ini, tile='$tilei
done





###################################



#i=19
#LISTTILES=`ls ${field}_DETECTIFzrun/tile* | grep tile`
#echo $LISTTILES
#lastTile=${LISTTILES: -5:4}
#echo $lastTile

#for i in {0000..$lastTile}
#for (( c=0; c<=$lastTile+2; c++ ))
#do
#   echo "Welcome $c times"
#done


#echo $lastFour

#tiles=`echo $LISTTILES | awk -F : '{print $0}'`;





#tiles=`echo $LISTTILES | awk -F : '{print $'$i'}'`;

#files=`ls -l ${field}_DETECTIFzrun/tile* | awk '{print $1}'`
#echo $tiles


#for filter in $FILTERS_IRAC; do
#    echo ""
#    echo "convolve KERNEL IRAC"
#    echo $filter
#    python $python_exec $rootdir $cluster $filter $tag $psftargetsize_IRAC
#done